import argparse

import os
import copy

import torch

torch.backends.cudnn.benchmark = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from tqdm import tqdm

from safetensors import safe_open


from src.config.config import base_model_name, bnb_config
from src.config.paths import (
    base_model_outputs_dir,
    fine_tuned_model_outputs_dir,
    models_outputs_dir,
    raw_datasets_dir,
    predicted_adapters_dir,
)
from src.data.LoRAs_Info import LoRAs_List

from peft.utils.save_and_load import set_peft_model_state_dict

from accelerate import Accelerator



def inference(prompts, model, tokenizer, device, max_input=4096, max_output=512):
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output_ids = model.generate(**batch, max_new_tokens=max_output)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def generate_outputs(metric, model, dataset_index, batch_size:int = 4):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available to use!")

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
    )
    base_model.eval().to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

    base_model_outputs_file_location = os.path.join(
        base_model_outputs_dir, f"{dataset_index}.pt"
    )

    gpu_fine_tuned_outputs_file_location = os.path.join(
        fine_tuned_model_outputs_dir, f"{dataset_index}.pt"
    )

    predicted_models_outputs_metric_dir = os.path.join(models_outputs_dir, metric)
    predicted_models_outputs_metric_model_dir = os.path.join(
        predicted_models_outputs_metric_dir, model
    )
    predicted_model_outputs_file_location = os.path.join(
        predicted_models_outputs_metric_model_dir, f"{dataset_index}.pt"
    )

    # Load dataset for this LoRA
    dataset_file_location = os.path.join(raw_datasets_dir, f"{dataset_index}.pt")
    dataset = torch.load(dataset_file_location, weights_only=False)
    test_data = dataset["test"]

    base_model.resize_token_embeddings(len(tokenizer))
    # Base model inference
    if not os.path.exists(base_model_outputs_file_location):
        print(f"Running base model inference for index: {dataset_index}")
        outputs = []
        for i in tqdm(range(0, len(test_data), batch_size),
                      desc=f"Base Model {dataset_index}"):
            batch_prompts = test_data[i : i + batch_size]["input"]
            gen_texts = inference(batch_prompts, base_model, tokenizer, device)
            for prompt, text in zip(test_data[i : i + batch_size], gen_texts):
                outputs.append([prompt, {"generated_text": [text]}])
        torch.save(outputs, base_model_outputs_file_location)

    # Load GPU fine tuned model
    print(f"Loading fine-tuned model for index: {dataset_index}")
    peft_name = LoRAs_List[dataset_index]
    try:
        peft_model = PeftModel.from_pretrained(base_model, peft_name)
    except Exception as e:
        raise RuntimeError(f"Error loading fine tuned model {dataset_index}: {e}")
    peft_model.eval().to(device)
    peft_model.resize_token_embeddings(len(tokenizer))

    # GPU fine tuned model inference
    if not os.path.exists(gpu_fine_tuned_outputs_file_location):
        print(f"Running GPU fine-tuned model inference for index: {dataset_index}")
        outputs = []
        for prompt in tqdm(
            dataset["test"], desc=f"GPU fine-tuned model {dataset_index}"
        ):
            result = inference(prompt, peft_model, tokenizer, device)
            outputs.append([prompt, result])
        torch.save(outputs, gpu_fine_tuned_outputs_file_location)

    if not os.path.exists(gpu_fine_tuned_outputs_file_location):
        print(f"Running GPU fine-tuned model inference for index: {dataset_index}")
        outputs = []
        for i in tqdm(range(0, len(test_data), batch_size),
                      desc=f"GPU fine-tuned model {dataset_index}"):
            batch_prompts = test_data[i : i + batch_size]["input"]
            gen_texts = inference(batch_prompts, peft_model, tokenizer, device)
            for prompt, text in zip(test_data[i :  i + batch_size], gen_texts):
                outputs.append([prompt, {"generated_text": [text]}])
        torch.save(outputs, gpu_fine_tuned_outputs_file_location)

    # Predicted model inference
    if not os.path.exists(predicted_model_outputs_file_location):
        print(f"Running predicted adapters model inference for index: {dataset_index}")

        # load and inject predicted adapters
        predicted_adapters_metric_dir = os.path.join(predicted_adapters_dir, metric)
        predicted_adapters_metric_model_dir = os.path.join(
            predicted_adapters_metric_dir, model
        )
        predicted_adapters_file_location = os.path.join(
            predicted_adapters_metric_model_dir,
            f"State_Dictionary_{dataset_index}.safetensors",
        )
        predicted_adapters = safe_open(predicted_adapters_file_location, "pt")
        predicted_adapters = {
            k: predicted_adapters.get_tensor(k) for k in predicted_adapters.keys()
        }
        set_peft_model_state_dict(peft_model, predicted_adapters)

        outputs = []
        for i in tqdm(range(0, len(test_data), batch_size),
                      desc=f"Predicted adapters model {dataset_index}"):
            batch_prompts = test_data[i : i + batch_size]["input"]
            gen_texts = inference(batch_prompts, peft_model, tokenizer, device)
            for prompt, text in zip(test_data[i : i + batch_size], gen_texts):
                outputs.append([prompt, {"generated_text": [text]}])
        torch.save(outputs, predicted_model_outputs_file_location)

    #### End of Run Print
    print(40 * "*")
    print("INFERENCE PART FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["WD", "KL", "JS", "MMD"])
    p.add_argument(
        "model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    p.add_argument("dataset_index", type=int)
    args = p.parse_args()
    generate_outputs(
        metric=args.metric, model=args.model, dataset_index=args.dataset_index
    )


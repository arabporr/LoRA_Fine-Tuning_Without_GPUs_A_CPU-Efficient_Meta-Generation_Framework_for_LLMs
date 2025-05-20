import argparse

import os
import copy

import torch

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


def inference(prompt, model, tokenizer, device, max_input=4096, max_output=256):
    input_ids = tokenizer(
        prompt["input"],
        padding=True,
        truncation=True,
        max_length=max_input,
        return_tensors="pt",
    ).to(device)
    output_ids = model.generate(**input_ids, max_new_tokens=max_output)
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return {"generated_text": generated_text}


def generate_outputs(metric, model, dataset_index):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available to use!")

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load base model and tokenizer
    org_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
    )
    org_base_model.eval()
    org_base_model.to("cpu")

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

    base_model = copy.deepcopy(org_base_model)
    base_model.to(device)

    # Load dataset for this LoRA
    dataset_file_location = os.path.join(raw_datasets_dir, f"{dataset_index}.pt")
    dataset = torch.load(dataset_file_location, weights_only=False)

    # Base model inference
    if not os.path.exists(base_model_outputs_file_location):
        print(f"Running base model inference for index: {dataset_index}")
        outputs = []
        for prompt in tqdm(dataset["test"], desc=f"Base Model {dataset_index}"):
            result = inference(prompt, base_model, tokenizer, device)
            outputs.append([prompt, result])
        torch.save(outputs, base_model_outputs_file_location)

    # Load GPU fine tuned model
    print(f"Loading fine tuned model for index: {dataset_index}")
    peft_model_name = LoRAs_List[dataset_index]
    try:
        peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading fine tuned model {dataset_index}: {str(e)}")
    peft_model = peft_model.to(device)
    peft_model.eval()

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

    # Predicted model inference
    if not os.path.exists(predicted_model_outputs_file_location):
        print(f"Running predicted adapters model inference for index: {dataset_index}")

        predicted_adapters_metric_dir = os.path.join(predicted_adapters_dir, metric)
        predicted_adapters_metric_model_dir = os.path.join(
            predicted_adapters_metric_dir, model
        )
        predicted_adapters_file_location = os.path.join(
            predicted_adapters_metric_model_dir,
            f"State_Dictionary{dataset_index}.safetensors",
        )
        predicted_adapters = safe_open(predicted_adapters_file_location, "pt")
        predicted_adapters = {
            k: predicted_adapters.get_tensor(k) for k in predicted_adapters.keys()
        }
        set_peft_model_state_dict(peft_model, predicted_adapters)

        outputs = []
        for prompt in tqdm(
            dataset["test"], desc=f"Predicted adapters model {dataset_index}"
        ):
            result = inference(prompt, peft_model, tokenizer, device)
            outputs.append([prompt, result])
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

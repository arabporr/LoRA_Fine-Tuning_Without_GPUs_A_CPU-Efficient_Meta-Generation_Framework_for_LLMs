import os
import gc
import wget
import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from datasets import load_dataset

from tqdm import tqdm

from safetensors import safe_open

from LoRAs_Info import *
from config import *

from peft.utils.save_and_load import (
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)

from accelerate import Accelerator

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


def inference(batch, model, tokenizer, device, max_input=16000, max_output=10):
    input_ids = tokenizer(
        batch["input"],
        padding=True,
        truncation=True,
        max_length=max_input,
        return_tensors="pt",
    ).to(device)
    output_ids = model.generate(**input_ids, max_new_tokens=max_output)
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return {"generated_text": generated_text}


# Prepare distributed processing
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs available for Accelerate to use.")


def process_lora(index):
    print(f"Starting processing for LoRA index: {index}")
    # Assign device for this task
    lora_device = accelerator.device

    base_model_outputs_file_path = os.path.join(base_models_outputs_dir, f"{index}.pt")
    base_lora_outputs_file_path = os.path.join(base_loras_outputs_dir, f"{index}.pt")
    predicted_lora_outputs_file_path = os.path.join(
        predicted_loras_outputs_dir, f"{index}.pt"
    )

    base_model = copy.deepcopy(org_base_model)
    base_model.to(lora_device)

    # Load dataset for this LoRA
    data_address = Datasets_List[index]
    dataset = load_dataset(data_address)

    # Base model inference
    if not os.path.exists(base_model_outputs_file_path):
        print(f"Running base model inference for index: {index}")
        outputs = []
        for batch in tqdm(dataset["test"], desc=f"Base Model {index}"):
            result = inference(batch, base_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, base_model_outputs_file_path)

    # Load LoRA
    print(f"Loading LoRA model for index: {index}")
    peft_model_name = LoRAs_List[index]
    try:
        peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading LoRA {index}: {str(e)}")
    peft_model = peft_model.to(lora_device)
    peft_model.eval()

    # Base LoRA inference
    if not os.path.exists(base_lora_outputs_file_path):
        print(f"Running base LoRA inference for index: {index}")
        outputs = []
        for batch in tqdm(dataset["test"], desc=f"Base LoRA {index}"):
            result = inference(batch, peft_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, base_lora_outputs_file_path)

    # Predicted LoRA inference
    if not os.path.exists(predicted_lora_outputs_file_path):
        print(f"Running predicted LoRA inference for index: {index}")
        pred_file_path = os.path.join(
            predictions_folder_path, f"State_Dictionary{index}.safetensors"
        )
        pred_lora = safe_open(pred_file_path, "pt")
        pred_lora = {k: pred_lora.get_tensor(k) for k in pred_lora.keys()}
        set_peft_model_state_dict(peft_model, pred_lora)

        outputs = []
        for batch in tqdm(dataset["test"], desc=f"Predicted LoRA {index}"):
            result = inference(batch, peft_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, predicted_lora_outputs_file_path)
    print(f"Finished processing for LoRA index: {index}")


# Distribute tasks across GPUs
for batch_start in range(0, Number_of_LoRAs, num_gpus):
    print(
        f"Starting batch from {batch_start} to {min(batch_start + num_gpus, Number_of_LoRAs)}"
    )
    batch_end = min(batch_start + num_gpus, Number_of_LoRAs)
    indices = range(batch_start, batch_end)

    for index in indices:
        process_lora(index)

    # Clear memory after each batch
    print("Clearing memory for next batch...")
    gc.collect()
    torch.cuda.empty_cache()
print("All tasks completed.")

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


#### Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    if GPU_Memory_Free_mb <= 7000:
        device = torch.device("cpu")


# load model
org_base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
)
org_base_model.to("cpu")
org_base_model.eval()

# load tokenizer
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


if device.type == "cuda":
    max_threads = torch.cuda.device_count()
else:
    max_threads = 1

# def Model_Tester(index):

for index in tqdm(range(1)):
    device_index = index % max_threads
    lora_device = torch.device(
        f"cuda:{device_index}" if device.type == "cuda" else "cpu"
    )

    base_model_outputs_file_path = os.path.join(base_models_outputs_dir, f"{index}.pt")
    base_lora_outputs_file_path = os.path.join(base_loras_outputs_dir, f"{index}.pt")
    predicted_lora_outputs_file_path = os.path.join(
        predicted_lora_outputs_file_path, f"{index}.pt"
    )

    base_model = copy.deepcopy(org_base_model)
    base_model.to(lora_device)
    data_address = Datasets_List[index]
    dataset = load_dataset(data_address)

    if not os.path.exists(base_model_outputs_file_path):
        outputs = []
        for batch in tqdm(dataset["test"], desc="Getting outputs from base model"):
            result = inference(batch, base_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, base_model_outputs_file_path)

    peft_model_name = LoRAs_List[index]
    try:
        peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
    except:
        raise Exception(f"Error in loading LoRA with index {index}")
    peft_model = peft_model.to(lora_device)
    peft_model.eval()

    if not os.path.exists(base_lora_outputs_file_path):
        outputs = []
        for batch in tqdm(dataset["test"], desc="Getting outputs from base LoRA"):
            result = inference(batch, peft_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, base_lora_outputs_file_path)

    if not os.path.exists(predicted_lora_outputs_file_path):
        pred_file_path = os.path.join(
            predictions_folder_path, f"State_Dictionary{index}.safetensors"
        )
        pred_lora = safe_open(pred_file_path, "pt")
        pred_lora = {k: pred_lora.get_tensor(k) for k in pred_lora.keys()}
        set_peft_model_state_dict(peft_model, pred_lora)
        outputs = []
        for batch in tqdm(dataset["test"], desc="Getting outputs from predicted LoRA"):
            result = inference(batch, peft_model, tokenizer, lora_device)
            outputs.append([batch, result])
        torch.save(outputs, predicted_lora_outputs_file_path)

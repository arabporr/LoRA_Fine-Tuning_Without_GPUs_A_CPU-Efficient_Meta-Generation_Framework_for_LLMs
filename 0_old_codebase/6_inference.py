import os
import gc
import wget
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model
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
from transformers import TrainingArguments, Trainer
import datasets

# Initialize Accelerator
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs available for Accelerate to use.")
accelerator = Accelerator()

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_index", type=int, default=1)
args = parser.parse_args()
model_index = args.model_index


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


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=16000):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for item in dataset:
            # Tokenize input
            inputs = tokenizer(
                item["input"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Tokenize output/target
            labels = tokenizer(
                item["output"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Store tensors
            self.input_ids.append(inputs["input_ids"].squeeze())
            self.attention_mask.append(inputs["attention_mask"].squeeze())
            self.labels.append(labels["input_ids"].squeeze())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def process_lora(index):
    print(f"Starting processing for LoRA index: {index}")
    device = accelerator.device

    base_model_outputs_file_path = os.path.join(base_models_outputs_dir, f"{index}.pt")
    base_lora_outputs_file_path = os.path.join(base_loras_outputs_dir, f"{index}.pt")
    predicted_lora_outputs_file_path = os.path.join(
        predicted_loras_outputs_dir, f"{index}.pt"
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
    )
    base_model.eval()
    base_model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})
    base_model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    data_address = Datasets_List[index]
    dataset = load_dataset(data_address)

    # Select 20 random samples from train split for fine-tuning
    train_dataset = dataset["train"].shuffle(seed=42).select(range(20))
    test_dataset = dataset["test"]

    # Base model inference
    if not os.path.exists(base_model_outputs_file_path):
        print(f"Running base model inference for index: {index}")
        outputs = []
        for batch in tqdm(test_dataset, desc=f"Base Model {index}"):
            result = inference(batch, base_model, tokenizer, device)
            outputs.append([batch, result])
        torch.save(outputs, base_model_outputs_file_path)

    # Load original LoRA
    print(f"Loading LoRA model for index: {index}")
    peft_model_name = LoRAs_List[index]
    try:
        peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading LoRA {index}: {str(e)}")
    peft_model = peft_model.to(device)

    # Base LoRA inference
    if not os.path.exists(base_lora_outputs_file_path):
        print(f"Running base LoRA inference for index: {index}")
        outputs = []
        for batch in tqdm(test_dataset, desc=f"Base LoRA {index}"):
            result = inference(batch, peft_model, tokenizer, device)
            outputs.append([batch, result])
        torch.save(outputs, base_lora_outputs_file_path)

    # Load and fine-tune predicted LoRA
    if not os.path.exists(predicted_lora_outputs_file_path):
        print(f"Loading and fine-tuning predicted LoRA for index: {index}")
        pred_file_path = os.path.join(
            predictions_folder_path, f"State_Dictionary{index}.safetensors"
        )
        pred_lora = safe_open(pred_file_path, "pt")
        pred_lora = {k: pred_lora.get_tensor(k) for k in pred_lora.keys()}
        # set_peft_model_state_dict(peft_model, pred_lora)

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=f"./fine_tuned_lora_{index}",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            warmup_steps=50,
            learning_rate=1e-4,
            fp16=True,
            logging_dir=f"./logs_{index}",
            logging_steps=10,
            save_strategy="no",
        )

        # Create tokenized dataset
        train_dataset = TokenizedDataset(train_dataset, tokenizer)
        peft_model.train()

        # Initialize trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Fine-tune the model
        print(f"Fine-tuning predicted LoRA {index} on 20 samples...")
        trainer.train()

        # Run inference with fine-tuned model
        print(f"Running inference with fine-tuned predicted LoRA {index}")
        outputs = []
        for batch in tqdm(test_dataset, desc=f"Fine-tuned Predicted LoRA {index}"):
            result = inference(batch, peft_model, tokenizer, device)
            outputs.append([batch, result])
        torch.save(outputs, predicted_lora_outputs_file_path)

    print(f"Finished processing for LoRA index: {index}")

    del peft_model
    gc.collect()
    torch.cuda.empty_cache()


process_lora(model_index)
print("All tasks completed.")

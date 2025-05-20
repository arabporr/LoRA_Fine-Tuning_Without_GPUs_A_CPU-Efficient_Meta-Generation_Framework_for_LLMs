import os
import gc
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import torch
from transformers import AutoTokenizer

from src.config.config import (
    max_threads_cpu_task,
    max_threads_memory_task,
    base_model_name,
)
from src.config.paths import raw_datasets_dir, tokenized_datasets_dir
from src.data.LoRAs_Info import Number_of_LoRAs


tokenizer = AutoTokenizer.from_pretrained(base_model_name)


#### Training Dataset Tokenization and Save
def tokenization_handler(index: int) -> str:
    dataset_file_location = os.path.join(raw_datasets_dir, f"{index}.pt")
    tokenized_dataset_file_location = os.path.join(
        tokenized_datasets_dir, f"{index}.pt"
    )

    if os.path.exists(dataset_file_location):
        dataset = torch.load(dataset_file_location)
        tokenized_dataset = []
        for sample in dataset["train"]:
            prompt = sample["input"] + sample["output"][0]
            tokenized_dataset.append(tokenizer(prompt).input_ids)
        tokenized_dataset = np.array(list(itertools.chain(*tokenized_dataset)))
        torch.save(tokenized_dataset, tokenized_dataset_file_location)
        return "LoRA {index} successfully tokenized and saved."
    else:
        raise Exception(f"Error, dataset {index} not found for tokenization!")


_max_threads = min(max_threads_cpu_task, max_threads_memory_task)
with ThreadPoolExecutor(max_workers=_max_threads) as executor:
    futures = [
        executor.submit(tokenization_handler, index) for index in range(Number_of_LoRAs)
    ]
    for future in as_completed(futures):
        if "Error" in future.result():
            print(future.result())


del tokenizer
gc.collect()

print(40 * "*")
print("DATASETS TOKENIZATION FINISHED SUCCESSFULLY")

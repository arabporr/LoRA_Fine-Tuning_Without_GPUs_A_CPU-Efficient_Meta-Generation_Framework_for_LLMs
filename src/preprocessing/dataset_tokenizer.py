import os
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
def dataset_tokenizer(index: int) -> str:
    dataset_file_location = os.path.join(raw_datasets_dir, f"{index}.pt")
    tokenized_dataset_file_location = os.path.join(
        tokenized_datasets_dir, f"{index}.pt"
    )

    if os.path.exists(dataset_file_location):
        if not os.path.exists(tokenized_dataset_file_location):
            dataset = torch.load(dataset_file_location, weights_only=False)
            tokenized_dataset = []
            for sample in dataset["train"]:
                prompt = sample["input"] + sample["output"][0]
                tokenized_dataset.append(tokenizer(prompt).input_ids)
            tokenized_dataset = np.array(list(itertools.chain(*tokenized_dataset)))
            torch.save(tokenized_dataset, tokenized_dataset_file_location)
            return "Dataset {index} successfully tokenized and saved."
        else:
            return "Dataset {index} was already tokenized and saved!"
    else:
        raise Exception(f"Error, dataset {index} not found for tokenization!")


def tokenization_handler():
    _max_threads = min(max_threads_cpu_task, max_threads_memory_task)
    with ThreadPoolExecutor(max_workers=_max_threads) as executor:
        futures = [
            executor.submit(dataset_tokenizer, index) for index in range(Number_of_LoRAs)
        ]
        for future in as_completed(futures):
            if "Error" in future.result():
                print(future.result())

    print(40 * "*")
    print("DATASETS TOKENIZATION FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    tokenization_handler()

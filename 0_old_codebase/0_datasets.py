import os
import gc
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from LoRAs_Info import *
from config import *


missing_datasets = []
working_datasets = []
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#### Dataset Tokenizer and Saver Function


def tokenize_and_save(index: int):
    data_address = Datasets_List[index]
    ds = load_dataset(data_address)
    tokenized_data = []
    for sample in ds["train"]:
        prompt = sample["input"] + sample["output"][0]
        tokenized_data.append(tokenizer(prompt).input_ids)
    tokenized_data = np.array(list(itertools.chain(*tokenized_data)))
    tokenized_data_file_path = os.path.join(datasets_folder_path, f"{index}.pt")
    torch.save(tokenized_data, tokenized_data_file_path)
    del tokenized_data
    gc.collect()


#### Multi-Threading


def dataset_handler(index: int) -> str:
    tokenized_data_file_path = os.path.join(datasets_folder_path, f"{index}.pt")
    if not os.path.exists(tokenized_data_file_path):
        try:
            tokenize_and_save(index)
            working_datasets.append(index)
            return "LoRA {index} successfully tokenized and saved."
        except:
            missing_datasets.append(index)
            return f"Error in loading LoRA {index}"
    else:
        working_datasets.append(index)
        return "LoRA {index} found in the folder!"


_max_threads = min(max_threads_cpu_task, max_threads_memory_task)
with ThreadPoolExecutor(max_workers=_max_threads) as executor:
    futures = [
        executor.submit(dataset_handler, index) for index in range(Number_of_LoRAs)
    ]
    for future in as_completed(futures):
        if "Error" in future.result():
            print(future.result())


print("Finished downloading and tokenizing LoRA datasets.")
missing_count = Number_of_LoRAs - len(os.listdir(datasets_folder_path))
if missing_count:
    print(f"*** BE CAREFUL, {missing_count} DATASETS ARE MISSING !!! ***")

torch.save(working_datasets, working_datasets_path)
print("Finished saving working LoRA IDs.")

del tokenizer
gc.collect()

#### End of Script Print
print(40 * "*")
print("DATASETS PART FINISHED SUCCESSFULLY")

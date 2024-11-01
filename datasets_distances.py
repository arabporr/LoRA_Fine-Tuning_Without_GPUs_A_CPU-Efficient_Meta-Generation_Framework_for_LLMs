import os
import gc
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import wasserstein_distance

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from LoRAs_Info import Number_of_LoRAs, LoRAs_IDs, LoRAs_List, Datasets_List
from config import *


#### Dataset Tokenizer and Saving Part
missing_datasets = []
working_datasets = []
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


def tokenize_and_save(index: int):
    data_address = Datasets_List[index]
    ds = load_dataset(data_address)
    tokenized_data = []
    for sample in ds["train"]:
        prompt = sample["input"] + sample["output"][0]
        tokenized_data.append(tokenizer(prompt).input_ids)
    tokenized_data = np.array(list(itertools.chain(*tokenized_data)))
    torch.save(tokenized_data, os.path.join(datasets_folder_path, f"{index}.pt"))
    del tokenized_data
    gc.collect()


def dataset_handler(index: int) -> str:
    if not os.path.exists(os.path.join(datasets_folder_path, f"{index}.pt")):
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


#### Distances Calculation and Saving Part
Distance_Vectors = torch.zeros((len(working_datasets), len(working_datasets)))


def W_Distance_Calculator(base_index: int) -> str:
    distance_file_name = os.path.join(distances_folder_path, f"{base_index}.pt")

    if not os.path.exists(distance_file_name):
        try:
            base_dataset = torch.load(
                os.path.join(datasets_folder_path, f"{base_index}.pt")
            )
            for second_index in working_datasets:
                if second_index > base_index:
                    print(
                        f"calculating w-distance, lora id: {base_index} and lora id: {second_index}"
                    )
                    second_dateset = torch.load(
                        os.path.join(datasets_folder_path, f"{second_index}.pt")
                    )
                    dist = wasserstein_distance(base_dataset, second_dateset)
                    Distance_Vectors[base_index][second_index] = dist
                    Distance_Vectors[second_index][base_index] = dist
            torch.save(Distance_Vectors[base_index], distance_file_name)
            return f"Done making base dataset: {base_index}"
        except:
            return f"Error in making base dataset: {base_index}"
    else:
        try:
            distances = torch.load(distance_file_name)
            for second_index in range(len(distances)):
                if Distance_Vectors[base_index][second_index] == 0:
                    Distance_Vectors[base_index][second_index] = distances[second_index]
                if Distance_Vectors[second_index][base_index] == 0:
                    Distance_Vectors[second_index][base_index] = distances[second_index]
            return f"Done loading base dataset: {base_index}"
        except:
            return f"Error in loading base dataset: {base_index}"


working_datasets.sort()
_max_threads = max_threads_cpu_task
with ThreadPoolExecutor(max_workers=_max_threads) as executor:
    futures = [
        executor.submit(W_Distance_Calculator, index) for index in working_datasets
    ]
for future in as_completed(futures):
    if "Error" in future.result():
        print(future.result())


results_path = os.path.join(parent_dir_data, distances_result_file)
torch.save(Distance_Vectors, results_path)
del Distance_Vectors
gc.collect()

#### End of Script Print
print(40 * "*")
print("DISTANCES PART FINISHED SUCCESSFULLY")

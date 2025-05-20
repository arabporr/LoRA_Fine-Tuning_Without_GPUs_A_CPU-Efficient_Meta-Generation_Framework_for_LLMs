import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from datasets import load_dataset

from tqdm import tqdm

from src.config.config import max_threads_cpu_task
from src.config.paths import raw_datasets_dir
from src.data.LoRAs_Info import Number_of_LoRAs, Datasets_List


#### Multi-Thread dataset downloading
def download_dataset(index: int) -> str:
    dataset_file_location = os.path.join(raw_datasets_dir, f"{index}.pt")
    if not os.path.exists(dataset_file_location):
        dataset_address = Datasets_List[index]
        dataset = load_dataset(dataset_address)
        torch.save(dataset, dataset_file_location)
        return f"dataset {index} successfully saved."
    else:
        return "dataset {index} already existed!"

def dataset_handler():
    _max_threads = max_threads_cpu_task
    with ThreadPoolExecutor(max_workers=_max_threads) as executor:
        futures = [
            executor.submit(download_dataset, index)
            for index in tqdm(range(Number_of_LoRAs))
        ]
        for future in as_completed(futures):
            if "Error" in future.result():
                print(future.result())

    missing_count = Number_of_LoRAs - len(os.listdir(raw_datasets_dir))
    if missing_count:
        print(f"*** BE CAREFUL, {missing_count} DATASETS ARE MISSING !!! ***")

    print(40 * "*")
    print("DATASETS DOWNLOADING FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    dataset_handler()


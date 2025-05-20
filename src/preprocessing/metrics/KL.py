import os
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed


import numpy as np

import torch

from tqdm import tqdm


from src.config.config import max_threads_cpu_task, tokenizer_dictionary_size
from src.config.paths import distances_dir, tokenized_datasets_dir
from src.data.LoRAs_Info import Number_of_LoRAs



def density_function_maker():
    bins = np.arange(0, tokenizer_dictionary_size)  # Domain is dictionary size
    epsilon = 1e-10

    for index in tqdm(range(Number_of_LoRAs), desc="Making density functions: "):
        first_tokenized_dataset = torch.load(
            os.path.join(tokenized_datasets_dir, f"{index}.pt"),
            weights_only=False,
        )
        hist, _ = np.histogram(first_tokenized_dataset, bins=bins, density=True)
        hist += epsilon
        hists.append(hist)


def distance_metric(index_1, index_2):
    f1_hist = hists[index_1]
    f2_hist = hists[index_2]

    KL_divergence = np.sum(f1_hist * np.log(f1_hist / f2_hist))
    return KL_divergence


#### Distances Calculation and Saving
def Distance_Calculator(first_index: int) -> str:
    distances_metric_dir = os.path.join(distances_dir, "KL")
    distance_file_location = os.path.join(distances_metric_dir, f"{first_index}.pt")

    if not os.path.exists(distance_file_location):
        try:
            for second_index in range(Number_of_LoRAs):
                print(
                    f"calculating distance, lora id: {first_index} and lora id: {second_index}"
                )
                dist = distance_metric(first_index, second_index)
                Distance_Vectors[first_index][second_index] = dist
            torch.save(Distance_Vectors[first_index], distance_file_location)
            return f"Done making distances for dataset: {first_index}"
        except Exception as e:
            return f"Error in making distances for dataset: {first_index}\n\t{e}"
    else:
        try:
            distances = torch.load(distance_file_location, weights_only=False)
            for second_index in range(Number_of_LoRAs):
                if Distance_Vectors[first_index][second_index] == 0:
                    Distance_Vectors[first_index][second_index] = distances[
                        second_index
                    ]
            return f"Done loading distances for dataset: {first_index}"
        except:
            raise Exception(f"Error in loading distances for dataset: {first_index}")


def KL_run():
    global Distance_Vectors
    Distance_Vectors = torch.zeros((Number_of_LoRAs, Number_of_LoRAs))

    global hists
    hists = []

    density_function_maker()
    
    #### Multi-Threading
    _max_threads = max_threads_cpu_task
    with ThreadPoolExecutor(max_workers=_max_threads) as executor:
        futures = [
            executor.submit(Distance_Calculator, index)
            for index in tqdm(range(Number_of_LoRAs))
        ]
    for future in as_completed(futures):
        if "Error" in future.result():
            print(future.result())

    All_distances_file_location = os.path.join(distances_dir, "KL_all_distances.pt")
    torch.save(Distance_Vectors, All_distances_file_location)

    del Distance_Vectors
    gc.collect()

    #### End of Script Print
    print(40 * "*")
    print("KL DISTANCE CALCULATIONS FINISHED SUCCESSFULLY")

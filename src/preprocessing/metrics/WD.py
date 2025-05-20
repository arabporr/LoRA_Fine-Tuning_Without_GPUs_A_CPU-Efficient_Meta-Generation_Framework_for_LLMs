import os
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.stats import wasserstein_distance

import torch

from tqdm import tqdm


from src.config.config import max_threads_cpu_task
from src.config.paths import distances_dir, tokenized_datasets_dir
from src.data.LoRAs_Info import Number_of_LoRAs


def distance_metric(f1_samples, f2_samples):
    WD_distance = wasserstein_distance(f1_samples, f2_samples)
    # if data dimension is bigger than 1, then use Sinkhorn algorithm
    return WD_distance


#### Distances Calculation and Saving
def Distance_Calculator(first_index: int) -> str:
    distances_metric_dir = os.path.join(distances_dir, "WD")
    distance_file_location = os.path.join(distances_metric_dir, f"{first_index}.pt")

    if not os.path.exists(distance_file_location):
        try:
            first_tokenized_dataset = torch.load(
                os.path.join(tokenized_datasets_dir, f"{first_index}.pt"),
                weights_only=False,
            )
            for second_index in range(first_index, Number_of_LoRAs):  # symmetric
                print(
                    f"calculating distance, lora id: {first_index} and lora id: {second_index}"
                )
                second_tokenized_dateset = torch.load(
                    os.path.join(tokenized_datasets_dir, f"{second_index}.pt"),
                    weights_only=False,
                )
                dist = distance_metric(
                    first_tokenized_dataset, second_tokenized_dateset
                )
                Distance_Vectors[first_index][second_index] = dist
                Distance_Vectors[second_index][first_index] = dist  # symmetric

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


def WD_run():
    global Distance_Vectors
    Distance_Vectors = torch.zeros((Number_of_LoRAs, Number_of_LoRAs))

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

    All_distances_file_location = os.path.join(distances_dir, "WD_all_distances.pt")
    torch.save(Distance_Vectors, All_distances_file_location)

    del Distance_Vectors
    gc.collect()

    #### End of Script Print
    print(40 * "*")
    print("WD DISTANCE CALCULATIONS FINISHED SUCCESSFULLY")

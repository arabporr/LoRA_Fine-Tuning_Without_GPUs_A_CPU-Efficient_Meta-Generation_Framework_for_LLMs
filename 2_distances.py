import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.stats import wasserstein_distance

import torch

from LoRAs_Info import *
from config import *


working_datasets = torch.load(working_datasets_path)
Distance_Vectors = torch.zeros((len(working_datasets), len(working_datasets)))


#### Setting Distance Measure
distance_metric = wasserstein_distance


#### Distances Calculation and Saving Part
def Distance_Calculator(base_index: int) -> str:
    distance_file_path = os.path.join(distances_folder_path, f"{base_index}.pt")

    if not os.path.exists(distance_file_path):
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
                    dist = distance_metric(base_dataset, second_dateset)
                    Distance_Vectors[base_index][second_index] = dist
                    Distance_Vectors[second_index][base_index] = dist
            torch.save(Distance_Vectors[base_index], distance_file_path)
            return f"Done making base dataset: {base_index}"
        except:
            return f"Error in making base dataset: {base_index}"
    else:
        try:
            distances = torch.load(distance_file_path)
            for second_index in range(len(distances)):
                if Distance_Vectors[base_index][second_index] == 0:
                    Distance_Vectors[base_index][second_index] = distances[second_index]
                if Distance_Vectors[second_index][base_index] == 0:
                    Distance_Vectors[second_index][base_index] = distances[second_index]
            return f"Done loading base dataset: {base_index}"
        except:
            return f"Error in loading base dataset: {base_index}"


#### Multi-Threading
working_datasets.sort()
_max_threads = max_threads_cpu_task
with ThreadPoolExecutor(max_workers=_max_threads) as executor:
    futures = [
        executor.submit(Distance_Calculator, index) for index in working_datasets
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

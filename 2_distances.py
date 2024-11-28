import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scipy.stats import wasserstein_distance

import torch

from LoRAs_Info import *
from config import *
from experiment_info import distance

working_datasets = torch.load(working_datasets_path, weights_only=False)
Distance_Vectors = torch.zeros((len(working_datasets), len(working_datasets)))


# Distance Function
def W_Distance(f1_samples, f2_samples):
    w_dist = wasserstein_distance(
        f1_samples, f2_samples
    )  # if data dimension is bigger than 1, then use Sinkhorn algorithm
    return w_dist


def KL_Divergence(f1_samples, f2_samples):
    bins = np.arange(0, tokenizer_dictionary_size)  # Domain is dictionary size
    f1_hist, _ = np.histogram(f1_samples, bins=bins, density=True)
    f2_hist, _ = np.histogram(f2_samples, bins=bins, density=True)

    epsilon = 1e-10
    f1_hist += epsilon
    f2_hist += epsilon

    kl_div = np.sum(f1_hist * np.log(f1_hist / f2_hist))
    return kl_div


def JS_Divergence(f1_samples, f2_samples):
    bins = np.arange(0, tokenizer_dictionary_size)  # Domain is dictionary size
    f1_hist, _ = np.histogram(f1_samples, bins=bins, density=True)
    f2_hist, _ = np.histogram(f2_samples, bins=bins, density=True)

    epsilon = 1e-10
    f1_hist += epsilon
    f2_hist += epsilon

    m_hist = 0.5 * (f1_hist + f2_hist)
    js_div = 0.5 * np.sum(f1_hist * np.log(f1_hist / m_hist)) + 0.5 * np.sum(
        f2_hist * np.log(f2_hist / m_hist)
    )
    return js_div


#### Setting Distance Measure
if distance == "WD":
    distance_metric = W_Distance
    dist_func_is_symmetric = True
elif distance == "KL":
    distance_metric = KL_Divergence
    dist_func_is_symmetric = False
elif distance == "JS":
    distance_metric = JS_Divergence
    dist_func_is_symmetric = True


#### Distances Calculation and Saving Part
def Distance_Calculator(base_index: int) -> str:
    distance_file_path = os.path.join(distances_folder_path, f"{base_index}.pt")

    if not os.path.exists(distance_file_path):
        try:
            base_dataset = torch.load(
                os.path.join(datasets_folder_path, f"{base_index}.pt"),
                weights_only=False,
            )
            for second_index in working_datasets:
                if dist_func_is_symmetric:
                    if second_index > base_index:
                        print(
                            f"calculating distance, lora id: {base_index} and lora id: {second_index}"
                        )
                        second_dateset = torch.load(
                            os.path.join(datasets_folder_path, f"{second_index}.pt"),
                            weights_only=False,
                        )
                        dist = distance_metric(base_dataset, second_dateset)
                        Distance_Vectors[base_index][second_index] = dist
                        Distance_Vectors[second_index][base_index] = dist
                else:
                    print(
                        f"calculating distance, lora id: {base_index} and lora id: {second_index}"
                    )
                    second_dateset = torch.load(
                        os.path.join(datasets_folder_path, f"{second_index}.pt"),
                        weights_only=False,
                    )
                    dist = distance_metric(base_dataset, second_dateset)
                    Distance_Vectors[base_index][second_index] = dist

            torch.save(Distance_Vectors[base_index], distance_file_path)
            return f"Done making base dataset: {base_index}"
        except:
            return f"Error in making base dataset: {base_index}"
    else:
        try:
            distances = torch.load(distance_file_path, weights_only=False)
            for second_index in range(Number_of_LoRAs):
                if Distance_Vectors[base_index][second_index] == 0:
                    Distance_Vectors[base_index][second_index] = distances[second_index]
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


torch.save(Distance_Vectors, distances_result_file_path)

del Distance_Vectors
gc.collect()

#### End of Script Print
print(40 * "*")
print("DISTANCES PART FINISHED SUCCESSFULLY")

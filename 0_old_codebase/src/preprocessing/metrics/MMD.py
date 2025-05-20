import os
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed


import numpy as np

import torch

from tqdm import tqdm


from src.config.config import max_threads_cpu_task
from src.config.paths import distances_dir, tokenized_datasets_dir
from src.data.LoRAs_Info import Number_of_LoRAs


Distance_Vectors = torch.zeros((Number_of_LoRAs, Number_of_LoRAs))


def distance_metric(f1_samples, f2_samples, kernel: str = "multiscale"):
    x = torch.tensor([np.random.choice(f1_samples, 1000)])
    y = torch.tensor([np.random.choice(f2_samples, 1000)])
    xx = torch.mm(x.t(), x)
    yy = torch.mm(y.t(), y)
    zz = torch.mm(x.t(), y)
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape),
        torch.zeros(xx.shape),
        torch.zeros(xx.shape),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    Maximum_mean_discrepancy = torch.mean(XX + YY - 2.0 * XY)
    return Maximum_mean_discrepancy


#### Distances Calculation and Saving
def Distance_Calculator(first_index: int) -> str:
    distances_metric_dir = os.path.join(distances_dir, "MMD")
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


def MMD_run():
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

    All_distances_file_location = os.path.join(distances_dir, "MMD_all_distances.pt")
    torch.save(Distance_Vectors, All_distances_file_location)

    del Distance_Vectors
    gc.collect()

    #### End of Script Print
    print(40 * "*")
    print("MMD DISTANCE CALCULATIONS FINISHED SUCCESSFULLY")

import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scipy.stats import wasserstein_distance

import torch

from tqdm import tqdm

from LoRAs_Info import *
from config import *
from experiment_info import distance

working_datasets = torch.load(working_datasets_path, weights_only=False)
Distance_Vectors = torch.zeros((len(working_datasets), len(working_datasets)))


# Distance Function
def pdf_maker():
    global hists
    hists = []

    bins = np.arange(0, tokenizer_dictionary_size)  # Domain is dictionary size
    epsilon = 1e-10

    for index in tqdm(range(Number_of_LoRAs), desc="Making density functions: "):
        base_dataset = torch.load(
            os.path.join(datasets_folder_path, f"{index}.pt"),
            weights_only=False,
        )
        hist, _ = np.histogram(base_dataset, bins=bins, density=True)
        hist += epsilon
        hists.append(hist)


def W_Distance(f1_samples, f2_samples, index_1, index_2):
    w_dist = wasserstein_distance(
        f1_samples, f2_samples
    )  # if data dimension is bigger than 1, then use Sinkhorn algorithm
    return w_dist


def KL_Divergence(f1_samples, f2_samples, index_1, index_2):
    f1_hist = hists[index_1]
    f2_hist = hists[index_2]

    kl_div = np.sum(f1_hist * np.log(f1_hist / f2_hist))
    return kl_div


def JS_Divergence(f1_samples, f2_samples, index_1, index_2):
    f1_hist = hists[index_1]
    f2_hist = hists[index_2]

    m_hist = 0.5 * (f1_hist + f2_hist)
    js_div = 0.5 * np.sum(f1_hist * np.log(f1_hist / m_hist)) + 0.5 * np.sum(
        f2_hist * np.log(f2_hist / m_hist)
    )
    return js_div


def MMD(f1_samples, f2_samples, index_1, index_2, kernel: str = "multiscale"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
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

    return torch.mean(XX + YY - 2.0 * XY)


#### Setting Distance Measure
if distance == "WD":
    distance_metric = W_Distance
    dist_func_is_symmetric = True
elif distance == "KL":
    pdf_maker()
    distance_metric = KL_Divergence
    dist_func_is_symmetric = False
elif distance == "JS":
    pdf_maker()
    distance_metric = JS_Divergence
    dist_func_is_symmetric = True
elif distance == "MMD":
    distance_metric = MMD
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
                        dist = distance_metric(
                            base_dataset, second_dateset, base_index, second_index
                        )
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
                    dist = distance_metric(
                        base_dataset, second_dateset, base_index, second_index
                    )
                    Distance_Vectors[base_index][second_index] = dist

            torch.save(Distance_Vectors[base_index], distance_file_path)
            return f"Done making base dataset: {base_index}"
        except Exception as e:
            return f"Error in making base dataset: {base_index}\n\t{e}"
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

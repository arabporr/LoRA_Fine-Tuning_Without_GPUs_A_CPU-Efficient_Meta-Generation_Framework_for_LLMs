import os

import psutil
import multiprocessing

import torch

n_cpu_cores = multiprocessing.cpu_count()
max_threads_cpu_task = (
    n_cpu_cores - 4
)  # All the cores except 4 for system functionality

ram_size_gb = int(psutil.virtual_memory().total / (1024 * 1024 * 1000))
max_threads_memory_task = (
    ram_size_gb * 0.5
)  # Assume each task consumes 1GB, we use 50% of RAM

max_threads_gpu_task = torch.cuda.device_count()


base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"


# Assuming that we want to put everything in the current directory
parent_dir = os.getcwd()

datasets_folder_name = "Datasets_Train_Tokenized"
datasets_folder_path = os.path.join(parent_dir, datasets_folder_name)
if not os.path.exists(datasets_folder_path):
    os.makedirs(datasets_folder_path)

distances_folder_name = "Distances"
distances_folder_path = os.path.join(parent_dir, distances_folder_name)
if not os.path.exists(distances_folder_path):
    os.makedirs(distances_folder_path)

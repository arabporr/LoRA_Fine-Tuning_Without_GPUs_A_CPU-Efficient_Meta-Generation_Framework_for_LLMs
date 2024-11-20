import os
import psutil
import multiprocessing

import torch

from transformers import BitsAndBytesConfig


#### APIs Configs
from dotenv import load_dotenv

load_dotenv()


from huggingface_hub import login

hf_token = os.getenv("hf_token")
login(hf_token)


#### MultiThreading Configs
if torch.cuda.is_available():
    GPU_Memory_Free, GPU_Memory_Total = torch.cuda.mem_get_info()
    GPU_Memory_Free_mb = GPU_Memory_Free / (1024**2)  # Convert bytes to MB
    GPU_Memory_Total_mb = GPU_Memory_Total / (1024**2)  # Convert bytes to MB

# All the cores except 4 for system functionality
n_cpu_cores = multiprocessing.cpu_count()
max_threads_cpu_task = n_cpu_cores - 4

# Assume each task consumes 1GB, we use 50% of RAM
ram_size_gb = int(psutil.virtual_memory().total / (1024 * 1024 * 1000))
max_threads_memory_task = ram_size_gb * 0.5

# All gpu devices
max_threads_gpu_task = torch.cuda.device_count()


#### LLM Configs
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

#### File handling Configs

# Assuming that we want to put everything in the current directory
current_dir = os.getcwd()

# Data Paths
parent_dir_data = os.path.join(current_dir, "data")
if not os.path.exists(parent_dir_data):
    os.makedirs(parent_dir_data)

datasets_folder_name = "Datasets_Train_Tokenized"
datasets_folder_path = os.path.join(parent_dir_data, datasets_folder_name)
if not os.path.exists(datasets_folder_path):
    os.makedirs(datasets_folder_path)
working_datasets_path = os.path.join(parent_dir_data, "Working_datasets_index.pt")

distances_folder_name = "Distances"
distances_result_file = "Distance_Vectors.pt"
distances_folder_path = os.path.join(parent_dir_data, distances_folder_name)
if not os.path.exists(distances_folder_path):
    os.makedirs(distances_folder_path)

adapters_folder_name = "Adapters"
adapters_result_file = "All_Adapters.pt"
adapters_folder_path = os.path.join(parent_dir_data, adapters_folder_name)
if not os.path.exists(adapters_folder_path):
    os.makedirs(adapters_folder_path)

processed_distances_result_file = "Distances_Processed.pt"

predictions_folder_name = "Predictions"
predictions_result_file = "All_Predictions.pt"
predictions_folder_path = os.path.join(parent_dir_data, predictions_folder_name)
if not os.path.exists(predictions_folder_path):
    os.makedirs(predictions_folder_path)

# Results Paths
parent_dir_results = os.path.join(current_dir, "results")
if not os.path.exists(parent_dir_results):
    os.makedirs(parent_dir_results)

# Loss Paths
parent_dir_loss = os.path.join(parent_dir_results, "loss")
if not os.path.exists(parent_dir_loss):
    os.makedirs(parent_dir_loss)

weights_loss_folder_name = "weights_pred_loss"
weights_loss_result_file = "All_Weights_Preds_Loss.pt"
weights_loss_folder_path = os.path.join(parent_dir_loss, weights_loss_folder_name)
if not os.path.exists(weights_loss_folder_path):
    os.makedirs(weights_loss_folder_path)

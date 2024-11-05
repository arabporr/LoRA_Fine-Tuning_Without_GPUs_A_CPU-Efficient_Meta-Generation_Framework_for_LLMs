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

parent_dir_data = os.path.join(current_dir, "data")
if not os.path.exists(parent_dir_data):
    os.makedirs(parent_dir_data)

datasets_folder_name = "Datasets_Train_Tokenized"
datasets_folder_path = os.path.join(parent_dir_data, datasets_folder_name)
if not os.path.exists(datasets_folder_path):
    os.makedirs(datasets_folder_path)

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


parent_dir_results = os.path.join(current_dir, "results")
if not os.path.exists(parent_dir_results):
    os.makedirs(parent_dir_results)

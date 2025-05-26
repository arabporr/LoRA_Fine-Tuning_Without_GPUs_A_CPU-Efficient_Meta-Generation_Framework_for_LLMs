from huggingface_hub import login
import os
import psutil
import multiprocessing

import torch

from transformers import BitsAndBytesConfig


# APIs Configs
from dotenv import load_dotenv

load_dotenv()


hf_token = os.getenv("hf_token")
login(hf_token)


# MultiThreading Configs

# All the cores except 4 for system functionality
n_cpu_cores = multiprocessing.cpu_count()
max_threads_cpu_task = n_cpu_cores - 2

# Assume each task consumes 1GB, we use 50% of RAM
ram_size_gb = int(psutil.virtual_memory().total / (1024 * 1024 * 1000))
max_threads_memory_task = ram_size_gb * 0.5

# All gpu devices
max_threads_gpu_task = torch.cuda.device_count()


# LLM Configs
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer_dictionary_size = 32000

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

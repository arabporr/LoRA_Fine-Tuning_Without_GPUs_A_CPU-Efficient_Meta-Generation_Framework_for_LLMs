import os
import gc
import wget

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from tqdm import tqdm

from safetensors import safe_open

from LoRAs_Info import Number_of_LoRAs, LoRAs_IDs, LoRAs_List, Datasets_List
from config import *


#### Downloading LoRA Adapters Params Part
dl_adapters_configs = False


def download_params(index: int) -> str:
    params_file_url = f"https://huggingface.co/{LoRAs_List[index]}/resolve/main/adapter_model.safetensors?download=true"
    params_file_name = os.path.join(adapters_folder_path, f"{index}.safetensors")

    if dl_adapters_configs:
        conf_file_url = f"https://huggingface.co/{LoRAs_List[index]}/resolve/main/adapter_config.json?download=true"
        conf_file_name = f"{index}_conf.json"

    if not os.path.exists(params_file_name):
        try:
            wget.download(url=params_file_url, out=params_file_name)
            if dl_adapters_configs:
                wget.download(url=conf_file_url, out=conf_file_name)
            return f"Done with downloading LoRA Adaptors {index}"
        except:
            return f"Error in downloading LoRA Adaptors {index}"
    else:
        return "Already exists."


_max_threads = max_threads_cpu_task
with ThreadPoolExecutor(max_workers=_max_threads) as executor:
    futures = [
        executor.submit(download_params, index)
        for index in tqdm(range(Number_of_LoRAs))
    ]
    for future in as_completed(futures):
        if "Error" in future.result():
            print(future.result())


#### Making Parameters Vector Part
all_params = []
for index in tqdm(range(Number_of_LoRAs)):
    file_name = os.path.join(adapters_folder_path, f"{index}.safetensors")
    file = safe_open(file_name, "pt")
    params = []
    for key in file.keys():
        params.append(file.get_tensor(key).flatten())
    params = torch.concat(params)
    all_params.append(params)

all_params = torch.stack(all_params)
result_path = os.path.join(parent_dir_data, adapters_result_file)
torch.save(all_params, result_path)
del all_params
gc.collect()

#### End of Run Print
print(40 * "*")
print("Done with making parameters vector.")

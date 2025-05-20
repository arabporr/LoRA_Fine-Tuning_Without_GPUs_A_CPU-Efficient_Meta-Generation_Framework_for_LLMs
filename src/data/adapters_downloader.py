import os
import gc
import wget

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from tqdm import tqdm

from safetensors import safe_open

from src.config.config import max_threads_cpu_task
from src.config.paths import raw_adapters_dir, all_adapters_file_location
from src.data.LoRAs_Info import LoRAs_List, Number_of_LoRAs


#### Downloading LoRA Adapters in Parallel (Multi-Thread)
def download_adapters(index: int, dl_adapters_configs = False) -> str:
    adapters_file_url = f"https://huggingface.co/{LoRAs_List[index]}/resolve/main/adapter_model.safetensors?download=true"
    adapters_file_path = os.path.join(raw_adapters_dir, f"{index}.safetensors")

    if dl_adapters_configs:
        conf_file_url = f"https://huggingface.co/{LoRAs_List[index]}/resolve/main/adapter_config.json?download=true"
        conf_file_name = f"{index}_conf.json"

    if not os.path.exists(adapters_file_path):
        wget.download(url=adapters_file_url, out=adapters_file_path)
        if dl_adapters_configs:
            wget.download(url=conf_file_url, out=conf_file_name)
        return f"Done with downloading LoRA Adaptors {index}"
    else:
        return "Already exists."


def adapter_handler():
    _max_threads = max_threads_cpu_task
    with ThreadPoolExecutor(max_workers=_max_threads) as executor:
        futures = [
            executor.submit(download_adapters, index)
            for index in tqdm(range(Number_of_LoRAs))
        ]
        for future in as_completed(futures):
            if "Error" in future.result():
                print(future.result())

    #### Making All Adapters Matrix
    all_adapters = []
    for index in tqdm(range(Number_of_LoRAs)):
        file_name = os.path.join(raw_adapters_dir, f"{index}.safetensors")
        file = safe_open(file_name, "pt")
        adapters = []
        for key in file.keys():
            adapters.append(file.get_tensor(key).flatten())
        adapters = torch.concat(adapters)
        all_adapters.append(adapters)

    all_adapters = torch.stack(all_adapters)
    torch.save(all_adapters, all_adapters_file_location)
    del all_adapters
    gc.collect()

    print(40 * "*")
    print("ADAPTERS DOWNLOADING FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    adapter_handler()

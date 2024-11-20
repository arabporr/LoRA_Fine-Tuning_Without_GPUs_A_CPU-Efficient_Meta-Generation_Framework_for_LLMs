import os
import gc

import torch
from safetensors import safe_open

from config import *
from LoRAs_Info import *


#### Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### Data Loading
dist_proc_file_path = os.path.join(parent_dir_data, processed_distances_result_file)
distances_processed = torch.load(dist_proc_file_path, weights_only=False).to(device)
all_adap_file_path = os.path.join(parent_dir_data, adapters_result_file)
all_adapters = torch.load(all_adap_file_path, weights_only=False).to(device)


#### Calculation
predicted_adapters = torch.matmul(distances_processed, all_adapters)
predicted_adapters.to("cpu")


#### Saving Results
result_path = os.path.join(parent_dir_data, predictions_result_file)
torch.save(predicted_adapters, result_path)

for index in range(Number_of_LoRAs):
    weights_pred = predicted_adapters[index]
    file_name = os.path.join(adapters_folder_path, f"{index}.safetensors")
    file = safe_open(file_name, "pt")

    pos = 0
    state_dict_pred = {}
    for key in file.keys():
        actual = file.get_tensor(key)
        length = len(actual.flatten())
        state_dict_pred[key] = weights_pred[pos : pos + length].reshape(actual.shape)
        pos += length

    pred_file_path = os.path.join(
        predictions_folder_path, f"State_Dictionary{index}.pt"
    )
    torch.save(state_dict_pred, pred_file_path)


#### Memory Cleaning
del distances_processed
del all_adapters
del predicted_adapters
gc.collect()
torch.cuda.empty_cache()


#### End of Script Print
print(40 * "*")
print("PREDICTION PART FINISHED SUCCESSFULLY")

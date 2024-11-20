import os
import gc

import torch

from tqdm import tqdm

from LoRAs_Info import *
from config import *


#### Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    if GPU_Memory_Free_mb <= 36000:
        device = torch.device("cpu")

#### Data Loading
actual_weights_path = os.path.join(parent_dir_data, adapters_result_file)
actual_weights = torch.load(actual_weights_path).to(device)

predicted_weights_path = os.path.join(parent_dir_data, predictions_result_file)
predicted_weights = torch.load(predicted_weights_path).to(device)


#### Error Function Definition
def error_metric(vector1: torch.tensor, vector2: torch.tensor) -> torch.tensor:
    mse_loss = torch.nn.MSELoss()
    try:
        output = mse_loss(vector1, vector2)
    except:
        raise Exception("Error in calculation loss for weights!")
    return output


#### Error Calculation
all_weights_loss = []
for index in tqdm(range(Number_of_LoRAs)):
    all_weights_loss.append(
        error_metric(actual_weights[index], predicted_weights[index])
    )


#### Results Saving
all_weights_loss = torch.stack(all_weights_loss)
result_path = os.path.join(parent_dir_loss, weights_loss_result_file)
torch.save(all_weights_loss, result_path)


#### Memory Cleaning
del actual_weights
del predicted_weights
del all_weights_loss
gc.collect()
torch.cuda.empty_cache()

#### End of Run Print
print(40 * "*")
print("Done with calculating loss for all weights predictions.")

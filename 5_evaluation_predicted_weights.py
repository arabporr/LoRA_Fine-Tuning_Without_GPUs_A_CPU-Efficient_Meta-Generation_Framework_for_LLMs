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
actual_weights = torch.load(adapters_result_file_path, weights_only=False).to(device)
predicted_weights = torch.load(predictions_result_file_path, weights_only=False).to(
    device
)


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
    LoRA_Index_Loss = error_metric(actual_weights[index], predicted_weights[index])
    loss_file_path = os.path.join(weights_losses_folder_path, f"{index}.pt")
    torch.save(LoRA_Index_Loss, loss_file_path)
    all_weights_loss.append(LoRA_Index_Loss)


#### Results Saving
all_weights_loss = torch.stack(all_weights_loss)
torch.save(all_weights_loss, weights_losses_result_file_path)


#### Memory Cleaning
del actual_weights
del predicted_weights
del all_weights_loss
gc.collect()
torch.cuda.empty_cache()

#### End of Run Print
print(40 * "*")
print("Done with calculating loss for all weights predictions.")

from config import *

import torch


#### Algorithm and Calculations Part
def Nonlearnable(distances_vectors: torch.tensor) -> torch.tensor:
    mask = torch.nn.Softmin(dim=1)
    distances_vectors.fill_diagonal_(torch.inf)
    results = mask(distances_vectors)
    return results


# def Normalized(distances_vectors: torch.tensor) -> torch.tensor:


distances_raw = torch.load(distances_result_file_path, weights_only=False)

distances_processed = Nonlearnable(distances_raw)

torch.save(distances_processed, processed_distances_result_file_path)


#### End of Script Print
print(40 * "*")
print("COEFFICIENT CALCULATIONS PART FINISHED SUCCESSFULLY")

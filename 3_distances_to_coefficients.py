from config import *

import torch


#### Algorithm and Calculations Part
def Algorithm(distances_vectors: torch.tensor) -> torch.tensor:
    mask = torch.nn.Softmin(dim=1)
    # distances_vectors = torch.div(
    #     distances_vectors.T, distances_vectors.max(dim=1).values
    # ).T
    # distances_vectors = torch.nn.functional.normalize(distances_vectors, dim=1)
    distances_vectors.fill_diagonal_(torch.inf)
    results = mask(distances_vectors)
    return results


distances_raw = torch.load(os.path.join(parent_dir_data, distances_result_file))

distances_processed = Algorithm(distances_raw)

result_path = os.path.join(parent_dir_data, processed_distances_result_file)
torch.save(distances_processed, result_path)


#### End of Script Print
print(40 * "*")
print("COEFFICIENT CALCULATIONS PART FINISHED SUCCESSFULLY")

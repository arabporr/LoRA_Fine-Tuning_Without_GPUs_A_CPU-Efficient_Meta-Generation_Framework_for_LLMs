from config import *
from LoRAs_Info import *
from experiment_info import sub_variant

import torch

from tqdm import tqdm


#### Algorithm and Calculations Part
def Nonlearnable(distances_vectors: torch.tensor) -> torch.tensor:
    converter = torch.nn.Softmin(dim=1)
    distances_vectors.fill_diagonal_(torch.inf)
    results = converter(distances_vectors)
    return results


def Normalized(distances_vectors: torch.tensor) -> torch.tensor:
    all_adapters = torch.load(adapters_result_file_path, weights_only=False)
    converter = torch.nn.Softmin(dim=1)
    loss_func = torch.nn.MSELoss()

    # Split into train and test indices (80-20 split)
    torch.manual_seed(42)
    num_indices = Number_of_LoRAs
    all_indices = torch.arange(num_indices)
    shuffled_indices = all_indices[torch.randperm(num_indices)]
    split_idx = int(num_indices * 0.8)
    train_indices = torch.sort(shuffled_indices[:split_idx])[0]
    test_indices = torch.sort(shuffled_indices[split_idx:])[0]

    train_matrix = distances_vectors[train_indices][:, train_indices]
    target_adapters = all_adapters[train_indices]

    norm = torch.tensor(10000.0, requires_grad=True)
    optimizer = torch.optim.Adam([norm], lr=1e-2)

    for epoch in tqdm(range(100), desc="Optimizer iteration"):
        optimizer.zero_grad()

        # Normalize and process distances
        dist = train_matrix * (1 / norm)
        dist.fill_diagonal_(torch.inf)
        dist_processed = converter(dist)

        # Predict adapters and calculate loss
        predicted_adapters = torch.matmul(dist_processed, target_adapters)
        loss = loss_func(predicted_adapters, target_adapters)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Ensure norm stays positive (clamp it)
        with torch.no_grad():
            norm.clamp_(min=1.0)

        # Print loss for monitoring
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Norm: {norm.item()}")

    print(f"Best norm: {norm.item()}, Final loss: {loss.item()}")


if sub_variant == "nonlearnable":
    Function = Nonlearnable
elif sub_variant == "normalized":
    Function = Normalized
elif sub_variant == "learnable":
    sub_variant = Learnable
else:
    raise Exception("Invalid sub-variant for distance processing.")

distances_raw = torch.load(distances_result_file_path, weights_only=False)

distances_processed = Nonlearnable(distances_raw)


torch.save(distances_processed, processed_distances_result_file_path)


#### End of Script Print
print(40 * "*")
print("COEFFICIENT CALCULATIONS PART FINISHED SUCCESSFULLY")

from config import *
from LoRAs_Info import *
from experiment_info import sub_variant

import torch
import torch.nn as nn

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

    del all_adapters

    norm = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.Adam([norm], lr=10)

    for epoch in tqdm(range(20), desc="Optimizer iteration"):
        optimizer.zero_grad()

        dist = train_matrix * norm
        dist.fill_diagonal_(torch.inf)
        dist_processed = converter(dist)

        predicted_adapters = torch.matmul(dist_processed, target_adapters)
        loss = loss_func(predicted_adapters, target_adapters)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Norm: {norm.item()}")

    print(f"Best norm: {norm.item()}, Final loss: {loss.item()}")
    normalization_res = {"norm": norm, "train_indices": train_indices}
    torch.save(
        normalization_res, os.path.join(parent_dir_results, "Normalization_Info.pt")
    )

    distances_vectors = distances_vectors * norm
    distances_vectors.fill_diagonal_(torch.inf)
    results = converter(distances_vectors)
    return results


# the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def Learnable(distances_vectors: torch.tensor) -> torch.tensor:
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

    del all_adapters

    mlp_model = MLP()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
    diag_mask = torch.eye(train_matrix.shape[0], dtype=torch.bool)

    for epoch in tqdm(range(10), desc="Optimizer iteration"):
        optimizer.zero_grad()

        # Extract only off-diagonal elements
        off_diag_values = train_matrix[~diag_mask].unsqueeze(-1)  # Shape: (N*(N-1), 1)

        # Pass through MLP
        off_diag_transformed = mlp_model(off_diag_values).squeeze(
            -1
        )  # Shape: (N*(N-1))

        # Create a new matrix and insert transformed values
        dist_processed = torch.zeros_like(train_matrix)  # Initialize with zeros
        dist_processed[~diag_mask] = off_diag_transformed  # Restore off-diagonal values
        dist_processed[diag_mask] = torch.inf  # Set diagonal to inf
        dist_processed = converter(dist_processed)

        predicted_adapters = torch.matmul(dist_processed, target_adapters)
        loss = loss_func(predicted_adapters, target_adapters)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    normalization_res = {"model": mlp_model, "train_indices": train_indices}
    torch.save(normalization_res, os.path.join(parent_dir_results, "MLP_Info.pt"))

    distances_vectors = distances_vectors.unsqueeze(-1)
    distances_vectors = mlp_model(distances_vectors)
    distances_vectors = distances_vectors.squeeze(-1)
    distances_vectors.fill_diagonal_(torch.inf)
    results = converter(distances_vectors)
    return results


if sub_variant == "nonlearnable":
    Function = Nonlearnable
elif sub_variant == "normalized":
    Function = Normalized
elif sub_variant == "learnable":
    sub_variant = Learnable
elif sub_variant == "fewshot":
    Function = Nonlearnable
else:
    raise Exception("Invalid sub-variant for distance processing.")

distances_raw = torch.load(distances_result_file_path, weights_only=False)

distances_processed = Function(distances_raw)


torch.save(distances_processed, processed_distances_result_file_path)


#### End of Script Print
print(40 * "*")
print("COEFFICIENT CALCULATIONS PART FINISHED SUCCESSFULLY")

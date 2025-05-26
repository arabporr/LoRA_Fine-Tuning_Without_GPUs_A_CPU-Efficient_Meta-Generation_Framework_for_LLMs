import random

import torch
import torch.nn as nn

from tqdm import tqdm

from src.config.paths import all_adapters_file_location
from src.data.LoRAs_Info import Number_of_LoRAs


# the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def mlp_version_coefficient_calculator(
    distances_vectors: torch.tensor,
) -> torch.tensor:

    all_adapters = torch.load(all_adapters_file_location, weights_only=False)

    # Split into train and test indices (80-20 split)
    random.seed(42)
    train_set_indexes = random.sample(
        range(Number_of_LoRAs), int(0.8 * Number_of_LoRAs)
    )

    train_matrix = distances_vectors[train_set_indexes][:, train_set_indexes]

    target_adapters = all_adapters[train_set_indexes]

    del all_adapters

    training_logs = []
    mlp_model = MLP()
    softmin = torch.nn.Softmin(dim=1)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    diag_mask = torch.eye(train_matrix.shape[0], dtype=torch.bool)

    for epoch in tqdm(range(40), desc="Optimizer iteration"):
        optimizer.zero_grad()

        # Extract only off-diagonal elements
        inputs = train_matrix[~diag_mask].unsqueeze(-1)  # Shape: (N*(N-1), 1)

        # Pass through MLP
        outputs = mlp_model(inputs).squeeze(-1)  # Shape: (N*(N-1))

        # Create a new matrix and insert transformed values
        predicted_coefficients = torch.zeros_like(
            train_matrix)  # Initialize with zeros
        # Restore off-diagonal values
        predicted_coefficients[~diag_mask] = outputs
        predicted_coefficients[diag_mask] = torch.inf  # Set diagonal to inf
        predicted_coefficients = softmin(predicted_coefficients)

        predicted_adapters = torch.matmul(
            predicted_coefficients, target_adapters)
        loss = loss_func(predicted_adapters, target_adapters)

        loss.backward()
        optimizer.step()

        epoch_log = f"Epoch {epoch + 1}, Loss: {loss.item()}"
        training_logs.append(epoch_log)
        print(epoch_log)

    distances_vectors_flat = distances_vectors.unsqueeze(-1)
    mlp_outputs_flat = mlp_model(distances_vectors_flat)
    mlp_outputs = mlp_outputs_flat.squeeze(-1)
    coefficients = softmin(mlp_outputs.fill_diagonal_(torch.inf))
    return coefficients, [training_logs, mlp_model]

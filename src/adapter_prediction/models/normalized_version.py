import random

import torch

from tqdm import tqdm

from src.data.LoRAs_Info import Number_of_LoRAs

from src.config.paths import all_adapters_file_location
from src.data.LoRAs_Info import Number_of_LoRAs


def normalized_version_coefficient_calculator(
    distances_vectors: torch.tensor,
) -> torch.tensor:

    all_adapters = torch.load(all_adapters_file_location, weights_only=False)

    random.seed(42)
    train_set_indexes = random.sample(
        range(Number_of_LoRAs), int(0.8 * Number_of_LoRAs)
    )
    train_matrix = distances_vectors[train_set_indexes][:, train_set_indexes]
    target_adapters = all_adapters[train_set_indexes]

    del all_adapters

    training_logs = []
    softmin = torch.nn.Softmin(dim=1)
    loss_func = torch.nn.MSELoss()
    norm = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.Adam([norm], lr=10)

    for epoch in tqdm(range(20), desc="Optimizer iteration"):
        optimizer.zero_grad()

        dist = train_matrix * norm
        dist.fill_diagonal_(torch.inf)
        dist_processed = softmin(dist)

        predicted_adapters = torch.matmul(dist_processed, target_adapters)
        loss = loss_func(predicted_adapters, target_adapters)

        loss.backward()
        optimizer.step()

        epoch_log = f"Epoch {epoch + 1}, Loss: {loss.item()}"
        training_logs.append(epoch_log)
        print(epoch_log)

    print(f"Best norm: {norm.item()}, Final loss: {loss.item()}")

    distances_normalized = distances_vectors * norm
    distances_normalized.fill_diagonal_(torch.inf)
    coefficients = softmin(distances_normalized)
    return coefficients, [training_logs, norm]

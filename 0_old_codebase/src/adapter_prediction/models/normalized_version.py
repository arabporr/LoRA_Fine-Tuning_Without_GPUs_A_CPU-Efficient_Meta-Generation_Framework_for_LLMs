import random

import torch

from src.data.LoRAs_Info import Number_of_LoRAs

from sklearn.preprocessing import StandardScaler


def normalized_version_coefficient_calculator(
    distances_vectors: torch.tensor,
) -> torch.tensor:

    random.seed(42)
    train_set_indexes = random.sample(
        range(Number_of_LoRAs), int(0.8 * Number_of_LoRAs)
    )
    train_set = distances_vectors[train_set_indexes]
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaler.fit_transform(train_set.T)
    distances_vectors_scaled = scaler.transform(distances_vectors.T).T

    softmin = torch.nn.Softmin(dim=1)
    coefficients = softmin(distances_vectors_scaled.fill_diagonal_(torch.inf))
    return coefficients

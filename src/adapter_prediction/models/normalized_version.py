import torch

from src.data.LoRAs_Info import Number_of_LoRAs

from sklearn.preprocessing import StandardScaler


def normalized_version_coefficient_calculator(
    distances_vectors: torch.tensor,
) -> torch.tensor:
    scaler = StandardScaler(with_mean=False, with_std=False)
    distances_vectors_scaled = scaler.fit_transform(distances_vectors.T).T
    distances_vectors_scaled = torch.tensor(distances_vectors_scaled).float()

    softmin = torch.nn.Softmin(dim=1)
    coefficients = softmin(distances_vectors_scaled.fill_diagonal_(torch.inf))
    return coefficients

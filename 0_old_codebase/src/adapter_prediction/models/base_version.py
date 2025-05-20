import torch


def Base_version_coefficient_calculator(
    distances_vectors: torch.tensor,
) -> torch.tensor:
    softmin = torch.nn.Softmin(dim=1)
    coefficients = softmin(distances_vectors.fill_diagonal_(torch.inf))
    return coefficients

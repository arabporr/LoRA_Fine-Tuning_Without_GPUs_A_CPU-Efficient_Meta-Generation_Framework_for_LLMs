import argparse

import os
import gc

import torch

from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import save_file


from src.adapter_prediction.models.base_version import (
    base_version_coefficient_calculator,
)
from src.adapter_prediction.models.normalized_version import (
    normalized_version_coefficient_calculator,
)
from src.adapter_prediction.models.mlp_version import mlp_version_coefficient_calculator


from src.config.paths import (
    distances_dir,
    coefficients_dir,
    all_adapters_file_location,
    predicted_adapters_dir,
    raw_adapters_dir,
)
from src.data.LoRAs_Info import Number_of_LoRAs


def generate_adapters(metric: str, model: str) -> None:
    if metric == "WD":
        distances_file_location = os.path.join(
            distances_dir, "WD_all_distances.pt")
        distances_raw = torch.load(distances_file_location, weights_only=False)
    elif metric == "KL":
        distances_file_location = os.path.join(
            distances_dir, "KL_all_distances.pt")
        distances_raw = torch.load(distances_file_location, weights_only=False)
    elif metric == "JS":
        distances_file_location = os.path.join(
            distances_dir, "JS_all_distances.pt")
        distances_raw = torch.load(distances_file_location, weights_only=False)
    elif metric == "MMD":
        distances_file_location = os.path.join(
            distances_dir, "MMD_all_distances.pt")
        distances_raw = torch.load(distances_file_location, weights_only=False)
    else:
        raise Exception("Invalid distance metric!")

    # Coefficient calculation
    if model == "base_version":
        coefficients = base_version_coefficient_calculator(distances_raw)
    elif model == "normalized_version":
        coefficients, details = normalized_version_coefficient_calculator(
            distances_raw)
    elif model == "mlp_version":
        coefficients, details = mlp_version_coefficient_calculator(
            distances_raw)
    else:
        raise Exception("Invalid model for distance processing!")

    coefficients_metric_dir = os.path.join(coefficients_dir, metric)
    coefficients_metric_model_dir = os.path.join(
        coefficients_metric_dir, model)
    coefficients_file_location = os.path.join(
        coefficients_metric_model_dir, "all_coefficients.pt"
    )

    torch.save(coefficients, coefficients_file_location)

    if model == "mlp_version" or model == "normalized_version":
        details_file_location = os.path.join(
            coefficients_metric_model_dir, "logs_and_details.pt")
        torch.save(details, details_file_location)

    print(40 * "*")
    print("COEFFICIENT CALCULATIONS PART FINISHED SUCCESSFULLY")

    # Adapter generation
    all_adapters = torch.load(all_adapters_file_location, weights_only=False)

    predicted_adapters = torch.matmul(coefficients, all_adapters)

    del coefficients
    del all_adapters
    gc.collect()

    # Saving Results
    predicted_adapters_metric_dir = os.path.join(
        predicted_adapters_dir, metric)
    predicted_adapters_metric_model_dir = os.path.join(
        predicted_adapters_metric_dir, model
    )
    predicted_adapters_file_location = os.path.join(
        predicted_adapters_metric_model_dir, "all_predicted_adapters.pt"
    )
    torch.save(predicted_adapters, predicted_adapters_file_location)

    for index in tqdm(range(Number_of_LoRAs)):
        weights_pred = predicted_adapters[index]
        file_name = os.path.join(raw_adapters_dir, f"{index}.safetensors")
        file = safe_open(file_name, "pt")

        pos = 0
        state_dict_pred = {}
        for key in file.keys():
            actual = file.get_tensor(key)
            length = len(actual.flatten())
            state_dict_pred[key] = weights_pred[pos: pos + length].reshape(
                actual.shape
            )
            pos += length

        pred_file_path = os.path.join(
            predicted_adapters_metric_model_dir, f"State_Dictionary_{index}.safetensors"
        )
        save_file(state_dict_pred, pred_file_path)

    del predicted_adapters
    gc.collect()

    print(40 * "*")
    print("ADAPTER PREDICTION FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["WD", "KL", "JS", "MMD"])
    p.add_argument(
        "model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    args = p.parse_args()
    generate_adapters(metric=args.metric, model=args.model)

import argparse

import os
import gc

import torch

from tqdm import tqdm

from src.config.paths import (
    numerical_results_dir,
    all_adapters_file_location,
    predicted_adapters_dir,
)
from src.data.LoRAs_Info import Number_of_LoRAs


def numerical_evaluation(metric: str, model: str) -> None:
    #### Data Loading
    predicted_adapters_metric_dir = os.path.join(predicted_adapters_dir, metric)
    predicted_adapters_metric_model_dir = os.path.join(
        predicted_adapters_metric_dir, model
    )
    predicted_adapters_file_location = os.path.join(
        predicted_adapters_metric_model_dir, "all_predicted_adapters.pt"
    )
    predicted_weights = torch.load(predicted_adapters_file_location, weights_only=False)
    actual_weights = torch.load(all_adapters_file_location, weights_only=False)

    #### Error Function Definition
    def error_calculator(vector1: torch.tensor, vector2: torch.tensor) -> torch.tensor:
        mse_loss = torch.nn.MSELoss()
        try:
            output = mse_loss(vector1, vector2)
        except:
            raise Exception("Error in calculation loss for weights!")
        return output

    #### Error Calculation
    numerical_evaluation_metric_dir = os.path.join(numerical_results_dir, metric)
    numerical_evaluation_metric_model_dir = os.path.join(
        numerical_evaluation_metric_dir, model
    )
    all_adapters_numerical_loss = []
    for index in tqdm(range(Number_of_LoRAs)):
        adapter_numerical_loss = error_calculator(
            actual_weights[index], predicted_weights[index]
        )
        adapter_numerical_loss_file_location = os.path.join(
            numerical_evaluation_metric_model_dir, f"{index}.pt"
        )
        torch.save(adapter_numerical_loss, adapter_numerical_loss_file_location)
        all_adapters_numerical_loss.append(adapter_numerical_loss)

    all_adapters_numerical_loss_file_location = os.path.join(
        numerical_evaluation_metric_model_dir, f"{index}.pt"
    )
    all_adapters_numerical_loss = torch.stack(all_adapters_numerical_loss)
    torch.save(all_adapters_numerical_loss, all_adapters_numerical_loss_file_location)

    #### Memory Cleaning
    del actual_weights
    del predicted_weights
    del all_adapters_numerical_loss
    gc.collect()

    #### End of Run Print
    print(40 * "*")
    print("CALCULATING NUMERICAL ERROR FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["WD", "KL", "JS", "MMD"])
    p.add_argument(
        "model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    args = p.parse_args()
    numerical_evaluation(metric=args.metric, model=args.model)

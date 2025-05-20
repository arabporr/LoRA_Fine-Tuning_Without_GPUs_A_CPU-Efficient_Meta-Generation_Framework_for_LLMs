import argparse
import os

import pandas as pd

import torch

from tqdm import tqdm
import evaluate

from src.config.paths import (
    raw_datasets_dir,
    base_model_outputs_dir,
    fine_tuned_model_outputs_dir,
    models_outputs_dir,
    outputs_results_dir,
)
from src.data.LoRAs_Info import Number_of_LoRAs


def outputs_evaluation(metric: str, model: str) -> None:
    exact_match = {
        "dataset_index": [],
        "base": [],
        "gpu_fine_tuned": [],
        "predicted_model": [],
        "error": [],
    }
    erroneous_indexes = []

    rouge = evaluate.load("rouge")

    rouge_metric_results = {
        "dataset_index": [],
        "base": [],
        "gpu_fine_tuned": [],
        "predicted_model": [],
        "error": [],
    }

    for index in tqdm(Number_of_LoRAs):

        try:
            dataset_file_location = os.path.join(raw_datasets_dir, f"{index}.pt")
            dataset = torch.load(dataset_file_location, weights_only=False)

            base_model_output_results_file = os.path.join(
                base_model_outputs_dir, f"{index}.pt"
            )
            base_model_outputs = torch.load(
                base_model_output_results_file,
                weights_only=False,
            )

            gpu_fine_tuned_model_output_results_file = os.path.join(
                fine_tuned_model_outputs_dir, f"{index}.pt"
            )
            gpu_fine_tuned_model_outputs = torch.load(
                gpu_fine_tuned_model_output_results_file,
                weights_only=False,
            )

            predicted_models_output_results_metric_dir = os.path.join(
                models_outputs_dir, metric
            )
            predicted_models_output_results_metric_model_dir = os.path.join(
                predicted_models_output_results_metric_dir, model
            )
            predicted_model_output_results_file = os.path.join(
                predicted_models_output_results_metric_model_dir, f"{index}.pt"
            )
            predicted_model_outputs = torch.load(
                predicted_model_output_results_file,
                weights_only=False,
            )

            expected_outputs = [item["output"][0] for item in dataset["test"]]

            base_model_outputs_trimmed = [
                item[1]["generated_text"][0][len(item[0]["input"]) :]
                for item in base_model_outputs
            ]
            gpu_fine_tuned_model_outputs_trimmed = [
                item[1]["generated_text"][0][len(item[0]["input"]) :]
                for item in gpu_fine_tuned_model_outputs
            ]
            predicted_model_outputs_trimmed = [
                item[1]["generated_text"][0][len(item[0]["input"]) :]
                for item in predicted_model_outputs
            ]

            rouge_metric_results["dataset_index"].append(index)
            rouge_metric_results["base_model"].append(
                rouge.compute(
                    predictions=base_model_outputs_trimmed, references=expected_outputs
                )
            )
            rouge_metric_results["gpu_fine_tuned"].append(
                rouge.compute(
                    predictions=gpu_fine_tuned_model_outputs_trimmed,
                    references=expected_outputs,
                )
            )
            rouge_metric_results["predicted_adapters_model"].append(
                rouge.compute(
                    predictions=predicted_model_outputs_trimmed,
                    references=expected_outputs,
                )
            )
            rouge_metric_results["error"].append("Good")

            base_model_exact_match_in_answer = []
            gpu_fine_tuned__exact_match_in_answer = []
            predicted_model_exact_match_in_answer = []

            number_of_rows = len(dataset["test"])
            for index in range(number_of_rows):
                expected_output = dataset["test"][index]["output"][0].lower()

                if expected_output in base_model_outputs_trimmed[index].lower():
                    base_model_exact_match_in_answer.append(1)
                else:
                    base_model_exact_match_in_answer.append(0)

                if (
                    expected_output
                    in gpu_fine_tuned_model_outputs_trimmed[index].lower()
                ):
                    gpu_fine_tuned__exact_match_in_answer.append(1)
                else:
                    gpu_fine_tuned__exact_match_in_answer.append(0)

                if expected_output in predicted_model_outputs_trimmed[index].lower():
                    predicted_model_exact_match_in_answer.append(1)
                else:
                    predicted_model_exact_match_in_answer.append(0)

            accuracy_base_model = sum(base_model_exact_match_in_answer) / number_of_rows
            accuracy_gpu_fine_tuned = (
                sum(gpu_fine_tuned__exact_match_in_answer) / number_of_rows
            )
            accuracy_predicted_model = (
                sum(predicted_model_exact_match_in_answer) / number_of_rows
            )
            print(f"Accuracy Table for model {index}:")
            print("Base Foundation Model Accuracy: ", accuracy_base_model)
            print("GPU Fine Tuned Model Accuracy: ", accuracy_gpu_fine_tuned)
            print("Predicted Adapters Model Accuracy: ", accuracy_predicted_model)

            exact_match["dataset_index"].append(index)
            exact_match["base_model"].append(accuracy_base_model)
            exact_match["gpu_fine_tuned"].append(accuracy_gpu_fine_tuned)
            exact_match["predicted_model"].append(accuracy_predicted_model)
            exact_match["error"].append("Good")

        except Exception as e:
            rouge_metric_results["dataset_index"].append(index)
            rouge_metric_results["base_model"].append(0)
            rouge_metric_results["gpu_fine_tuned"].append(0)
            rouge_metric_results["predicted_model"].append(0)
            rouge_metric_results["error"].append(e)

            exact_match["dataset_index"].append(index)
            exact_match["base_model"].append(0)
            exact_match["gpu_fine_tuned"].append(0)
            exact_match["predicted_model"].append(0)
            exact_match["error"].append(e)
            erroneous_indexes.append({"index": index, "error": e})

    rouge_l = {
        "dataset": [],
        "base_model": [],
        "gpu_fine_tuned": [],
        "predicted_model": [],
    }
    rouge_l_sum = {
        "dataset": [],
        "base_model": [],
        "gpu_fine_tuned": [],
        "predicted_model": [],
    }

    for index in range(Number_of_LoRAs):
        if rouge_metric_results["error"][index] == "Good":
            rouge_l["dataset"].append(index)
            rouge_l["base_model"].append(
                rouge_metric_results["base_model"][index]["rougeL"]
            )
            rouge_l["gpu_fine_tuned"].append(
                rouge_metric_results["gpu_fine_tuned"][index]["rougeL"]
            )
            rouge_l["predicted_model"].append(
                rouge_metric_results["predicted_model"][index]["rougeL"]
            )

            rouge_l_sum["dataset"].append(index)
            rouge_l_sum["base_model"].append(
                rouge_metric_results["base_model"][index]["rougeLsum"]
            )
            rouge_l_sum["gpu_fine_tuned"].append(
                rouge_metric_results["gpu_fine_tuned"][index]["rougeLsum"]
            )
            rouge_l_sum["predicted_model"].append(
                rouge_metric_results["predicted_model"][index]["rougeLsum"]
            )

    outputs_results_metric_dir = os.path.join(outputs_results_dir, metric)
    outputs_results_metric_model_dir = os.path.join(outputs_results_metric_dir, model)
    outputs_results_file_location = os.path.join(
        outputs_results_metric_model_dir, "all_rouge_metrics.pt"
    )
    torch.save(rouge_metric_results, outputs_results_file_location)

    df_exact_match = pd.DataFrame.from_dict(exact_match)
    df_exact_match["predicted_model_better_than_base_model"] = (
        df_exact_match["predicted_model"] > df_exact_match["base_model"]
    )
    df_exact_match["predicted_model_better_than_gpu_fine_tuned"] = (
        df_exact_match["predicted_model"] > df_exact_match["gpu_fine_tuned"]
    )
    df_exact_match["predicted_model_diff_with_base_model"] = (
        df_exact_match["predicted_model"] - df_exact_match["base_model"]
    )
    df_exact_match["predicted_model_diff_with_gpu_fine_tuned"] = (
        df_exact_match["predicted_model"] - df_exact_match["gpu_fine_tuned"]
    )

    exact_match_results_file_location = os.path.join(
        outputs_results_metric_model_dir, "exact_match.csv"
    )
    df_exact_match.to_csv(exact_match_results_file_location)

    df_rouge_l = pd.DataFrame.from_dict(rouge_l)
    rouge_l_results_file_location = os.path.join(
        outputs_results_metric_model_dir, "rouge_l.csv"
    )
    df_rouge_l.to_csv(rouge_l_results_file_location)

    df_rouge_l_sum = pd.DataFrame.from_dict(rouge_l_sum)
    rouge_l_sum_results_file_location = os.path.join(
        outputs_results_metric_model_dir, "rouge_l_sum.csv"
    )
    df_rouge_l_sum.to_csv(rouge_l_sum_results_file_location)

    #### End of Run Print
    print(40 * "*")
    print("MODEL OUTPUTS ANALYSIS AND ERROR CALCULATION FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["WD", "KL", "JS", "MMD"])
    p.add_argument(
        "model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    args = p.parse_args()
    outputs_evaluation(metric=args.metric, model=args.model)

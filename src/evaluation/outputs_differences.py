import os

import pandas as pd

import torch

import evaluate


from src.config.config import max_threads_cpu_task
from src.config.paths import (
    all_distance_metrics,
    all_models,
    raw_datasets_dir,
    base_model_outputs_dir,
    fine_tuned_model_outputs_dir,
    models_outputs_dir,
    outputs_results_dir,
)
from src.data.LoRAs_Info import Number_of_LoRAs


all_results = [None] * Number_of_LoRAs
all_error_metrics = ["exact_match", "subsequence_match", "rougeL", "rougeLSum"]
rouge = evaluate.load("rouge")


def eval(model: str, data_index: int, outputs: str, refs: str):
    results = {"model": model, "data_index": data_index}
    exact_match = 0
    subsequence_match = 0
    for o, r in zip(outputs, refs):
        if o == r:
            exact_match += 1
        if r in o:
            subsequence_match += 1
    exact_match = exact_match/len(refs)
    subsequence_match = subsequence_match/len(refs)

    rouge_scores = rouge.compute(
        predictions=outputs,
        references=refs,
        use_stemmer=True
    )

    results["exact_match"] = exact_match
    results["subsequence_match"] = subsequence_match
    results["rougeL"] = rouge_scores["rougeL"]
    results["rougeLSum"] = rouge_scores["rougeLsum"]

    return results


def eval_handler(index: int):
    try:
        print("Working on dataset: ", index)
        dataset_file_location = os.path.join(
            raw_datasets_dir, f"{index}.pt")
        dataset = torch.load(dataset_file_location, weights_only=False)
        index_results = []

        reference_answer = [item["output"][0] for item in dataset["test"]]

        base_model_outputs_file = os.path.join(
            base_model_outputs_dir, f"{index}.pt"
        )
        base_model_outputs = torch.load(
            base_model_outputs_file,
            weights_only=False,
        )

        base_model_outputs_cleaned = [
            item[1]["generated_text"][0][len(item[0]["input"]):]
            for item in base_model_outputs
        ]
        index_results.append(
            eval("base_model", index, base_model_outputs_cleaned, reference_answer))

        gpu_fine_tuned_model_outputs_file = os.path.join(
            fine_tuned_model_outputs_dir, f"{index}.pt"
        )
        gpu_fine_tuned_model_outputs = torch.load(
            gpu_fine_tuned_model_outputs_file,
            weights_only=False,
        )

        gpu_fine_tuned_model_outputs_cleaned = [
            item[1]["generated_text"][0][len(item[0]["input"]):]
            for item in gpu_fine_tuned_model_outputs
        ]
        index_results.append(
            eval("gpu_fine_tuned", index, gpu_fine_tuned_model_outputs_cleaned, reference_answer))

        for metric in all_distance_metrics:
            for model in all_models:
                predicted_models_outputs_metric_dir = os.path.join(
                    models_outputs_dir, metric
                )
                predicted_models_outputs_metric_model_dir = os.path.join(
                    predicted_models_outputs_metric_dir, model
                )
                predicted_model_outputs_file = os.path.join(
                    predicted_models_outputs_metric_model_dir, f"{index}.pt"
                )
                predicted_model_outputs = torch.load(
                    predicted_model_outputs_file,
                    weights_only=False,
                )
                predicted_model_outputs_cleaned = [item[1]["generated_text"][0][len(
                    item[0]["input"]):] for item in predicted_model_outputs]

                index_results.append(
                    eval(f"predicted_model_({metric}_{model})", index, predicted_model_outputs_cleaned, reference_answer))
        all_results[index] = index_results
        return f"Done with {index}"
    except Exception as e:
        return f"Error with {index}! Error: {e}"


def outputs_evaluation() -> None:
    for i in range(Number_of_LoRAs):
        eval_handler(i)
    torch.save(all_results, os.path.join(
        outputs_results_dir, "all_results.pt"))

    for metric in all_error_metrics:
        models_results = {'base_model': [],
                          'gpu_fine_tuned': [],
                          'predicted_model_(WD_base_version)': [],
                          'predicted_model_(WD_normalized_version)': [],
                          'predicted_model_(WD_mlp_version)': [],
                          'predicted_model_(KL_base_version)': [],
                          'predicted_model_(KL_normalized_version)': [],
                          'predicted_model_(KL_mlp_version)': [],
                          'predicted_model_(JS_base_version)': [],
                          'predicted_model_(JS_normalized_version)': [],
                          'predicted_model_(JS_mlp_version)': [],
                          'predicted_model_(MMD_base_version)': [],
                          'predicted_model_(MMD_normalized_version)': [],
                          'predicted_model_(MMD_mlp_version)': []}
        for index in range(Number_of_LoRAs):
            dataset_res = all_results[index]
            for model_res in dataset_res:
                models_results[model_res["model"]].append(model_res[metric])

        df = pd.DataFrame.from_dict(models_results)
        resutls_file_metric_location = os.path.join(
            outputs_results_dir, f"{metric}_results.csv")
        df.to_csv(resutls_file_metric_location)

    # End of Run Print
    print(40 * "*")
    print("MODEL OUTPUTS ANALYSIS AND ERROR CALCULATION FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    outputs_evaluation()

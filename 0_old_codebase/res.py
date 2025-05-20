import torch
import evaluate
from tqdm import tqdm
import pandas as pd


Distane_Function = "MMD"

exact_match = {"dataset_index": [], "base": [], "lora": [], "pred": [], "error": []}
missing_loras = []

rouge = evaluate.load("rouge")

rouge_metric_results = {
    "dataset_index": [],
    "base": [],
    "lora": [],
    "pred": [],
    "error": [],
}

for index in tqdm(range(502)):
    try:
        pred = torch.load(
            f"R:\\Research\\FineTuning_LoRAs\\results\\{Distane_Function}\\nonlearnable\\predicted_loras_outputs\\{index}.pt",
            weights_only=False,
        )

        lora = torch.load(
            f"R:\\Research\\FineTuning_LoRAs\\data\\Outputs\\base_lora\\{index}.pt",
            weights_only=False,
        )

        base = torch.load(
            f"R:\\Research\\FineTuning_LoRAs\\data\\Outputs\\base_model\\{index}.pt",
            weights_only=False,
        )

        labels = [item[0]["output"][0] for item in base]
        base_out = [
            item[1]["generated_text"][0][len(item[0]["input"]) :] for item in base
        ]
        lora_out = [
            item[1]["generated_text"][0][len(item[0]["input"]) :] for item in lora
        ]
        pred_out = [
            item[1]["generated_text"][0][len(item[0]["input"]) :] for item in pred
        ]

        rouge_metric_results["dataset_index"].append(index)
        rouge_metric_results["base"].append(
            rouge.compute(predictions=base_out, references=labels)
        )
        rouge_metric_results["lora"].append(
            rouge.compute(predictions=lora_out, references=labels)
        )
        rouge_metric_results["pred"].append(
            rouge.compute(predictions=pred_out, references=labels)
        )
        rouge_metric_results["error"].append("Good")

        pred_correct = []
        base_correct = []
        lora_correct = []
        for data_index in range(len(pred)):
            if (
                base[data_index][0]["output"][0].lower()
                in base[data_index][1]["generated_text"][0][-10:].lower()
            ):
                base_correct.append(1)
            else:
                base_correct.append(0)

            if (
                lora[data_index][0]["output"][0].lower()
                in lora[data_index][1]["generated_text"][0][-10:].lower()
            ):
                lora_correct.append(1)
            else:
                lora_correct.append(0)

            if (
                pred[data_index][0]["output"][0].lower()
                in pred[data_index][1]["generated_text"][0][-10:].lower()
            ):
                pred_correct.append(1)
            else:
                pred_correct.append(0)

        accuracy_base = sum(base_correct) / len(base_correct)
        accuracy_lora = sum(lora_correct) / len(lora_correct)
        accuracy_pred = sum(pred_correct) / len(pred_correct)
        print(f"Accuracy Table for model {index}:")
        print("Base Model Accuracy: ", accuracy_base)
        print("LoRA Model Accuracy: ", accuracy_lora)
        print("Predicted Model Accuracy: ", accuracy_pred)

        exact_match["dataset_index"].append(index)
        exact_match["base"].append(accuracy_base)
        exact_match["lora"].append(accuracy_lora)
        exact_match["pred"].append(accuracy_pred)
        exact_match["error"].append("Good")

    except Exception as e:
        rouge_metric_results["dataset_index"].append(index)
        rouge_metric_results["base"].append(0)
        rouge_metric_results["lora"].append(0)
        rouge_metric_results["pred"].append(0)
        rouge_metric_results["error"].append(e)

        exact_match["dataset_index"].append(index)
        exact_match["base"].append(0)
        exact_match["lora"].append(0)
        exact_match["pred"].append(0)
        exact_match["error"].append(e)
        missing_loras.append({"index": index, "error": e})

rouge_l = {"dataset": [], "base": [], "lora": [], "pred": []}
rouge_l_sum = {"dataset": [], "base": [], "lora": [], "pred": []}

for index in range(502):
    if rouge_metric_results["error"][index] == "Good":
        rouge_l["dataset"].append(index)
        rouge_l["base"].append(rouge_metric_results["base"][index]["rougeL"])
        rouge_l["lora"].append(rouge_metric_results["lora"][index]["rougeL"])
        rouge_l["pred"].append(rouge_metric_results["pred"][index]["rougeL"])

        rouge_l_sum["dataset"].append(index)
        rouge_l_sum["base"].append(rouge_metric_results["base"][index]["rougeLsum"])
        rouge_l_sum["lora"].append(rouge_metric_results["lora"][index]["rougeLsum"])
        rouge_l_sum["pred"].append(rouge_metric_results["pred"][index]["rougeLsum"])


torch.save(rouge_metric_results, f"results/{Distane_Function}_AllRouges.pt")

df_exact_match = pd.DataFrame.from_dict(exact_match)
df_exact_match["pred_is_better_than_base"] = (
    df_exact_match["pred"] > df_exact_match["base"]
)
df_exact_match["pred_is_better_than_lora"] = (
    df_exact_match["pred"] > df_exact_match["lora"]
)
df_exact_match["pred_diff_base"] = df_exact_match["pred"] - df_exact_match["base"]
df_exact_match["pred_diff_lora"] = df_exact_match["pred"] - df_exact_match["lora"]
df_exact_match.to_csv(f"{Distane_Function}_ExactMatach.csv")

df_rouge_l = pd.DataFrame.from_dict(rouge_l)
df_rouge_l.to_csv(f"{Distane_Function}_RougeL.csv")

df_rouge_l_sum = pd.DataFrame.from_dict(rouge_l_sum)
df_rouge_l_sum.to_csv(f"{Distane_Function}_RougeLSum.csv")

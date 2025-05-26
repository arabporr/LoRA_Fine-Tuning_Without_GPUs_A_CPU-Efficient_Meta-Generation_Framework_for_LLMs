import os

from src.config.paths import all_distance_metrics, all_models, current_dir, models_outputs_dir
from src.data.LoRAs_Info import Number_of_LoRAs

script_content = """#!/bin/bash
#SBATCH --job-name=LoRA_inference_{metric}_{model}_{dataset_index}_{qos}
#SBATCH --qos={qos}
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time={time_limit}
#SBATCH --output=logs/inference_{metric}_{model}_{dataset_index}_{qos}/slurm-%j.out
#SBATCH --error=logs/inference_{metric}_{model}_{dataset_index}_{qos}/slurm-%j.err

# Environment Setup
cd
cd LoRA
source venv/bin/activate

# Run Experiments
python scripts/04_models_inference.py -metric={metric} -model={model} -dataset_index={dataset_index}
exit
"""


_cur_dir = os.path.join(
    current_dir, "running_experiments/3_parallel_inference/")

scripts_dir = os.path.join(_cur_dir, "scripts/")
os.makedirs(scripts_dir, exist_ok=True)


qos_and_time_limit = [("m5", "00:59:00"), ("m3", "03:59:00"),
                      ("m2", "07:59:00"), ("normal", "15:59:00")]


oreder_of_execution = 0
for q_t in qos_and_time_limit:
    qos, time_limit = q_t
    oreder_of_execution += 1
    run_command_file_location = os.path.join(
        _cur_dir, f"{oreder_of_execution}_server_commands_{qos}.sh")

    for metric in all_distance_metrics:
        for model in all_models:
            for dataset_index in range(Number_of_LoRAs):
                outputs_metric_dir = os.path.join(models_outputs_dir, metric)
                outputs_metric_model_dir = os.path.join(
                    outputs_metric_dir, model)
                output_file_location = os.path.join(
                    outputs_metric_model_dir, f"{dataset_index}.pt")
                if not os.path.exists(output_file_location):
                    script_filename = f"Slurm_{metric}_{model}_{dataset_index}_{qos}.sh"
                    with open(os.path.join(scripts_dir, script_filename), "w") as f:
                        f.write(script_content.format(metric=metric, model=model,
                                dataset_index=dataset_index, qos=qos, time_limit=time_limit))

                    with open(run_command_file_location, "a") as f:
                        f.write(
                            f"sbatch running_experiments/3_parallel_inference/scripts/Slurm_{metric}_{model}_{dataset_index}_{qos}.sh \n")

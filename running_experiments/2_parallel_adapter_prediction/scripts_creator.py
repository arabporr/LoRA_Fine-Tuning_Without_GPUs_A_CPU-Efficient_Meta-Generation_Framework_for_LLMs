import os

from src.config.paths import all_distance_metrics, all_models ,current_dir

script_content = """#!/bin/bash
#SBATCH --job-name=LoRA_prediction_{metric}_{model}
#SBATCH --qos=m2
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/prediction_{metric}_{model}/slurm-%j.out
#SBATCH --error=logs/prediction_{metric}_{model}/slurm-%j.err

# Environment Setup
cd
cd LoRA
source venv/bin/activate

# Run Experiments
python scripts/03_adapter_prediction.py -metric={metric} -model={model}
exit
"""


_cur_dir =  os.path.join(current_dir, "running_experiments/2_parallel_adapter_prediction/")

scripts_dir = os.path.join(_cur_dir, "scripts/")
os.makedirs(scripts_dir, exist_ok=True)

run_command_file_location = os.path.join(_cur_dir, "server_commands.sh")


for metric in all_distance_metrics:
    for model in all_models:
        script_filename = f"Slurm_{metric}_{model}.sh"
        with open(os.path.join(scripts_dir, script_filename), "w") as f:
            f.write(script_content.format(metric=metric, model=model))

        with open(run_command_file_location, "a") as f:
            f.write(f"sbatch running_experiments/2_parallel_adapter_prediction/scripts/Slurm_{metric}_{model}.sh \n")

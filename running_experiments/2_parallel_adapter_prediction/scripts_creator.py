import os

from src.config.paths import all_distance_metrics, all_models ,current_dir

script_content = """#!/bin/bash
#SBATCH --job-name=Meta_LoRA_prediction_{metric}_{model}
#SBATCH --qos=normal
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=prediction_{metric}_{model}/slurm-%j.out
#SBATCH --error=prediction_{metric}_{model}/slurm-%j.err

# Environment Setup
cd Meta_LoRA
conda activate venv

# Run Experiments
python scripts/03_adapter_prediction.py -metric={metric} -model={model}
exit
"""


_cur_dir =  os.path.join(current_dir, "running_experiments/2_parallel_adapter_prediction/")

scripts_dir = os.path.join(_cur_dir, "scripts/")
os.makedirs(scripts_dir, exist_ok=True)

for metric in all_distance_metrics:
    for model in all_models:
        script_filename = f"Slurm_{metric}_{model}.sh"
        with open(os.path.join(scripts_dir, script_filename), "w") as f:
            f.write(script_content.format(metric=metric, model=model))

        with open(os.path.join(_cur_dir, "server_commands.sh"), "a") as f:
            f.write(f"sbatch scripts/Slurm_{metric}_{model}.sh \n")

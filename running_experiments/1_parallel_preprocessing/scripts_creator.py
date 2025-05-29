import os

from src.config.paths import all_distance_metrics, current_dir

script_content = """#!/bin/bash
#SBATCH --job-name=LoRA_preprocessing_{metric}
#SBATCH --qos=m2
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/preprocessing_{metric}/slurm-%j.out
#SBATCH --error=logs/preprocessing_{metric}/slurm-%j.err

# Environment Setup
cd
cd Low_Rank_Adaptation_on_CPU
source venv/bin/activate

# Run Experiments
python scripts/02_preprocessing.py -metric={metric}
exit
"""


_cur_dir = os.path.join(
    current_dir, "running_experiments/1_parallel_preprocessing/")

scripts_dir = os.path.join(_cur_dir, "scripts/")
os.makedirs(scripts_dir, exist_ok=True)

run_command_file_location = os.path.join(_cur_dir, "server_commands.sh")

for metric in all_distance_metrics:
    script_filename = f"Slurm_{metric}.sh"
    with open(os.path.join(scripts_dir, script_filename), "w") as f:
        f.write(script_content.format(metric=metric))

    with open(run_command_file_location, "a") as f:
        f.write(
            f"sbatch running_experiments/1_parallel_preprocessing/scripts/Slurm_{metric}.sh \n")

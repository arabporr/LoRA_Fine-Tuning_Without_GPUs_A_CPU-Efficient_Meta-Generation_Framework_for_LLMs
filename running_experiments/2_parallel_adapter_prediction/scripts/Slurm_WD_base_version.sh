#!/bin/bash
#SBATCH --job-name=Meta_LoRA_prediction_WD_base_version
#SBATCH --qos=normal
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=prediction_WD_base_version/slurm-%j.out
#SBATCH --error=prediction_WD_base_version/slurm-%j.err

# Environment Setup
cd Meta_LoRA
conda activate venv

# Run Experiments
python scripts/03_adapter_prediction.py -metric=WD -model=base_version
exit

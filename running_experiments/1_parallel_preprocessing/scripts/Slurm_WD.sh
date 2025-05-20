#!/bin/bash
#SBATCH --job-name=Meta_LoRA_preprocessing_WD
#SBATCH --qos=normal
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=preprocessing_WD/slurm-%j.out
#SBATCH --error=preprocessing_WD/slurm-%j.err

# Environment Setup
cd Meta_LoRA
conda activate venv

# Run Experiments
python scripts/02_preprocessing.py -metric=WD
exit

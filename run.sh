#!/bin/bash

# Default values
metric="KL"
model="base_version"
inference=false

# All available metrics and models
all_metrics=(WD KL JS MMD)
all_models=(base_version normalized_version mlp_version)

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -metric) metric="$2"; shift ;;
    -model) model="$2"; shift ;;
    -inference) inference=true ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Run download script
python scripts/01_downloads.py

# Determine metrics to run
if [ "$metric" == "all" ]; then
  metrics=(${all_metrics[@]})
else
  metrics=($metric)
fi

# Determine models to run
if [ "$model" == "all" ]; then
  models=(${all_models[@]})
else
  models=($model)
fi

# Run preprocessing and adapter prediction
for m in "${metrics[@]}"; do
  python scripts/02_preprocessing.py -metric="$m"
  for mod in "${models[@]}"; do
    python scripts/03_adapter_prediction.py -metric="$m" -model="$mod"

    # Run inference if the flag is true
    if [ "$inference" = true ]; then
      for data_index in {0..501}; do
        python scripts/04_models_inference.py -metric="$m" -model="$mod" -data_index=$data_index
      done
    fi
  done
done


# Run evaluation if inference was true
if [ "$inference" = true ]; then
  python scripts/05_evaluations.py
fi
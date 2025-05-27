# Low-Rank Adaptation (LoRA) of Large Language Models on CPU

This repository accompanies our paper, **"Large Language Model Low-Rank Adaptation on CPU"**, which introduces a novel, CPU-based method for efficiently generating Low-Rank Adapters (LoRA) to fine-tune large language models (LLMs). The method addresses the computational limitations many users face due to the intensive GPU requirements of traditional fine-tuning.

---

## Paper Details

Our work introduces a theoretically grounded approach to fine-tuning large language models using Low-Rank Adapters without the need for GPU resources. The method leverages similarity metrics (Wasserstein Distance, Kullback–Leibler Divergence, Jensen-Shannon Divergence, and Maximum Mean Discrepancy) to combine pre-trained adapters based on dataset similarities. This combination approach generates lightweight adapters suitable for performing downstream tasks efficiently.

The paper demonstrates that even on CPUs, our proposed adapter generation pipelines significantly outperform the raw performance of base foundation models, achieving up to a 30.9% increase in Rouge-L scores. We provide comprehensive experimental validation using the Mistral-7B-Instruct-v0.2 model across 500 diverse natural language datasets.

**Important Note:** Although adapter generation and prediction (the main part of the paper) are optimized for CPU-only execution, the subsequent testing on 500 datasets (inference and evaluation processes) are computationally heavy and require GPU resources to generate model outputs for results tables.

---

## Setup Instructions

### Environment Setup

First, clone this repository and create your virtual environment:

```bash
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn
pip install -r requirements.txt
```

### Configuration

Place your HuggingFace token into `.env` in the `src/config` folder to avoid HuggingFace authentication errors:

```bash
HUGGINGFACE_TOKEN="your_token_here"
```

---

## Project Structure

```
.
├── data/
│   ├── raw_input/
│   │   ├── adapters/
│   │   └── datasets/
│   ├── preprocessed/
│   ├── coefficients/
│   └── predicted_adapters/
├── results/
│   ├── models_generated_outputs/
│   └── evaluation_outputs_results/
├── scripts/
├── src/
│   ├── adapter_prediction/
│   ├── config/
│   ├── evaluation/
│   ├── inference/
│   └── preprocessing/
└── running_experiments/
```

---

## Running the Experiments

### Quick Start

To download datasets, preprocess data, and predict adapters for a single metric and version:

```bash
python scripts/01_downloads.py
python scripts/02_preprocessing.py -metric=KL
python scripts/03_adapter_prediction.py -metric=KL -model=base_version
```

### Comprehensive Execution

Execute the full pipeline for all metrics (WD, KL, JS, MMD) and all adapter model versions (`base_version`, `normalized_version`, `mlp_version`):

```bash
python scripts/01_downloads.py

for metric in WD KL JS MMD; do
  python scripts/02_preprocessing.py -metric=$metric
  for model_version in base_version normalized_version mlp_version; do
    python scripts/03_adapter_prediction.py -metric=$metric -model=$model_version
  done
done
```

This comprehensive run takes approximately 6-8 hours on a single laptop with 20 CPU cores and 64 GB RAM.

### Inference and Evaluation (GPU required)

After generating adapters, run inference on your selected dataset (change `data_index` as needed):

```bash
for metric in WD KL JS MMD; do
  for model_version in base_version normalized_version mlp_version; do
    for data_index in {0..501}; do
      python scripts/04_models_inference.py -metric=$metric -model=$model_version -data_index=$data_index
    done
  done
done
```

Then, evaluate all generated model outputs:

```bash
python scripts/05_evaluations.py
```

**Note:** These steps are GPU-intensive and require significant computational resources.

---

## Notes

* **CPU-only Prediction:** The adapter prediction pipeline is optimized explicitly for CPU environments, reducing computational barriers significantly.
* **Heavy GPU Usage for Inference and Evaluation:** Despite CPU-optimized adapter prediction, this code performs a comprehensive testing which is not required for end used and thus inference require substantial GPU resources.
* **Performance:** Our pipeline provides a notable performance improvement over unmodified base models, making fine-tuning more accessible to resource-constrained users.

Thank you for your interest in our work! Feel free to open an issue if you encounter any problems.

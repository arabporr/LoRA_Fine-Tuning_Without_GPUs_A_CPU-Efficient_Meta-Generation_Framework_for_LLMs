# LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs

This repository accompanies our paper, **"LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs"** presented at **ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models (ICML 2025)**, which introduces a novel, CPU-based method for efficiently generating Low-Rank Adapters (LoRA) to fine-tune large language models (LLMs). The method addresses the computational limitations many users face due to the intensive GPU requirements of traditional fine-tuning.

```bibtex
@inproceedings{
  arabpour2025large,
  title={Large Language Model Low-Rank Adaptation on {CPU}},
  author={Reza Arabpour and Anastasis Kratsios and Haitz S{\'a}ez de Oc{\'a}riz Borde},
  booktitle={ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models},
  year={2025},
  url={https://openreview.net/forum?id=4xn3oNRHIx}
}
```

---

## Paper Details

Our work introduces a theoretically grounded approach to fine-tuning large language models using Low-Rank Adapters without the need for GPU resources. The method leverages similarity metrics (Wasserstein Distance, Kullback–Leibler Divergence, Jensen-Shannon Divergence, and Maximum Mean Discrepancy) to combine pre-trained adapters based on dataset similarities. This combination approach generates lightweight adapters suitable for performing downstream tasks efficiently.

The paper demonstrates that even on CPUs, our proposed adapter generation pipelines significantly outperform the raw performance of the base Mistral foundation model, achieving up to a 30.9% increase in Rouge-L scores. We provide comprehensive experimental validation using the Mistral-7B-Instruct-v0.2 model across 500 diverse natural language datasets.

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
pip install -r requirements.txt
```

### Configuration

Place your HuggingFace token into `.env` in the `src/config` folder to avoid HuggingFace authentication errors:

```bash
HUGGINGFACE_TOKEN="your_token_here"
```

### Customizing Datasets and Model Configuration

For our paper, we utilized an extensive and valuable collection of datasets and adapters from the ["Lots of LoRAs"](https://huggingface.co/Lots-of-LoRAs) Hugging Face repository, generously shared by Rickard Brüel Gabrielsson and his teammates (**THANK YOU!**).

However, you might wish to use different models or settings tailored to your own experiments. To achieve this:

- **Update Model and BitsAndBytes Settings:**
  Adjust model and quantization settings by modifying the configuration file located at:
  ```bash
  src/conf/config.py
  ```
- **Update Dataset and Adapter Information:**
  Change the datasets and adapter list by editing:
  ```bash
  src/data/LoRA_Info.py
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
│   │   └── models/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   └── preprocessing/
│       └── metrics/
└── running_experiments/
```

---

## Running the Experiments

### Quick Start

The following steps outline each stage explicitly to help you understand the process clearly:

**Note:** These steps showcase the explicit individual processes. For streamlined and efficient execution, use the provided `run_script.sh` script as described in the next (Comprehensive Execution) section.

**Step 1: Download Datasets**

Downloads necessary datasets and adapters from external sources available at the `LoRA_Info.py` file at the `src/data/` folder:

```bash
python scripts/01_downloads.py
```

**Step 2: Preprocess Data**

Processes (tokenize) raw datasets and calculate pairwise distances using the specified similarity metric (default is `KL`, but `WD`, `JS`, `MMD` are also available):

```bash
python scripts/02_preprocessing.py -metric=KL
```

**Step 3: Adapter Prediction**

Generates adapters based on the distances and selected model to integrate other adapters' information (`base_version` by default, `normalized_version`, `mlp_version` are also available):

```bash
python scripts/03_adapter_prediction.py -metric=KL -model=base_version
```



### Comprehensive Execution

A convenient script is provided (`run_script.sh`) to simplify the execution process:

- Basic execution with default parameters (metric: KL, model: base_version):

```bash
bash run_script.sh
```

- Custom execution:

```bash
bash run_script.sh -metric WD -model normalized_version
```

- Comprehensive execution for all metrics (`WD`, `KL`, `JS`, `MMD`) and all models (`base_version`, `normalized_version`, `mlp_version`):

```bash
bash run_script.sh -metric all -model all
```

**Note:** Running inference (`-inference` flag) is extremely time-consuming and computationally intensive. Ensure sufficient GPU resources are available. We tested our code on 32 RTX 6000 GPUs in parallel for 2 days since we had to test 500 datasets with 12 different settings resulting in generating outputs for more than 6000 times. If you have access to such computational power, for your convenience, we also provided the scripts for doing some steps (including the inference) in parallel at `running_experiments` folder.

- Execution with inference and evaluation of the outputs (time-intensive!):

```bash
bash run_script.sh -metric KL -model base_version -inference
```

**Note:** If you change your LoRA dataset bank, make sure to adjust the loop in the inference part of the script accordingly (`data_index` range).


## Notes

* **CPU-only Prediction:** The adapter prediction pipeline is optimized explicitly for CPU environments, reducing computational barriers significantly.
* **Heavy GPU Usage for Inference and Evaluation:** Despite CPU-optimized adapter prediction, this code performs a comprehensive testing which is not required for end used and thus inference require substantial GPU resources.
* **Performance:** Our pipeline provides a notable performance improvement over unmodified base models, making fine-tuning more accessible to resource-constrained users.

Thank you for your interest in our work! Feel free to open an issue if you encounter any problems.

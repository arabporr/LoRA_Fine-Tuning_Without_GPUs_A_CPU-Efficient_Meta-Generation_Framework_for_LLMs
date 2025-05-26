# srun -c 30 --gres=gpu:1 --mem=64GB --pty --time=06:00:00 bash

python -m venv venv

source venv/bin/activate

python -V

python -m pip install --upgrade pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

python scripts/01_downloads.py
python scripts/02_preprocessing.py -metric=WD
python scripts/02_preprocessing.py -metric=KL
python scripts/02_preprocessing.py -metric=JS
python scripts/02_preprocessing.py -metric=MMD

python scripts/03_adapter_prediction.py -metric=WD -model=base_version
python scripts/03_adapter_prediction.py -metric=WD -model=normalized_version
python scripts/03_adapter_prediction.py -metric=WD -model=mlp_version

python scripts/03_adapter_prediction.py -metric=KL -model=base_version
python scripts/03_adapter_prediction.py -metric=KL -model=normalized_version
python scripts/03_adapter_prediction.py -metric=KL -model=mlp_version

python scripts/03_adapter_prediction.py -metric=JS -model=base_version
python scripts/03_adapter_prediction.py -metric=JS -model=normalized_version
python scripts/03_adapter_prediction.py -metric=JS -model=mlp_version

python scripts/03_adapter_prediction.py -metric=MMD -model=base_version
python scripts/03_adapter_prediction.py -metric=MMD -model=normalized_version
python scripts/03_adapter_prediction.py -metric=MMD -model=mlp_version


# python running_experiments/1_parallel_preprocessing/scripts_creator.py
# bash running_experiments/1_parallel_preprocessing/server_commands.sh

# python running_experiments/2_parallel_adapter_prediction/scripts_creator.py
# bash running_experiments/2_parallel_adapter_prediction/server_commands.sh



#################### This part is the paper statistics generator that makes 500 LLMs and tests each of them, thus require heavy computation #####################
# now you can load and inference from any model you want by just running the line below with your desired dataset index instead of 0
# python scripts/04_models_inference.py -metric=WD -model=base_version -data_index=0

# finally when calculated all of them, run the following line to get the outputs
# python scripts/05_evaluations.py
python -m venv venv

source venv/bin/activate

python -V

python -m pip install --upgrade pip

pip install -r requirements.txt

python scripts/01_downloads.py
python src/preprocessing/dataset_tokenizer.py

python running_experiments/1_parallel_preprocessing/scripts_creator.py
bash running_experiments/1_parallel_preprocessing/server_commands.sh

python running_experiments/2_parallel_adapter_prediction/scripts_creator.py
bash running_experiments/2_parallel_adapter_prediction/server_commands.sh
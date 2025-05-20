# conda create -n venv python=3.10 -y
# conda activate venv
# python -V
# python -m pip install --upgrade pip
# conda install numpy scipy pandas scikit-learn matplotlib jupyterlab  -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m venv venv

source venv/bin/activate

python -V

python -m pip install --upgrade pip

pip install -r requirements.txt

python scripts/01_downloads.py

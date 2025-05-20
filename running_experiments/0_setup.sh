conda create -n venv python=3.10 -y

conda activate venv

python -V

python -m pip install --upgrade pip

conda install numpy scipy pandas scikit-learn matplotlib jupyterlab  -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt


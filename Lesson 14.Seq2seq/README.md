## Обробка мови
```
python --version
py -3.12 -m venv .venv
py -m venv .venv

.venv\Scripts\activate.bat
source .venv/bin/activate
python.exe -m pip install --upgrade pip
python3 -m pip install --upgrade pip


pip install torch

pip install transformers torch sentencepiece

Для CUDA GPU
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
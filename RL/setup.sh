python3 -m venv .venv
source .venv/bin/activate

# 1. First, uninstall problematic packages
pip uninstall numpy gym opencv-python -y

# 2. Install compatible versions
pip install numpy==1.23.5
pip install gym==0.26.2
pip install gym[mujoco]
pip install opencv-python==4.8.1.78

# 3. Install other dependenciesi
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm wandb

pip install mujoco-py


@echo off

REM Name of the conda environment
SET ENV_NAME=pytorch_benchmark

conda config --add channels https://mirrors.sustech.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.sustech.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.sustech.edu.cn/anaconda/cloud/bioconda/


REM Create a new conda environment with Python 3.10
conda create --name %ENV_NAME% python=3.10 -y

REM Activate the environment
CALL activate %ENV_NAME%

REM Install PyTorch 2.1.2 with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install pytorch-benchmark

REM Clone the repositories
git clone https://github.com/mli/transformers-benchmarks.git

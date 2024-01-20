#!/bin/bash

# Download and install Miniconda
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.9.0-0-MacOSX-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Initialize Conda
source $HOME/miniconda/etc/profile.d/conda.sh
conda init

conda config --add channels https://mirrors.sustech.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.sustech.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.sustech.edu.cn/anaconda/cloud/bioconda/


# Name of the conda environment
ENV_NAME="pytorch_benchmark"

# Create a new conda environment with Python 3.10
conda create --name $ENV_NAME python=3.10 -y

# Activate the environment
conda activate $ENV_NAME

# Install PyTorch 2.1.2 with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install pytorch-benchmark

# Clone the repositories
git clone https://github.com/mli/transformers-benchmarks.git

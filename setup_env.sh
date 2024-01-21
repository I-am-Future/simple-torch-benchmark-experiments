#!/bin/bash

# Download and install Miniconda
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.9.0-0-MacOSX-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Initialize Conda
source $HOME/miniconda/etc/profile.d/conda.sh
conda init

# Name of the conda environment
ENV_NAME="pytorch_benchmark"

# Create a new conda environment with Python 3.10
conda create --name $ENV_NAME python=3.10 -y

# Activate the environment
conda activate $ENV_NAME

# Install PyTorch 2.1.1 with CUDA support
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install pytorch-benchmark

conda install matplotlib pandas -y

# # Clone the repositories
# git clone https://github.com/mli/transformers-benchmarks.git

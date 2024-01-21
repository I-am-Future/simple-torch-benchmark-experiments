# Set execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Name of the conda environment
$ENV_NAME="pytorch_benchmark"

# Create a new conda environment with Python 3.10
conda create --name $ENV_NAME python=3.10 -y

# Activate the environment
conda activate $ENV_NAME

# Install PyTorch 2.1.1 with CUDA support
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

conda install matplotlib pandas -y

# Install pytorch-benchmark
pip install pytorch-benchmark

# # Clone the repositories
# git clone https://github.com/mli/transformers-benchmarks.git

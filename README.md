# Simple PyTorch Benchmark Experiments

This repo is used to answer a question puzzled me for long time: **Do WSL2 and Docker have computation performance lost in PyTorch compared to Native Windows?**

The answer is crucial because for many students and normal users, they don't have Linux environment. But some models are only available on Linux platform. We want to figure out if we have performance lost when we run the model on WSL2/Docker on Windows.

**Credits:**

The `micro_bench.ipynb` is modified from [transformer-benchmarks](https://github.com/mli/transformers-benchmarks).

The `torch_benchmark.py` is modified from [pytorch-benchmarks](https://github.com/LukasHedegaard/pytorch-benchmark).

The `train_benchmark` is modified from Assignment 2 of [DDA4220](http://www.zhangruimao.site/DDA4220.html): Deep Learning and Applications, 2023 Spring, CUHK-Shenzhen.

## 1. System configuration 

+ Hardware

Desktop: a commercial PC which type is "Lenovo Ren 7000K, 2021 version".

|         | Configuration                      |
| ------- | ---------------------------------- |
| CPU     | Intel Core i5-11400F               |
| GPU     | Nvidia GeForce RTX3060             |
| Memory  | Kingston DDR4 2400 MHz 32GB        |
| Storage | WD Black SN770 (run on PCIE 3.0x4) |

+ Platform

|            | Configuration                                                |
| ---------- | ------------------------------------------------------------ |
| Windows 11 | Version 22H2, Native running on the PC.                      |
| WSL2       | Ubuntu 18.04, Running on Windows 11.                         |
| Docker     | Desktop Version xx.xx, Backend is WSL2. Image is Nvidia HPC SDK 21.7, see [here](https://docs.nvidia.com/hpc-sdk/archive/21.7/index.html). |

+ Environment

We use miniconda environment in our experiments.

|                    | Configuration                                                |
| ------------------ | ------------------------------------------------------------ |
| Python             | 3.10                                                         |
| PyTorch:           | 2.1.1, with CUDA 11.8 Runtime                                |
| torchvision:       | 0.16.1                                                       |
| torchaudio:        | 2.1.1                                                        |
| torchtext:         |                                                              |
| pytorch-benchmark: | 0.3.6                                                        |
| transformer:       | Built from source on commit with hash `3f69f415adcbdaedec154ba8eac220ef3276975d` |



## 2. Installation

We use `setup_env.ps1` and `setup_env.sh` to install the environment on Windows and Linux respectively. You can find these scripts in the repo. Run by:

```shell
# Windows (powershell)
./setup_env.ps1

# Linux
sh setup_env.sh
```



## 3. TFLOPS Benchmark

**Description:** 

**Goal:** 

**Method:** 



## 4. Light Calculation Benchmark



## 5. Heavy Calculation Benchmark



## 6. Transformer Training Benchmark



## 7. Conclusion


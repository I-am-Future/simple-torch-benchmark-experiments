# Simple PyTorch Benchmark Experiments

This repo is used to answer a question puzzled me for long time: **Do WSL2 and Docker have computation performance lost in PyTorch compared to Native Windows?** (Short answer: No!)

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
| Docker     | Desktop Version 24.0.6, Backend is WSL2. Image is Nvidia HPC SDK 21.7, see [here](https://docs.nvidia.com/hpc-sdk/archive/21.7/index.html). |

+ Environment

We use miniconda environment in our experiments.

|                    | Configuration                                                |
| ------------------ | ------------------------------------------------------------ |
| Python             | 3.10                                                         |
| PyTorch:           | 2.1.1, with CUDA 11.8 Runtime                                |
| torchvision:       | 0.16.1                                                       |
| torchaudio:        | 2.1.1                                                        |
| torchtext:         | 0.6.0                                                        |
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



## 3. TFLOPS & Bandwidth Benchmark

**Goal:** Compare **basic calculation performance** differences on different environment.

**Method:** Timing for basic tensor operations, code is at first section of `micro_bench.ipynb`.

Results are at `results/micro_bench-<Env>.ipynb`.

+ Results (Matrix Multiplication):

Multiply two `n x n` matrices (`a @ b`).  

|        TFLOPS | n=128 |  n=512 | n=2048 | n=8192 |
| ------------: | ----: | -----: | -----: | ------ |
| **Windows11** |       |        |        |        |
| torch.float32 | 0.142 |  5.282 |  7.110 | 8.935  |
| torch.float16 | 0.134 |  6.823 | 23.234 | 25.453 |
|      **WSL2** |       |        |        |        |
| torch.float32 | 0.072 |  5.533 |  7.392 | 9.359  |
| torch.float16 | 0.246 | 11.542 | 24.641 | 26.478 |
|    **Docker** |       |        |        |        |
| torch.float32 | 0.203 |  5.546 |  7.391 | 9.304  |
| torch.float16 | 0.055 |  9.246 | 24.469 | 26.654 |

+ Results (Element-wise Matrix Multiplication):

Multiply a `n x n` matrix by 1.2 (`1.2 * a`).  

|        TFLOPS | n=65536 | n=262144 | n=1048576 | n=4194304 |
| ------------: | ------: | -------: | --------: | --------- |
| **Windows11** |         |          |           |           |
|        TFLOPS |   0.003 |    0.012 |     0.036 | 0.039     |
|          GB/s |  21.692 |   93.016 |   290.520 | 308.859   |
|      **WSL2** |         |          |           |           |
|        TFLOPS |   0.002 |    0.010 |     0.037 | 0.040     |
|          GB/s |  13.653 |   82.898 |   293.531 | 323.947   |
|    **Docker** |         |          |           |           |
|        TFLOPS |   0.004 |    0.019 |     0.037 | 0.040     |
|          GB/s |  34.424 |  148.973 |   292.299 | 323.464   |

**Conclusion:** For three environments, there are **no big difference** in the TFLOPS and bandwidth performance. 



## 4. FP32 CNN Benchmark

**Goal:** Compare **light-weight calculation performance** differences on different environment.

**Method:** Timing for inference on light-weight models (ResNet18 and ResNet101), code is at first section of `torch_benchmark.py`.

Results are at `results/<Env>.csv`. All tests are conducted with `batch_size = 32`, input shape `torch.Size[3, 224, 224]`

|    Batch Time | ResNet18 (# param=11.7M) | ResNet101 (# param=44.5M) |
| ------------: | -----------------------: | ------------------------: |
| **Windows11** |   21.063 ms +/- 1.112 ms |    95.355 ms +/- 6.126 ms |
|      **WSL2** | 20.552 ms +/- 479.427 us |  94.718 ms +/- 962.084 us |
|    **Docker** | 21.453 ms +/- 501.774 us |  94.861 ms +/- 770.122 us |

**Conclusion:** For three environments, there are **no big difference** in the performance of calculating some light weight tasks (Inference in ResNet18 and ResNet101). The Windows 11 has highest variance in computing.



## 5. FP16 Language Models Benchmark

**Goal:** Compare **heavy-weight calculation performance** differences on different environment.

**Method:** Timing for a simple forward and backward calculation on one Bert layer, code is at first section of `micro_bench.py`.

Results are at ``results/micro_bench-<Env>.ipynb`. 

We used "bert-large-uncased" model from hugging face. One layer contains around 12M parameters.

`batch` indicates the batch size. `seq_len` indicates the input sequence length. 

We also provided benchmarks for other models such as GPT2 and T5. But since the results are generally same, we don't show them here.

|              TFLOPS | batch=2 | batch=4 | batch=8 | batch=16 | batch=32 | batch=64 | batch=128 |
| ------------------: | ------: | ------: | ------: | -------- | -------- | -------- | --------- |
|       **Windows11** |         |         |         |          |          |          |           |
|     fwd seq_len=128 |   3.497 |   7.054 |  14.589 | 16.263   | 16.538   | 17.016   | 16.941    |
| fwd+bwd seq_len=128 |   3.910 |   7.888 |  15.973 | 18.015   | 18.770   | 19.375   | 19.572    |
|     fwd seq_len=512 |  12.992 |  14.077 |  14.303 | 14.703   | 14.687   | 14.950   | 14.948    |
| fwd+bwd seq_len=512 |  14.239 |  15.674 |  16.231 | 16.724   | 16.854   | 17.261   | 17.275    |
|            **WSL2** |         |         |         |          |          |          |           |
|     fwd seq_len=128 |   1.926 |   3.754 |  16.043 | 16.787   | 18.229   | 18.432   | 18.759    |
| fwd+bwd seq_len=128 |   1.912 |  10.977 |  17.070 | 18.661   | 19.932   | 20.449   | 20.847    |
|     fwd seq_len=512 |  13.967 |  14.522 |  15.617 | 15.735   | 16.088   | 15.834   | 15.939    |
| fwd+bwd seq_len=512 |  14.952 |  16.232 |  17.197 | 17.578   | 17.824   | 18.155   | 17.830    |
|          **Docker** |         |         |         |          |          |          |           |
|     fwd seq_len=128 |   4.801 |   9.171 |  15.920 | 16.627   | 18.256   | 18.387   | 18.801    |
| fwd+bwd seq_len=128 |   4.907 |  10.831 |  16.895 | 18.585   | 19.955   | 20.469   | 20.897    |
|     fwd seq_len=512 |  14.029 |  14.547 |  15.659 | 15.763   | 16.060   | 15.939   | 15.934    |
| fwd+bwd seq_len=512 |  14.990 |  16.256 |  17.228 | 17.611   | 17.910   | 18.160   | 17.844    |

**Conclusion:** For three environments, there are **no big difference** on the performance. we can find that when batch size is large (e.g., 32, 64, 128), WSL2 and Docker are a little bit faster than the Windows11.



## 6. Transformer Training Benchmark

**Goal:** Compare **overall training performance (i.e., testing GPU, CPU, Dick performance)** differences on different environment.

**Method:** Timing to train a transformer-based fake-tweets analysis model, code is at `train_benchmark/`.

Results are at ``results/<Env>.log`. 

Please refer the training arguments at `train_benchmark/train_transformer.py line 46`.

| Environment   | Training time (sec / epoch) |
| ------------- | --------------------------- |
| **Windows11** | 15.50                       |
| **WSL2**      | 15.52                       |
| **Docker**    | 14.36                       |

**Conclusion:** For three environments, there are **no big difference** on the performance. 



## 7. Conclusion

Whether on simple TFLOPS benchmarks, or FP16/FP32 calculation on light-weight models, or full-scale transformer training, there are no big performance difference among three environments, Windows11, WSL2 and Docker. 

This is probably because modern WSL2 runs on hardware virtualization, which has fewer overheads. In addition, since Docker runs on WSL2 backend, so Docker has similar performance as WSL2. 

Therefore, for Windows users, they can use WSL2 and Docker as deep learning environments without concerns for performance loss. 

# K-Means: Comparison between oneAPI (CPU/GPU) and OpenMP (CPU/GPU)

> Undergraduate Research Project – performance evaluation of *K-Means* on CPU and GPU using OpenMP and oneAPI

**Advisor:** Henrique Cota de Freitas

---

## 📌 Overview

This repository implements and compares the **K-Means** algorithm running on:

* **oneAPI/SYCL on CPU**
* **oneAPI/SYCL on GPU**
* **OpenMP on CPU** (classic parallelism)
* **OpenMP with *offloading* to GPU** (OpenMP Target)

The comparison focuses on **execution time** and **scalability (strong and weak)**.

> ⚠️ Note: The K-Means codebase is kept as unified as possible across variants, changing only the parallel primitives and *backend* details.

---

## 🎯 Objectives

1. Measure performance gains between CPU and GPU under **oneAPI** and **OpenMP**.
2. Investigate **strong scaling** (fixed problem size, increase *threads*) and **weak scaling** (grow the problem proportionally to resources).

---

## 🛠️ Requirements

* **Compilers**

  * oneAPI DPC++ (*icpx*) for SYCL (CPU/GPU)
  * LLVM/Clang with OpenMP (CPU) and **OpenMP Target** support for your GPU (NVIDIA/AMD/Intel) *(optional for GPU)*
  * GCC (alternative) for **OpenMP CPU**
* **Libraries**: C++17, OpenMP, SYCL/oneAPI
* **Python 3.8+** (for metrics and plots): `numpy`, `pandas`, `scikit-learn`, `matplotlib`

> 💡 For NVIDIA GPU with SYCL, you need a CUDA-compatible *plugin/backend* available at [https://codeplay.com/solutions/oneapi/plugins/](https://codeplay.com/solutions/oneapi/plugins/) . For OpenMP Target on NVIDIA, use Clang with `-fopenmp-targets=nvptx64-nvidia-cuda`.

---

## ⚙️ Build

#### oneAPI/SYCL (CPU)

```bash
icpx -O3 -std=c++17 -fsycl src/kmeans_sycl.cpp -o bin/kmeans_oneapi_cpu
```

#### oneAPI/SYCL (GPU)

> NVIDIA GPU via a compatible *backend* (e.g., nvptx):

```bash
icpx -O3 -std=c++17 -fsycl \
  -fsycl-targets=nvptx64-nvidia-cuda \
  -Xsycl-target-backend=nvptx64-nvidia-cuda:"--cuda-gpu-arch=sm_86" \
  src/kmeans_sycl.cpp -o bin/kmeans_oneapi_gpu
export SYCL_DEVICE_FILTER=gpu
```

#### OpenMP (CPU)

```bash
# Clang
clang++ -O3 -std=c++17 -fopenmp src/kmeans_omp_cpu.cpp -o bin/kmeans_omp_cpu
# or GCC
g++     -O3 -std=c++17 -fopenmp src/kmeans_omp_cpu.cpp -o bin/kmeans_omp_cpu
```

#### OpenMP Target (GPU)

> NVIDIA (Clang):

```bash
clang++ -O3 -std=c++17 -fopenmp \
  -fopenmp-targets=nvptx64-nvidia-cuda \
  -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_86 \
  src/kmeans_omp_gpu.cpp -o bin/kmeans_omp_gpu
export OMP_TARGET_OFFLOAD=MANDATORY
```

---

## ✅ Expected Results (guide)

* **oneAPI GPU** tends to outperform **oneAPI CPU** on large problems (higher parallelism and *throughput*), as long as data transfer doesn’t dominate.
* **OpenMP CPU** provides a good *baseline* and is easy to port.
* **OpenMP Target GPU** can approach SYCL performance on GPU, depending on *backend* maturity.
* Significant gains appear in scenarios with **large number of instances** and **moderate/high k**.

> Interpret *speedups* considering allocation/copy *overheads*, memory access patterns (coalescing), and parallelization policies in each variant.

---

## 📬 Contact

* **Authors:** Laura Costa, Luiz Fernando Frassi – *Puc MINAS/ CArt*
* **E-mail:** [laura.costa3141@gmail.com](mailto:laura.costa3141@gmail.com) and [luizfernandoe30@gmail.com](mailto:luizfernandoe30@gmail.com)

> Suggestions and *issues* are welcome!

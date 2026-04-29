# HPC-Vision: Matrix Multiplication Performance Benchmark

## Overview

HPC-Vision is a high-performance computing benchmark project that compares multiple implementations of matrix multiplication across different architectures:

- Serial (baseline CPU)
- OpenMP (multi-threaded CPU)
- OpenCL (portable GPU / heterogeneous devices)
- CUDA (NVIDIA GPU)

The project evaluates execution time, scalability, and speedup using both naive and tiled algorithms with automated benchmarking and visualization.

---

## Features

- Serial matrix multiplication baseline
- OpenMP parallel CPU implementation
- OpenCL GPU implementation
- CUDA GPU implementation
- Naive vs tiled optimization comparison
- Automated benchmark execution script
- Performance visualization plots
- Architecture speedup comparison tools

---

## Project Structure

HPC-Vision/ │ ├── cuda/ │   ├── matrix_cuda_naive.cu │   └── matrix_cuda_tiled.cu │ ├── opencl/ │   ├── matrix_kernel.cl │   └── matrix_opencl.c │ ├── openmp/ │   ├── matrix_openmp.c │   └── matrix_serial.c │ ├── python_modules/ │   ├── plot_results.py │   ├── plot_gpu_comparison.py │   ├── plot_speedup_architecture.py │   └── plot_thread_scaling.py │ ├── datasets/ │   └── generate_matrices.py │ ├── main.py ├── run_all.sh └── .gitignore

---

## Matrix Sizes Tested

Benchmarks are executed on:

256 × 256 512 × 512 1024 × 1024 1536 × 1536 2048 × 2048 3072 × 3072

Datasets are generated automatically using:

datasets/generate_matrices.py

---

## How to Run the Project

### Step 1: Generate datasets

python datasets/generate_matrices.py

### Step 2: Run benchmarks

chmod +x run_all.sh ./run_all.sh

This executes all implementations sequentially and records performance metrics.

---

## Generate Performance Plots

### GPU comparison (CUDA vs OpenCL)

python python_modules/plot_gpu_comparison.py

### Overall performance comparison

python python_modules/plot_results.py

### Thread scaling visualization (OpenMP)

python python_modules/plot_thread_scaling.py

### Architecture speedup comparison

python python_modules/plot_speedup_architecture.py

---

## Optimization Techniques Used

### Naive Algorithm

Standard triple-nested loop matrix multiplication:

O(n³)

Used as baseline performance reference.

---

### Tiled Algorithm

Improves cache locality and shared memory utilization by processing matrices in sub-blocks.

Benefits:

- Reduced global memory access
- Improved cache reuse
- Higher throughput on GPUs
- Better scalability for large matrices

---

## Performance Metrics Evaluated

The project measures:

- Execution time
- Speedup vs serial implementation
- Thread scalability (OpenMP)
- GPU vs CPU acceleration
- CUDA vs OpenCL performance comparison

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| C | Serial implementation |
| OpenMP | CPU parallelization |
| OpenCL | Cross-platform GPU computing |
| CUDA | NVIDIA GPU acceleration |
| Python | Visualization & automation |
| Bash | Benchmark execution pipeline |

---

## Sample Output Visualization

Generated plots include:

- Execution time vs matrix size
- Speedup vs architecture
- Thread scaling efficiency
- CUDA vs OpenCL comparison

These help evaluate performance tradeoffs across compute models.

---

## Applications

This benchmarking framework is useful for:

- High-performance computing coursework
- Parallel computing experiments
- GPU acceleration benchmarking
- Algorithm optimization studies
- Architecture comparison research

---

## Author

Ishaan Singh  
Thapar Institute of Engineering & Technology

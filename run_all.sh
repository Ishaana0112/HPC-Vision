#!/bin/bash

echo "Generating datasets..."
python3 datasets/generate_matrices.py

echo "Running Serial benchmarks..."
cd openmp
gcc matrix_serial.c -O3 -o serial_run

for size in 256 512 1024 1536 2048 3072
do
    ./serial_run $size
done

echo "Running OpenMP benchmarks..."
gcc matrix_openmp.c -Xpreprocessor -fopenmp \
-I/opt/homebrew/opt/libomp/include \
-L/opt/homebrew/opt/libomp/lib \
-lomp -o openmp_run

for size in 256 512 1024 1536 2048 3072
do
    OMP_NUM_THREADS=8 ./openmp_run $size
done

cd ..

echo "Running OpenCL benchmarks..."
cd opencl

gcc matrix_opencl.c -framework OpenCL -o opencl_run

for size in 256 512 1024 1536 2048 3072
do
    ./opencl_run $size
done

cd ..

echo "Generating plots..."

python3 python_modules/plot_results.py
python3 python_modules/plot_thread_scaling.py
python3 python_modules/plot_speedup_architecture.py
python3 python_modules/plot_gpu_comparison.py
python3 python_modules/generate_summary_table.py

echo "All benchmarks complete."
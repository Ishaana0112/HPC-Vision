import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project root dynamically (works regardless of where script is run from)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Helper loader with validation
import os

def load_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"CSV file is empty: {path}")
    return pd.read_csv(path)

# Load datasets safely
serial = load_csv_safe(RESULTS_DIR / "serial_results.csv")
openmp = load_csv_safe(RESULTS_DIR / "openmp_results.csv")
opencl = load_csv_safe(RESULTS_DIR / "opencl_results.csv")
cuda_naive = load_csv_safe(RESULTS_DIR / "cuda_naive_results.csv")
cuda_tiled = load_csv_safe(RESULTS_DIR / "cuda_tiled_results.csv")

# Rename columns clearly
# Standardize column names (expecting each CSV to contain: size,time)
serial.columns = ["size", "serial"]
openmp.columns = ["size", "openmp"]
opencl.columns = ["size", "opencl"]
cuda_naive.columns = ["size", "cuda_naive"]
cuda_tiled.columns = ["size", "cuda_tiled"]

# Merge datasets by matrix size
merged = serial.merge(openmp, on="size")
merged = merged.merge(opencl, on="size")
merged = merged.merge(cuda_naive, on="size")
merged = merged.merge(cuda_tiled, on="size")

# Plot runtime comparison
plt.figure(figsize=(10,6))

plt.plot(merged["size"], merged["serial"], marker='o', label="Serial CPU")
plt.plot(merged["size"], merged["openmp"], marker='o', label="OpenMP CPU")
plt.plot(merged["size"], merged["opencl"], marker='o', label="OpenCL GPU")
plt.plot(merged["size"], merged["cuda_naive"], marker='o', label="CUDA Naive GPU")
plt.plot(merged["size"], merged["cuda_tiled"], marker='o', label="CUDA Tiled GPU")

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication Performance Comparison")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.savefig(RESULTS_DIR / "final_architecture_comparison.png")
plt.show()

print("Saved: results/final_architecture_comparison.png")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Load datasets
serial = pd.read_csv(RESULTS_DIR / "serial_results.csv")
openmp = pd.read_csv(RESULTS_DIR / "openmp_results.csv")
opencl = pd.read_csv(RESULTS_DIR / "opencl_results.csv")
cuda_naive = pd.read_csv(RESULTS_DIR / "cuda_naive_results.csv")
cuda_tiled = pd.read_csv(RESULTS_DIR / "cuda_tiled_results.csv")

# Merge datasets
merged = serial.merge(openmp, on="size", suffixes=("_serial", "_openmp"))
merged = merged.merge(opencl, on="size")
merged = merged.merge(cuda_naive, on="size", suffixes=("", "_cuda_naive"))
merged = merged.merge(cuda_tiled, on="size", suffixes=("", "_cuda_tiled"))

# Rename columns clearly
merged.columns = [
    "size",
    "serial",
    "openmp",
    "opencl",
    "cuda_naive",
    "cuda_tiled"
]

# Compute speedups
merged["openmp_speedup"] = merged["serial"] / merged["openmp"]
merged["opencl_speedup"] = merged["serial"] / merged["opencl"]
merged["cuda_naive_speedup"] = merged["serial"] / merged["cuda_naive"]
merged["cuda_tiled_speedup"] = merged["serial"] / merged["cuda_tiled"]

# Plot
plt.figure(figsize=(10,6))

plt.plot(merged["size"], merged["openmp_speedup"], marker='o', label="OpenMP Speedup")
plt.plot(merged["size"], merged["opencl_speedup"], marker='o', label="OpenCL Speedup")
plt.plot(merged["size"], merged["cuda_naive_speedup"], marker='o', label="CUDA Naive Speedup")
plt.plot(merged["size"], merged["cuda_tiled_speedup"], marker='o', label="CUDA Tiled Speedup")

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Speedup vs Serial")
plt.title("Architecture Speedup Comparison")
plt.yscale("log")
plt.legend()
plt.grid(True)

output_path = RESULTS_DIR / "speedup_vs_architecture.png"
plt.savefig(output_path)
plt.show()

print(f"Saved: {output_path}")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

opencl = pd.read_csv(RESULTS_DIR / "opencl_results.csv")
cuda_naive = pd.read_csv(RESULTS_DIR / "cuda_naive_results.csv")
cuda_tiled = pd.read_csv(RESULTS_DIR / "cuda_tiled_results.csv")

merged = opencl.merge(cuda_naive, on="size")
merged = merged.merge(cuda_tiled, on="size")

merged.columns = [
    "size",
    "opencl",
    "cuda_naive",
    "cuda_tiled"
]

plt.figure(figsize=(10,6))

plt.plot(merged["size"], merged["opencl"],
         marker='o', label="OpenCL GPU")

plt.plot(merged["size"], merged["cuda_naive"],
         marker='o', label="CUDA Naive GPU")

plt.plot(merged["size"], merged["cuda_tiled"],
         marker='o', label="CUDA Tiled GPU")

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Execution Time (seconds)")
plt.title("GPU Architecture Comparison (OpenCL vs CUDA)")
plt.legend()
plt.grid(True)

output_path = RESULTS_DIR / "gpu_architecture_comparison.png"
plt.savefig(output_path)

plt.show()

print("Saved:", output_path)
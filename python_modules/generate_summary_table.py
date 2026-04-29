import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Load datasets
serial = pd.read_csv(RESULTS_DIR / "serial_results.csv")
openmp = pd.read_csv(RESULTS_DIR / "openmp_results.csv")
opencl = pd.read_csv(RESULTS_DIR / "opencl_results.csv")
cuda_naive = pd.read_csv(RESULTS_DIR / "cuda_naive_results.csv")
cuda_tiled = pd.read_csv(RESULTS_DIR / "cuda_tiled_results.csv")

# Rename columns
serial.columns = ["size", "serial"]
openmp.columns = ["size", "openmp"]
opencl.columns = ["size", "opencl"]
cuda_naive.columns = ["size", "cuda_naive"]
cuda_tiled.columns = ["size", "cuda_tiled"]

# Merge everything
merged = serial.merge(openmp, on="size")
merged = merged.merge(opencl, on="size")
merged = merged.merge(cuda_naive, on="size")
merged = merged.merge(cuda_tiled, on="size")

# Compute best method per row
methods = ["serial", "openmp", "opencl", "cuda_naive", "cuda_tiled"]

merged["best_method"] = merged[methods].idxmin(axis=1)
merged["max_speedup"] = merged["serial"] / merged[methods].min(axis=1)

# Save summary table
output_path = RESULTS_DIR / "summary_results_table.csv"
merged.to_csv(output_path, index=False)

print("Saved:", output_path)
print("\nPreview:\n")
print(merged)
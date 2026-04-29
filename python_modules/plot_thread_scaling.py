import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/thread_scaling_results.csv")

sizes = sorted(df["size"].unique())

# Runtime vs Threads
for size in sizes:
    subset = df[df["size"] == size]
    plt.plot(subset["threads"], subset["time"], marker="o", label=f"{size}x{size}")

plt.xlabel("Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Runtime vs Threads")
plt.legend()
plt.savefig("results/runtime_vs_threads.png")
plt.clf()


# Speedup vs Threads
for size in sizes:
    subset = df[df["size"] == size]
    baseline = subset[subset["threads"] == 1]["time"].values[0]
    speedup = baseline / subset["time"]
    plt.plot(subset["threads"], speedup, marker="o", label=f"{size}x{size}")

plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.savefig("results/speedup_vs_threads.png")
plt.clf()


# Efficiency vs Threads
for size in sizes:
    subset = df[df["size"] == size]
    baseline = subset[subset["threads"] == 1]["time"].values[0]
    efficiency = (baseline / subset["time"]) / subset["threads"]
    plt.plot(subset["threads"], efficiency, marker="o", label=f"{size}x{size}")

plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.title("Parallel Efficiency vs Threads")
plt.legend()
plt.savefig("results/efficiency_vs_threads.png")
plt.clf()


# Runtime vs Matrix Size (8 threads only)
subset = df[df["threads"] == 8]
plt.plot(subset["size"], subset["time"], marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Runtime vs Matrix Size (8 Threads)")
plt.savefig("results/runtime_vs_matrix_size.png")
plt.clf()

print("Scaling plots generated successfully.")
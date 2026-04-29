import subprocess
import os

def run_serial():
    print("\nRunning Serial Version...")
    subprocess.run(["gcc", "openmp/matrix_serial.c", "-o", "openmp/serial_run"])
    subprocess.run(["./openmp/serial_run"])


def run_openmp():
    print("\nRunning OpenMP Version...")

    compile_cmd = [
        "gcc",
        "openmp/matrix_openmp.c",
        "-Xpreprocessor",
        "-fopenmp",
        "-I/opt/homebrew/opt/libomp/include",
        "-L/opt/homebrew/opt/libomp/lib",
        "-lomp",
        "-o",
        "openmp/openmp_run"
    ]

    subprocess.run(compile_cmd)

    for threads in [4, 8]:
        print(f"\nRunning with {threads} threads...")
        os.environ["OMP_NUM_THREADS"] = str(threads)
        subprocess.run(["./openmp/openmp_run"])


def run_cuda_note():
    print("\nCUDA versions already executed inside Google Colab.")
    print("Refer stored results for CUDA Naive and CUDA Tiled.")


def plot_results():
    print("\nGenerating performance graphs...")
    subprocess.run(["python3", "python_modules/plot_results.py"])


if __name__ == "__main__":
    run_serial()
    run_openmp()
    run_cuda_note()
    plot_results()
import numpy as np

sizes = [256, 512, 1024, 1536, 2048, 3072]

for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    np.savetxt(f"matrixA_{n}.txt", A)
    np.savetxt(f"matrixB_{n}.txt", B)

print("Datasets generated successfully.")
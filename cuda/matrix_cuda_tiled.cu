

// NOTE:
// This CUDA matrix multiplication program uses shared-memory tiling
// and is executed in Google Colab using an NVIDIA T4 GPU because
// Apple Silicon Macs do not support CUDA runtime locally.
// Matrix size used for benchmarking: 512 × 512.

#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define TILE 16

__global__ void tiledMatrixMul(float *A, float *B, float *C)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++)
    {
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    int size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    tiledMatrixMul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("CUDA Tiled Output sample: %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
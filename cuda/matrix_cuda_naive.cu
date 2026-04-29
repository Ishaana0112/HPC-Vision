// NOTE:
// This CUDA matrix multiplication program is executed in Google Colab
// using an NVIDIA T4 GPU because Apple Silicon Macs do not support CUDA.
// Matrix size used for benchmarking: 512 × 512.
#include <stdio.h>
#include <cuda_runtime.h>
#define N 512

__global__ void matrixMul(float *A, float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float sum = 0.0f;

        for(int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    int size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i=0;i<N*N;i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }

    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(N/16,N/16);

    matrixMul<<<blocks,threads>>>(d_A,d_B,d_C);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    printf("CUDA Naive Output sample: %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
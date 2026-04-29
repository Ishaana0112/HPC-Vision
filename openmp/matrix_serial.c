#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);

    printf("Running matrix size: %d x %d\n", N, N);

    float *A, *B, *C;

    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*N*sizeof(float));
    C = (float*)malloc(N*N*sizeof(float));

    if(!A || !B || !C)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for(int i = 0; i < N*N; i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    clock_t start = clock();

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for(int k = 0; k < N; k++)
            {
                sum += A[i*N + k] * B[k*N + j];
            }

            C[i*N + j] = sum;
        }
    }

    // Prevent compiler optimization removing computation
    double checksum = 0.0;
    for(int i = 0; i < N*N; i++)
    {
        checksum += C[i];
    }
    printf("Checksum: %f\n", checksum);

    clock_t end = clock();

    printf("Serial CPU time: %f seconds\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);

    return 0;
}
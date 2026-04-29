#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    printf("Running matrix size: %d x %d\n", N, N);

    float *A, *B, *C;

    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*N*sizeof(float));
    C = (float*)malloc(N*N*sizeof(float));

    char fileAname[256], fileBname[256];
    sprintf(fileAname, "../datasets/matrixA_%d.txt", N);
    sprintf(fileBname, "../datasets/matrixB_%d.txt", N);
    printf("Loading dataset: %s\n", fileAname);
    printf("Loading dataset: %s\n", fileBname);

    FILE *fileA = fopen(fileAname, "r");
    FILE *fileB = fopen(fileBname, "r");

    if (!fileA || !fileB)
    {
        printf("Error opening dataset files.\n");
        return 1;
    }

    for(int i = 0; i < N*N; i++)
    {
        fscanf(fileA, "%f", &A[i]);
        fscanf(fileB, "%f", &B[i]);
    }

    fclose(fileA);
    fclose(fileB);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum = 0;

            for(int k=0;k<N;k++)
            {
                sum += A[i*N+k]*B[k*N+j];
            }

            C[i*N+j] = sum;
        }
    }

    double end = omp_get_wtime();

    printf("OpenMP CPU time: %f seconds\n", end-start);

    free(A);
    free(B);
    free(C);

    return 0;
}
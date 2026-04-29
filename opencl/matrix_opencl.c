// OpenCL matrix multiplication benchmark implementation
// Executes matrix multiplication on the Apple GPU via OpenCL
// Loads datasets dynamically and reports execution time

#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);

    char fileAname[128], fileBname[128];
    sprintf(fileAname, "../datasets/matrixA_%d.txt", N);
    sprintf(fileBname, "../datasets/matrixB_%d.txt", N);

    FILE *fileA = fopen(fileAname, "r");
    FILE *fileB = fopen(fileBname, "r");

    if (!fileA || !fileB)
    {
        printf("Dataset files not found.\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++)
    {
        fscanf(fileA, "%f", &A[i]);
        fscanf(fileB, "%f", &B[i]);
    }

    fclose(fileA);
    fclose(fileB);

    FILE *kernelFile = fopen("matrix_kernel.cl", "r");
    if (!kernelFile)
    {
        printf("Kernel file not found.\n");
        return 1;
    }

    char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &ret);

    double start_time = (double)clock() / CLOCKS_PER_SEC;

    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   (const size_t *)&source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "matrixMul", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);

    size_t global_item_size[2] = {N, N};

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                global_item_size, NULL, 0, NULL, NULL);

    clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                              bytes, C, 0, NULL, NULL);

    double end_time = (double)clock() / CLOCKS_PER_SEC;

    double gpu_time = end_time - start_time;

    printf("OpenCL GPU time: %f seconds\n", gpu_time);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(c_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);
    free(source_str);

    return 0;
}
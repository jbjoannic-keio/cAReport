#include <stdio.h>
#include <stdlib.h>

const int N = (1024 * 1024);
const int BSIZE = 1024;

__global__ void gpu_kernel(float *d_A, float *d_B, float *d_C, int len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        d_C[i] = d_A[i] + d_B[i];
}

__host__ void cpu_kernel(float *d_A, float *d_B, float *d_C, int len)
{
    for (int i = 0; i < len; i++)
        d_C[i] = d_A[i] + d_B[i];
}

int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C;  // for host memory
    float *d_A, *d_B, *d_C;  // for device memory
    float result;     // resut
    dim3 grid(N/BSIZE, 1, 1), block(BSIZE, 1, 1); // grid and block size

    /* host memory allocation */
    h_A = (float *)malloc(sizeof(float) * N);
    h_B = (float *)malloc(sizeof(float) * N);
    h_C = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; h_B[i] = 2.0f; h_C[i] = 0.0f;
    }

    /* device memory allocation */
    cudaMalloc((void **)&d_A, sizeof(float) * N);
    cudaMalloc((void **)&d_B, sizeof(float) * N);
    cudaMalloc((void **)&d_C, sizeof(float) * N);

    /* copy data the host to the device */
    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    /* The host calles the karnel */ 
    gpu_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

    /*  Result write back */
    cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    /*  Release device memory */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    /* check the result for GPU */
    result = 0.0;
    for (int i = 0; i < N; ++i)
        result += h_C[i];
    result /= (float) N;
    printf("GPU: result = %f\n", result);

    /* check the result for CPU */
    result = 0.0;
    cpu_kernel(h_A, h_B, h_C, N);
    for (int i = 0; i < N; ++i)
        result += h_C[i];
    result /= (float) N;
    printf("CPU: result = %f\n", result);

    /*  Release host memory  */
    free(h_A); free(h_B); free(h_C);

  return 0;
}

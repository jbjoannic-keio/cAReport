#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1024     /* problem size */
#define NSTEP 1000 /* Num. of time steps */
#define KAPPA 0.2

#define BSIZE 32

double f[2][N][N] = {0};

__global__ void gpu_kernel(double *d_f)
{
  __shared__ double s_f[BSIZE + 2][BSIZE + 2];
  int i = blockIdx.x * blockDim.x + threadIdx.x; // line // index in global memory
  int j = blockIdx.y * blockDim.y + threadIdx.y; // column // index in global memory
  int lI = threadIdx.x + 1;                      // with radius offset // index in shared memory
  int lJ = threadIdx.y + 1;                      // with radius offset // index in shared memory

  // load data from global memory to shared memory
  s_f[lI][lJ] = d_f[i * N + j];
  if (threadIdx.x == 0) // complete halo region the first row load the a row before the block from global memory and a row after the block from global memory
  {
    if (i != 0)
      s_f[lI - 1][lJ] = d_f[(i - 1) * N + j];
    if (i + BSIZE <= N - 1)
      s_f[lI + BSIZE][lJ] = d_f[(i + BSIZE) * N + j];
  }

  if (threadIdx.y == 0)
  {
    if (j != 0)
      s_f[lI][lJ - 1] = d_f[i * N + j - 1];
    if (j + BSIZE <= N - 1)
      s_f[lI][lJ + BSIZE] = d_f[i * N + j + BSIZE];
  }
  __syncthreads();

  // main computation
  double result = s_f[lI][lJ] + KAPPA * (s_f[lI][lJ - 1] + s_f[lI][lJ + 1] + s_f[lI - 1][lJ] + s_f[lI + 1][lJ] - 4.0 * s_f[lI][lJ]);
  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
    d_f[i * N + j] = result;
  else
    d_f[i * N + j] = 0.0;
}

__host__ void cpu_kernel()
{
  int st, i, j;
  for (st = 0; st < NSTEP; st++)
  {
    for (i = 1; i < N - 1; i++)
    {
      for (j = 1; j < N - 1; j++)
      {
        f[1][i][j] = f[0][i][j] + KAPPA * (f[0][i][j - 1] + f[0][i][j + 1] + f[0][i - 1][j] + f[0][i + 1][j] - 4.0 * f[0][i][j]);
      }
    }

    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
        f[0][i][j] = f[1][i][j];
  }
}

void init()
{
  int i, j;

  /* data initialization */
  for (i = 1; i < N - 1; i++)
  {
    for (j = 1; j < N - 1; j++)
    {
      f[0][i][j] = 1.0;
      if (i > N / 4 && i < 3 * N / 4 && j > N / 4 && j < 3 * N / 4)
        f[0][i][j] += 2.0;
      if (i > N / 2 - 20 && i < N / 2 + 20 && j > N / 2 - 20 && j < N / 2 + 20)
        f[0][i][j] += 4.0;

      f[1][i][j] = 0.0;
    }
  }
}

int main(int argc, char **argv)
{
  int i, j;
  double *d_f;
  double tmp_gpu = 0.0, tmp_cpu = 0.0;
  dim3 grid(N / BSIZE, N / BSIZE, 1), block(BSIZE, BSIZE, 1); // grid and block size   // 2 Dim, each block have 32*32 = 1024 threads
  cudaEvent_t start, stop;                                    // for measument time on GPU
  struct timeval start_time, end_time;                        // for measument time on CPU
  float elapsed_time_gpu, elapsed_time_cpu;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  init();

  cudaMalloc((void **)&d_f, sizeof(double) * 2 * N * N);

  // We copy only the first matrix because the second one is not used
  cudaMemcpy(d_f, f, sizeof(double) * N * N, cudaMemcpyHostToDevice);

  /* main calculation on GPU */
  cudaEventRecord(start, 0);

  /* write your gpu kernel call(s) here */

  // we call the kernel NSTEP times, this successive call is not a problem because the kernel is not asynchronous
  for (int st = 0; st < NSTEP; st++)
  {
    gpu_kernel<<<grid, block>>>(d_f);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_gpu, start, stop);

  // copy back the result to host
  cudaMemcpy(f, d_f, sizeof(double) * N * N, cudaMemcpyDeviceToHost);

  /* error checking: don't remove */
  for (i = N / 4; i < N / 2; i++)
    for (j = N / 4; j < N / 2; j++)
      tmp_gpu += f[0][i][j];

  init();

  /* main calculation on CPU*/
  gettimeofday(&start_time, NULL);

  cpu_kernel();

  gettimeofday(&end_time, NULL);
  elapsed_time_cpu = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                     (end_time.tv_usec - start_time.tv_usec) / 1000.0;

  /* error checking: don't remove */
  for (i = N / 4; i < N / 2; i++)
    for (j = N / 4; j < N / 2; j++)
      tmp_cpu += f[0][i][j];

  /* don't modify below */
  printf("check sum: cpu - %lf, gpu - %lf\n", tmp_cpu, tmp_gpu);
  printf("cpu time = %lf [msec]\n", elapsed_time_cpu);
  printf("gpu time = %lf [msec]\n", elapsed_time_gpu);
}

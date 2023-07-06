#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 512     /* problem size */
#define NSTEP 1000 /* Num. of time steps */
#define KAPPA 0.2

#define BSIZE 32

double f[2][N][N]={0}; 

__global__ void gpu_kernel()
{
   // write your code here
}

__host__ void cpu_kernel()
{
   int st, i, j;
   for(st = 0; st < NSTEP; st++) {
     for(i = 1; i < N-1; i++) {
       for(j = 1; j < N-1; j++) {
         f[1][i][j] = f[0][i][j] + KAPPA * (f[0][i][j-1] + f[0][i][j+1]
			                  + f[0][i-1][j] + f[0][i+1][j] 
			                  - 4.0*f[0][i][j]);
       }
     }

     for(i = 0; i < N; i++)
       for(j = 0; j < N; j++)
         f[0][i][j]= f[1][i][j];
   }
}

void init()
{
  int i, j;

   /* data initialization */
   for(i = 1; i < N-1; i++) {
     for(j = 1; j < N-1; j++) {
       f[0][i][j] = 1.0;
       if (i > N/4 && i < 3*N/4 && j > N/4 && j < 3*N/4)
         f[0][i][j] += 2.0;
       if (i > N/2-20 && i < N/2+20 && j > N/2-20 && j < N/2+20)
         f[0][i][j] += 4.0;

       f[1][i][j] = 0.0;
     }
   } 
}

int main(int argc, char **argv)
{ 
   int i, j;
   double *d_f;
   double tmp_gpu=0.0, tmp_cpu=0.0;
   dim3 grid(N/BSIZE, 1, 1), block(BSIZE, 1, 1); // grid and block size
   cudaEvent_t start, stop;              // for measument time on GPU
   struct timeval start_time, end_time;  // for measument time on CPU
   float elapsed_time_gpu, elapsed_time_cpu;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   init();

   cudaMalloc((void **)&d_f, sizeof(double) * 2*N*N);

   /* (sample) copy data the host to the device */
   cudaMemcpy(d_f, f, sizeof(double) * 2*N*N, cudaMemcpyHostToDevice);
   
   /* main calculation on GPU */ 
   cudaEventRecord(start, 0);

   /* write your gpu kernel call(s) here */
   // gpu_kernel<<<grid, block>>>(...);
  
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed_time_gpu, start, stop);

   /* (sample) copy data from gpu to cpu */
   cudaMemcpy(f, d_f, sizeof(double) * 2*N*N, cudaMemcpyDeviceToHost);

   /* error checking: don't remove */
   for(i = N/4; i < N/2; i++)
     for(j = N/4; j < N/2; j++)
       tmp_gpu += f[0][i][j];

   init();

   /* main calculation on CPU*/
   gettimeofday(&start_time, NULL);

   cpu_kernel();
   
   gettimeofday(&end_time, NULL);
   elapsed_time_cpu = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0;

   /* error checking: don't remove */
   for(i = N/4; i < N/2; i++)
     for(j = N/4; j < N/2; j++)
       tmp_cpu += f[0][i][j];

   /* don't modify below */
   printf("check sum: cpu - %lf, gpu - %lf\n", tmp_cpu, tmp_gpu);
   printf("cpu time = %lf [msec]\n", elapsed_time_cpu);
   printf("gpu time = %lf [msec]\n", elapsed_time_gpu);
}


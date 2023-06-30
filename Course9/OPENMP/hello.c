#include <stdio.h>
#include <omp.h>

void main()
{
#pragma omp parallel
  {
    printf("Hello OpenMP from %d of %d\n",
           omp_get_thread_num(), omp_get_num_threads());
  }
}


/*
 *   reduct.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define N 100000

int main(int argc, char **argv)
{
  int i,tnum;
  double a[N],b[N],c[N];
  double sum;
  double start, end;
  	srand(10);
  	for (i=0; i<N; i++) {
		a[i] = (double)rand();
		b[i] = (double)rand();}
	start = omp_get_wtime();
#pragma omp parallel 
  {
  tnum = omp_get_num_threads();
#pragma omp for
    for (i = 0; i < N; i++) {
	  c[i] = a[i]*b[i];
	  }
  }
    sum=0.0;
#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; i++) { 
		sum += c[i];
	}
	end = omp_get_wtime();
	printf("%e\n", sum);
	printf("Num. of threads %d Total time = %lf [sec] \n",tnum, end-start);
}

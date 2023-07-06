#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define N 1024     /* problem size */
#define NSTEP 1000 /* Num. of time steps */
#define KAPPA 0.2

double f[2][N][N] = {0};

void fout(char *fname) /* output data to file */
{
  FILE *ofile;
  int i, j;

  ofile = fopen(fname, "w");

  for (i = 1; i < N - 1; i++)
    for (j = 1; j < N - 1; j++)
      fprintf(ofile, "%d %d %f\n", i, j, f[0][i][j]);
  fprintf(ofile, "\n");

  fclose(ofile);
}

int main(int argc, char **argv)
{
  int i, j, st;
  int myid, nproc;
  double tmp = 0.0;
  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

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
    }
  }

  /* for writing matrix data to file */
  // fout("out.data");

  start = MPI_Wtime();

  /* main calculation */
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

  end = MPI_Wtime();

  /* don't modify below */
  /* error checking: don't remove */
  for (i = N / 4; i < N / 2; i++)
    for (j = N / 4; j < N / 2; j++)
      tmp += f[0][i][j];

  printf("check sum = %lf\n", tmp);
  printf("time = %lf [sec]\n", end - start);
  MPI_Finalize();
}

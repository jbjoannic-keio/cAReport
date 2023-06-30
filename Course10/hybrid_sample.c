#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

int main(int argc, char *argv[])
{
    int  myid, nprocs, tag, count, data, nth;
    int  i, t;
    MPI_Status status;

    if (argc != 2) {
      printf("Error: argc must be 2\n");
      exit(1);
    }

    nth = atoi(argv[1]);
    omp_set_num_threads(nth);
		 
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    tag = 100;
    count = 1;
    data = 1;

    if(myid != 0){
	#pragma omp parallel for reduction(*:data)
	for (t = 0; t < N; t++) {
	  data *= myid;
	}
        MPI_Send(&data, count, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }
    else {
        printf("number of thread is %d\n", nth);
        for (i = 1; i < nprocs; i++) {
          MPI_Recv(&data, count, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
          printf("Got data from Rank-%d: %d\n", i, data);
	}
    } 

    MPI_Finalize();
}

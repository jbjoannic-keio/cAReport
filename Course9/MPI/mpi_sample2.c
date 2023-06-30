#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16*1024

int main(int argc, char **argv)
{
    int myid, nproc, i, tmp;
    FILE *fin;
    double x[N];
    double sum, psum;
    double start, startcomp, end;
    MPI_Status status;

    if((fin = fopen("mat16k.dat", "r")) == NULL) {
        fprintf(stderr, "mat16.dat is not existing\n");
        exit(1);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    sum=0.0;
    if (myid == 0) {
        for (i = 0; i < N; i++)
            tmp = fscanf(fin, "%lf", &x[i]);
	
	start = MPI_Wtime();

        for (i = 1; i < nproc; i++) 
            MPI_Send(&x[i*N/nproc], N/nproc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
	
	startcomp = MPI_Wtime();

	for(i = 0; i < N/nproc; i++)
	    sum += x[i];

        for (i = 1; i < nproc; i++) {
            MPI_Recv(&psum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            sum += psum;
        }

	end = MPI_Wtime();
        printf("result: %lf\n", sum);
        printf("Total time = %lf [sec], Comp. time = %lf [sec]\n", end - start, end - startcomp);
    }
    else {
        i=0;
        MPI_Recv(&x[i], N/nproc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

	for(i = 0; i < N/nproc; i++)
            sum += x[i];

        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}


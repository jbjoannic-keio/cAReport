#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int  myid, nprocs, tag, count, data, i;
    int len = 100;
    char hname[len];
    MPI_Status status;
		 
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Get_processor_name(hname, &len);
    printf("Rank: %d (Machine: %s)\n", myid, hname);

    tag = 100;
    count = 1;
    data = myid;

    if(myid != 0){
        MPI_Send(&data, count, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }
    else {
        for (i = 1; i < nprocs; i++) {
            MPI_Recv(&data, count, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            printf("Got data from Rank = %d\n", data);
	}
    } 

    MPI_Finalize();
}


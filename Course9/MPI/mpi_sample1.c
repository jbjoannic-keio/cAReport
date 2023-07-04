#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int myid, nprocs, tag, count, i;
    MPI_Status status;
    double data[2][3][3];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    tag = 100;
    count = 3;
    data[0][0][0] = myid;
    data[0][0][1] = myid * 2;
    data[0][0][2] = myid * 3;
    data[0][1][0] = myid * 4;
    data[0][1][1] = myid * 5;
    data[0][1][2] = myid * 6;

    if (myid != 0)
    {
        double *linePtr = &data[0][0][0];
        MPI_Send(linePtr, 6, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        for (i = 1; i < nprocs; i++)
        {
            double received[6];
            MPI_Recv(received, 6, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(data[0][0], received, 6 * sizeof(double));
            for (int j = 0; j < 3; j++)
            {
                printf("%f ", data[0][0][j]);
            }
            for (int j = 0; j < 3; j++)
            {
                printf("%f ", data[0][1][j]);
            }
        }
    }

    MPI_Finalize();
}

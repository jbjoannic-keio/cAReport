#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

#define N 1024     /* problem size */
#define NSTEP 1000 /* Num. of time steps */
#define KAPPA 0.2

double f[2][N][N] = {0};

void fout(char *fname) /* output data to file */
{
    FILE *ofile;
    int i, j;
    printf("\noutname = %s\n", fname);
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
    int len = 100;
    char hname[len];

    int nth; // get number of threads from command line argument
    nth = atoi(argv[1]);
    omp_set_num_threads(nth);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Status status;

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
    /* fout("out.data"); */

    start = MPI_Wtime();
    MPI_Get_processor_name(hname, &len);

    // each rank will process this number of elements, except the last one that will process the rest also
    int element_per_rank = N / nproc;

    // I will use a strategy where the processor 0 is receiving the data and broadcasting it to the other processors
    // this way, there is no communication between other processors

    // only proc 0 will have the separations array, to avoid sending it to all the other procs
    // the array will have the limits of each processor, the first line and the last one to process
    int *separations;
    int limits[2];
    if (myid == 0)
    {
        separations = malloc(2 * nproc * sizeof(int));
        limits[0] = 1;
        limits[1] = element_per_rank;
        if (nproc == 1)
            limits[1] = N - 1;
        separations[0] = 1;
        separations[1] = element_per_rank;
        if (nproc == 1)
            separations[1] = N - 1;
        for (i = 1; i < nproc; i++)
        {
            if (i == nproc - 1)
            {
                separations[2 * i] = i * element_per_rank;
                separations[2 * i + 1] = N - 1;
            }
            else
            {
                separations[2 * i] = i * element_per_rank;
                separations[2 * i + 1] = (i + 1) * element_per_rank;
            }
        }
    }

    // Each proc will have its own upper and lower limits for the lines to process
    else
    {
        if (myid == nproc - 1)
        {
            limits[0] = myid * element_per_rank;
            limits[1] = N - 1;
        }
        else
        {
            limits[0] = myid * element_per_rank;
            limits[1] = (myid + 1) * element_per_rank;
        }
    }

    /* main calculation */
    for (st = 0; st < NSTEP; st++)
    {

        // We separate the lines for each processors given there limits
        for (i = limits[0]; i < limits[1]; i++) // MPI (limits are unique to each process), but not f
        {
            // We separate the column for each threads
#pragma omp parallel for
            for (j = 1; j < N - 1; j++)
            {
                f[1][i][j] = f[0][i][j] + KAPPA * (f[0][i][j - 1] + f[0][i][j + 1] + f[0][i - 1][j] + f[0][i + 1][j] - 4.0 * f[0][i][j]);
            }
        }

        for (i = limits[0]; i < limits[1]; i++)
#pragma omp parallel for
            for (j = 0; j < N; j++)
                f[0][i][j] = f[1][i][j];

        // We should only communicate the lines that are next to the other processors for the next step

        // We send all the internal boundary lines to the processor 0
        if (myid != 0)
        {

            double *linePtr = &(f[0][limits[0]][0]);
            MPI_Send(linePtr, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // Send rows that are next to the previous rank (up in the scheme)
            if (myid != nproc - 1)
            {
                linePtr = &(f[0][limits[1] - 1][0]);
                MPI_Send(linePtr, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD); // Send rows that are next to the next rank (down in the scheme) the last row
            }
        }
        if (myid == 0)
        {
            for (int i = 1; i < nproc; i++)
            {
                double received[N];
                MPI_Recv(received, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); // Receive rows
                memcpy(f[0][separations[2 * i]], received, N * sizeof(double));
                if (i != nproc - 1)
                {
                    MPI_Recv(received, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status); // Receive rows
                    memcpy(f[0][separations[2 * i + 1] - 1], received, N * sizeof(double));

                } // if not the last rank
            }
        }

        //

        // We send all the external boundary lines to all the processors
        if (myid == 0)
        {
            for (int i = 1; i < nproc; i++)
            {
                double *linePtr = &(f[0][separations[2 * i] - 1][0]);
                MPI_Send(linePtr, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); // Send last row from previous rank
                if (i != nproc - 1)
                {
                    linePtr = &(f[0][separations[2 * (i + 1)]][0]);
                    MPI_Send(linePtr, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD); // Send first row from next rank
                }
            }
        }
        else
        {
            double received[N];
            MPI_Recv(received, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status); // Receive row from previous rank
            memcpy(f[0][limits[0] - 1], received, N * sizeof(double));
            if (myid != nproc - 1)
            {
                MPI_Recv(received, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status); // Receive row from next rank
                memcpy(f[0][limits[1]], received, N * sizeof(double));
            }
        }
    }

    // We send all the lines to the processor 0 to have the final result
    if (myid != 0)
    {
        int elements_nb = (limits[1] - limits[0]) * N;

        if (myid == nproc - 1)
            elements_nb += N; // we need to add the last row if this is the last rank
        double *linePtr = &(f[0][limits[0]][0]);
        MPI_Send(linePtr, elements_nb, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // Send row from limit 0 to limit 1
    }
    else
    {
        for (int i = 1; i < nproc; i++)
        {

            int elements_nb = (separations[2 * i + 1] - separations[2 * i]) * N;
            if (i == nproc - 1)
                elements_nb += N; // we need to add the last row if this is the last rank
            double received[elements_nb];
            MPI_Recv(received, elements_nb, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); // Receive row from limit 0 to limit 1
            memcpy(f[0][separations[2 * i]], received, elements_nb * sizeof(double));
        }
    }

    end = MPI_Wtime();

    /* don't modify below */
    /* error checking: don't remove */
    for (i = N / 4; i < N / 2; i++)
        for (j = N / 4; j < N / 2; j++)
            tmp += f[0][i][j];

    if (myid == 0) // only processor 0 will have the correct result
    {
        printf("rank %d check sum = %lf\n", myid, tmp);
        printf("time = %lf [sec]\n", end - start);
    }
    char name[20];
    sprintf(name, "%d", myid);
    fout(name);
    MPI_Finalize();
}

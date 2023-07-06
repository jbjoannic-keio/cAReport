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

    int nth;
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
    printf("Rank: %d (Machine: %s)\n", myid, hname);
    printf("procnum = %d\n", nproc);
    int element_per_rank = N / nproc;

    // only proc 0 will have the separations array, to avoid sending it to all the other procs
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
    // print separations
    if (myid == 0)
    {
        for (i = 0; i < 2 * nproc; i++)
            printf("separations[%d] = %d\n", i, separations[i]);
    }

    // Each proc will have its own upper and lower limits
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

    printf("N %d \nrank=%d lower=%d upper=%d\n", N, myid, limits[0], limits[1]);
    /* main calculation */

    for (st = 0; st < NSTEP; st++)
    {
        printf("start STEP number %d on rank %d\n", st, myid);
        for (i = limits[0]; i < limits[1]; i++) // MPI (limits are unique to each process), but not f
        {
#pragma omp parallel for
            for (j = 1; j < N - 1; j++)
            {
                if (st == 0 && i == 1 && j == 1)
                    printf("thread from %d\n", omp_get_thread_num());
                f[1][i][j] = f[0][i][j] + KAPPA * (f[0][i][j - 1] + f[0][i][j + 1] + f[0][i - 1][j] + f[0][i + 1][j] - 4.0 * f[0][i][j]);
            }
        }
        printf("end computation on rank %d\n", myid);

        // ACTUALIZE
        for (i = limits[0]; i < limits[1]; i++)
#pragma omp parallel for
            for (j = 0; j < N; j++)
                f[0][i][j] = f[1][i][j];

        //
        // SEND ALL TO THE FIRST RANK TO ACTUALIZE THE BOUNDARIES
        if (myid != 0)
        {
            printf("R%d  sending to rank 0   || ligne %d\n", myid, limits[0]);

            double *linePtr = &(f[0][limits[0]][0]);
            MPI_Send(linePtr, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // Send rows that are next to the previous rank (up in the scheme)
            if (myid != nproc - 1)
            {
                printf("R%d  sending to rank 0   || ligne %d\n", myid, limits[1] - 1);
                linePtr = &(f[0][limits[1] - 1][0]);
                MPI_Send(linePtr, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD); // Send rows that are next to the next rank (down in the scheme) the last row
            }
        }
        if (myid == 0)
        {
            for (int i = 1; i < nproc; i++)
            {

                printf("R0 waiting receiving from rank %d     ||  lignes %d\n", i, separations[2 * i]);
                double received[N];
                MPI_Recv(received, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); // Receive rows
                memcpy(f[0][separations[2 * i]], received, N * sizeof(double));
                if (i != nproc - 1)
                {
                    printf("R0 waiting receiving from rank %d     ||  lignes %d\n", i, separations[2 * i + 1]);
                    MPI_Recv(received, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status); // Receive rows
                    memcpy(f[0][separations[2 * i + 1] - 1], received, N * sizeof(double));

                } // if not the last rank
                printf("R0 received from rank %d\n\n", i);
            }
        }

        //

        // SEND TO ALL RANKS TO ACTUALIZE THEIR EXTERNAL BOUNDARIES
        if (myid == 0)
        {
            for (int i = 1; i < nproc; i++)
            {
                printf("R0 sending to rank %d    ||   lignes %d\n", i, separations[2 * i] - 1);

                double *linePtr = &(f[0][separations[2 * i] - 1][0]);
                MPI_Send(linePtr, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD); // Send last row from previous rank
                if (i != nproc - 1)
                {
                    printf("R0 sending to rank %d    ||   lignes %d\n", i, separations[2 * (i + 1)]);

                    linePtr = &(f[0][separations[2 * (i + 1)]][0]);
                    MPI_Send(linePtr, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD); // Send first row from next rank
                }
            }
        }
        else
        {
            printf("R%d waiting receiving ||    lignes %d\n", myid, limits[0] - 1);
            double received[N];
            MPI_Recv(received, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status); // Receive row from previous rank
            memcpy(f[0][limits[0] - 1], received, N * sizeof(double));
            if (myid != nproc - 1)
            {
                printf("R%d waiting receiving ||    lignes %d\n", myid, limits[1]);
                MPI_Recv(received, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status); // Receive row from next rank
                memcpy(f[0][limits[1]], received, N * sizeof(double));
            }
            printf("receiving on rank %d\n", myid);
        }
    }

    // FINAL ACTUALIZATION. ALL IS SEND TO 0

    if (myid != 0)
    {
        int elements_nb = (limits[1] - limits[0]) * N;

        if (myid == nproc - 1)
            elements_nb += N; // we need to add the last row if this is the last rank
        printf("FINAL sending to rank 0 from rank %d     || line %d to %d   ||  element_nb %d\n", myid, limits[0], limits[1], elements_nb);
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
            printf("FINAL waiting receiving from rank %d     ||    line %d to %d    ||   elementnb %d\n", i, separations[2 * i], separations[2 * i + 1], elements_nb);
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

    printf("rank %d check sum = %lf\n", myid, tmp);
    printf("time = %lf [sec]\n", end - start);
    char name[20];
    sprintf(name, "%d", myid);
    fout(name);
    MPI_Finalize();
}

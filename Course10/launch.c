#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void launchCommand(int i, int n)
{
    char command[100];
    sprintf(command, "MPICH_PORT_RANGE=55100:55150 mpirun -np %d --host remote01,remote07 ./fdm %d", i, n);

    FILE *fp;
    char output[100];
    char timeStr[20];
    double time;

    fp = popen(command, "r");
    if (fp == NULL)
    {
        printf("Failed to run command.\n");
        exit(1);
    }

    while (fgets(output, sizeof(output), fp) != NULL)
    {
        if (sscanf(output, "time = %lf [sec]", &time) == 1)
        {
            sprintf(timeStr, "%lf", time);
            printf("Time = %lf [sec]\n", time);
            break;
        }
    }

    pclose(fp);

    // Save the time to a file for comparison
    FILE *file = fopen("time.txt", "a");
    if (file == NULL)
    {
        printf("Failed to open file.\n");
        exit(1);
    }

    fprintf(file, "rank %d, thread %d, time%s\n", i, n, timeStr);
    fclose(file);
}

int main()
{
    int i, n;

    // Loop over values of i and n
    for (i = 1; i <= 4; i++)
    {
        for (n = 1; n <= 4; n++)
        {
            launchCommand(i, n);
        }
    }

    return 0;
}
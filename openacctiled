#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>
  #include <sys/time.h>

#define TILE_SIZE 16

double* randomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;  // Random double between 0 and 100
    }
    return matrix;
}

void multiplyMatrices(int m, int n, int p, double* A, double* B, double* C) {
    #pragma acc data copyin(A[0:m*n], B[0:n*p]) copyout(C[0:m*p])
    {
        #pragma acc parallel loop tile(TILE_SIZE, TILE_SIZE)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * p + j];
                }
                C[i * p + j] = sum;
            }
        }
    }
}

int main() {
    srand(time(NULL));

    int m = 1000, n = 600, p = 800; 

 
    double cpu_time_used, total_time = 0;

    for (int iteration = 0; iteration < 10; iteration++) {
        double* A = randomMatrix(m, n);
        double* B = randomMatrix(n, p);
        double* C = (double*)malloc(m * p * sizeof(double)); 
      
        struct timeval start, end;
        gettimeofday(&start, NULL);
        multiplyMatrices(m, n, p, A, B, C);
        gettimeofday(&end, NULL);

        double cpu_time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        total_time += cpu_time_used;

        free(A);
        free(B);
        free(C);
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", total_time / 10);
    return 0;
}

%%writefile matrix.cu
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>
__global__ void matrixMultiplication(float *A, float *B, float *C, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < m && col < n)
    {
        for (int k = 0; k < p; k++)
        {
            sum += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main()
{
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    int m = 1000;
    int n = 500;
    int p = 2000;
  
    int sizeA = m * n * sizeof(float);
    int sizeB = n * p * sizeof(float);
    int sizeC = m * p * sizeof(float);
    
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *d_A, *d_B, *d_C;
    
    srand(time(NULL));
    for (int i = 0; i < m * n; i++)
        h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < n * p; i++)
        h_B[i] = (float)rand() / RAND_MAX;
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    
    dim3 dimBlock(32, 32);
    dim3 dimGridC((p + dimBlock.x - 1) / dimBlock.x, (p + dimBlock.y - 1) / dimBlock.y);
    
    matrixMultiplication<<<dimGridC, dimBlock>>>(d_A, d_B, d_C, m, n, m);
  
        
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to generate:  %3.1f ms \n", time);
    
  //  for (int i = 0; i < m; i++)
   // {
    //    for (int j = 0; j < p; j++)
     //   {
      //      printf("%f ", h_C[i * p + j]);
    //  }
   // printf("\n");
   // }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}

%%writefile matrixTiles.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define TILE_WIDTH 32

__global__ void matrix_multiply(float *A, float *B, float *C,
                                 int heightA, int widthA, int widthB)
{
    // Calculate the row and column of the element in C that this thread will compute
      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y;
      int row = by * blockDim.y + ty;
      int col = bx * blockDim.x + tx;

    // Allocate shared memory for the tiles of A and B
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    // Accumulate the dot product of the tiles of A and B for this element in C
    float sum = 0.0;
    for (int k = 0; k < (widthA - 1) / TILE_WIDTH + 1; k++)
    {
        // Load the tiles of A and B from global memory into shared memory
        int a_row = row;
        int a_col = k * TILE_WIDTH + tx;
        if (a_row < heightA && a_col < widthA)
            shared_A[ty][tx] = A[a_row * widthA + a_col];
        else
            shared_A[ty][tx] = 0.0;

        int b_row = k * TILE_WIDTH + ty;
        int b_col = col;
        if (b_row < widthA && b_col < widthB)
            shared_B[ty][tx] = B[b_row * widthB + b_col];
        else
            shared_B[ty][tx] = 0.0;

        // Synchronize threads to ensure the tiles are loaded into shared memory
        __syncthreads();

        // Multiply the tiles together and accumulate the result in the sum variable
        for (int i = 0; i < TILE_WIDTH; i++)
            sum += shared_A[ty][i] * shared_B[i][tx];

        // Synchronize threads again before loading the next tiles
        __syncthreads();
    }

    // Write the final result for this element in C to global memory
    if (row < heightA && col < widthB)
        C[row * widthB + col] = sum;
}

int main()
{   struct timeval t1, t2;
    gettimeofday(&t1, 0);
    // Set the dimensions of the matrices
    int heightA = 1000;
    int widthA = 500;
    int heightB = 500;
    int widthB = 2000;

    // Allocate memory for the matrices on the host
    float *h_A = (float*)malloc(heightA * widthA * sizeof(float));
    float *h_B = (float*)malloc(heightB * widthB * sizeof(float));
    float *h_C = (float*)malloc(heightA * widthB * sizeof(float));

    // Initialize the matrices with random numbers
    srand(time(NULL));
    for (int i = 0; i < heightA * widthA; i++)
        h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < heightB * widthB; i++)
        h_B[i] = (float)rand() / RAND_MAX;

    // Allocate memory for the matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, heightA * widthA * sizeof(float));
    cudaMalloc(&d_A, heightA * widthA * sizeof(float));
    cudaMalloc(&d_B, heightB * widthB * sizeof(float));
    cudaMalloc(&d_C, heightA * widthB * sizeof(float));

    // Copy the matrices from host to device
    cudaMemcpy(d_A, h_A, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, heightB * widthB * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel to perform matrix multiplication
    dim3 grid((heightA - 1) / TILE_WIDTH + 1, (widthB - 1) / TILE_WIDTH + 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    matrix_multiply<<<grid, block>>>(d_A, d_B, d_C, heightA, widthA, widthB);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to generate:  %3.1f ms \n", time);

    // Print out the first few elements of the result matrix
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
            printf("%f ", h_C[i * widthB + j]);
        printf("\n");
    }

    // Free the memory on the device and host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

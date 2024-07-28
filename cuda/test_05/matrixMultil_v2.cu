#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cublas_v2.h>

#define MATRIX_A_ROWS 128
#define MATRIX_A_COLS 128
#define MATRIX_B_ROWS 128
#define MATRIX_B_COLS 128

using namespace std;

// CUDA 核函数，用于计算矩阵乘法 C = A * B
__global__ void matrixMultiplyKernel(int *d_A, int *d_B, int *d_C, int A_rows, int A_cols, int B_cols)
{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < A_rows && col < B_cols)
        {
                int value = 0;
                for (int k = 0; k < A_cols; ++k)
                {
                        value += d_A[row * A_cols + k] * d_B[k * B_cols + col];
                }
                d_C[row * B_cols + col] = value;
        }
}

void matrixMultiplyCUBLAS(int *h_A, int *h_B, int *h_C, int A_rows, int A_cols, int B_cols)
{
        int *d_A, *d_B, *d_C;
        cublasHandle_t handle;
        cublasCreate(&handle);

        cudaMalloc((void **)&d_A, A_rows * A_cols * sizeof(int));
        cudaMalloc((void **)&d_B, B_cols * B_cols * sizeof(int));
        cudaMalloc((void **)&d_C, A_rows * B_cols * sizeof(int));

        cudaMemcpy(d_A, h_A, A_rows * A_cols * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, B_cols * B_cols * sizeof(int), cudaMemcpyHostToDevice);

        const int alpha = 1;
        const int beta = 0;
        auto start = chrono::high_resolution_clock::now();
        cublasGemmEx(handle, CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     B_cols,
                     A_rows,
                     A_cols,
                     &alpha,
                     d_B,
                     CUDA_R_32I, B_cols, d_A, CUDA_R_32I, A_cols, &beta, d_C, CUDA_R_32I, B_cols, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> durationCUBLAS = end - start;
        cout << "GPU (cuBLAS) Time: " << durationCUBLAS.count() << " ms" << endl;
        cudaMemcpy(h_C, d_C, A_rows * B_cols * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
}

int main()
{
        int *h_A = (int *)malloc(MATRIX_A_ROWS * MATRIX_A_COLS * sizeof(int));
        int *h_B = (int *)malloc(MATRIX_B_ROWS * MATRIX_B_COLS * sizeof(int));
        int *h_C_GPU = (int *)malloc(MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int));
        int *h_C_CUBLAS = (int *)malloc(MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int));

        srand(time(0));
        for (int i = 0; i < MATRIX_A_ROWS * MATRIX_A_COLS; ++i)
        {
                h_A[i] = rand() % 10;
        }
        for (int i = 0; i < MATRIX_B_ROWS * MATRIX_B_COLS; ++i)
        {
                h_B[i] = rand() % 10;
        }

        // 分配设备内存
        int *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, MATRIX_A_ROWS * MATRIX_A_COLS * sizeof(int));
        cudaMalloc((void **)&d_B, MATRIX_B_ROWS * MATRIX_B_COLS * sizeof(int));
        cudaMalloc((void **)&d_C, MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int));

        cudaMemcpy(d_A, h_A, MATRIX_A_ROWS * MATRIX_A_COLS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, MATRIX_B_ROWS * MATRIX_B_COLS * sizeof(int), cudaMemcpyHostToDevice);

        // 设置 CUDA 核函数的网格和块的维度
        dim3 blockSize(16, 16);
        dim3 gridSize((MATRIX_B_COLS + blockSize.x - 1) / blockSize.x,
                      (MATRIX_A_ROWS + blockSize.y - 1) / blockSize.y);

        // 自定义 CUDA 核函数 运行时间
        auto start = chrono::high_resolution_clock::now();
        matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, MATRIX_A_ROWS, MATRIX_A_COLS, MATRIX_B_COLS);
        cudaDeviceSynchronize(); // 确保 GPU 计算完成
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> durationGPU = end - start;
        cout << "GPU (Custom Kernel) Time: " << durationGPU.count() << " ms" << endl;

        cudaMemcpy(h_C_GPU, d_C, MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int), cudaMemcpyDeviceToHost);

        // cuBLAS 运行
        matrixMultiplyCUBLAS(h_A, h_B, h_C_CUBLAS, MATRIX_A_ROWS, MATRIX_A_COLS, MATRIX_B_COLS);

        //     打印结果（可选）
        //     for (int i = 0; i < MATRIX_A_ROWS; ++i)
        //     {
        //         for (int j = 0; j < MATRIX_B_COLS; ++j)
        //         {
        //             cout << h_C_GPU[i * MATRIX_B_COLS + j] << " ";
        //         }
        //         cout << endl;
        //     }

        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 释放主机内存
        free(h_A);
        free(h_B);
        free(h_C_GPU);
        free(h_C_CUBLAS);

        return 0;
}

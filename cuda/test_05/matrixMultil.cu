#include <iostream>
#include <cstdlib>
#include <ctime>

#define MATRIX_A_ROWS 3
#define MATRIX_A_COLS 3
#define MATRIX_B_ROWS 3
#define MATRIX_B_COLS 3

using namespace std;

// CUDA 核函数，用于计算矩阵乘法 C = A * B
__global__ void matrixMultiply(int *d_A, int *d_B, int *d_C, int A_rows, int A_cols, int B_cols)
{
    // 计算当前线程在矩阵 C 中的行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 计算线程的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程的列索引

    // 检查线程是否在有效的矩阵 C 范围内
    if (row < A_rows && col < B_cols)
    {
        int value = 0; // 初始化结果值为 0

        // 计算矩阵 C 中 (row, col) 元素的值
        for (int k = 0; k < A_cols; ++k) // 遍历 A 的列和 B 的行
        {
            // 计算 A 的第 row 行和 B 的第 col 列的点积
            value += d_A[row * A_cols + k] * d_B[k * B_cols + col];
        }

        // 将计算结果存储到矩阵 C 中
        d_C[row * B_cols + col] = value;
    }
}


int main()
{
        // 分配主机内存
        int *h_A = (int *)malloc(MATRIX_A_ROWS * MATRIX_A_COLS * sizeof(int));
        int *h_B = (int *)malloc(MATRIX_B_ROWS * MATRIX_B_COLS * sizeof(int));
        int *h_C = (int *)malloc(MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int));

        // 初始化矩阵 A 和 B
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

        // 将矩阵 A 和 B 从主机复制到设备
        cudaMemcpy(d_A, h_A, MATRIX_A_ROWS * MATRIX_A_COLS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, MATRIX_B_ROWS * MATRIX_B_COLS * sizeof(int), cudaMemcpyHostToDevice);

        // 设置 CUDA 核函数的网格和块的维度
        dim3 blockSize(16, 16);
        dim3 gridSize((MATRIX_B_COLS + blockSize.x - 1) / blockSize.x,
                      (MATRIX_A_ROWS + blockSize.y - 1) / blockSize.y);

        // 调用 CUDA 核函数
        matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, MATRIX_A_ROWS, MATRIX_A_COLS, MATRIX_B_COLS);

        // 将结果从设备复制到主机
        cudaMemcpy(h_C, d_C, MATRIX_A_ROWS * MATRIX_B_COLS * sizeof(int), cudaMemcpyDeviceToHost);

        // 打印结果（可选）
        for (int i = 0; i < MATRIX_A_ROWS; ++i)
        {
                for (int j = 0; j < MATRIX_B_COLS; ++j)
                {
                        cout << h_C[i * MATRIX_B_COLS + j] << " ";
                }
                cout << endl;
        }

        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 释放主机内存
        free(h_A);
        free(h_B);
        free(h_C);

        return 0;
}

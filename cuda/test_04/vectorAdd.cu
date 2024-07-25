#include <iostream>
#include <string>
#include <cstdlib> // 包含头文件以使用 rand() 函数

#define LENVECTOR 10240 * 10240
#define THREADSPERBLOCK 256

using namespace std;

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c)
{
        int tId = blockIdx.x * blockDim.x + threadIdx.x;
        d_c[tId] = d_a[tId] + d_b[tId];
}

int main(int argc, char const *argv[])
{
        int *a = nullptr, *b = nullptr, *c1 = nullptr, *c2 = nullptr;
        int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

        a = (int *)malloc(sizeof(int) * LENVECTOR);
        b = (int *)malloc(sizeof(int) * LENVECTOR);
        c1 = (int *)malloc(sizeof(int) * LENVECTOR);
        c2 = (int *)malloc(sizeof(int) * LENVECTOR);

        int lower_bound = 1;
        int upper_bound = 100;

        for (int i = 0; i < LENVECTOR; i++)
        {
                a[i] = std::rand() % (upper_bound - lower_bound + 1) + lower_bound;
                b[i] = std::rand() % (upper_bound - lower_bound + 1) + lower_bound;
        }
        cudaMalloc((void **)&d_a, sizeof(int) * LENVECTOR);
        cudaMalloc((void **)&d_b, sizeof(int) * LENVECTOR);
        cudaMalloc((void **)&d_c, sizeof(int) * LENVECTOR);
        cudaMemcpy(d_a, a, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int THREADSPERBLOCK_TMP = 32;
        while (1)
        {
                cudaEventRecord(start);
                // vectorAdd<<<LENVECTOR / THREADSPERBLOCK, THREADSPERBLOCK>>>(d_a, d_b, d_c);
                vectorAdd<<<LENVECTOR / THREADSPERBLOCK_TMP, THREADSPERBLOCK_TMP>>>(d_a, d_b, d_c);

                cudaMemcpy(c1, d_c, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                // 记录结束时间
                cudaEventRecord(stop);
                // 同步 GPU
                cudaEventSynchronize(stop);
                // 计算执行时间
                float milliseconds = 0.0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                cout << "GPU Execution Time: " << milliseconds << " ms" << "   THREADSPERBLOCK_TMP:"<<THREADSPERBLOCK_TMP<<endl;
                bool tag = true;
                for (int i = 0; i < LENVECTOR; i++)
                {
                        c2[i] = a[i] + b[i];
                        if (c2[i] != c1[i])
                        {
                                tag = false;
                        }
                }
                if (tag)
                {
                        cout << "ok" << endl;
                }
                else
                {
                        cout << "error" << endl;
                }
                THREADSPERBLOCK_TMP+=32;
                if(THREADSPERBLOCK_TMP>1024)
                {
                        break;
                }
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(a);
        free(b);
        free(c1);
        free(c2);

        return 0;
}
#include <iostream>
#include <string>
#include <cstdlib> // 包含头文件以使用 rand() 函数

#define LENVECTOR 1024 * 1024
#define THREADSPERBLOCK 1024

using namespace std;

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c)
{
        int tId = threadIdx.x;
        while (tId < LENVECTOR)
        {
                d_c[tId] = d_a[tId] + d_b[tId];
                tId += THREADSPERBLOCK;
        }
}

int main(int argc, char const *argv[])
{
        int *a = nullptr, *b = nullptr, *c1 = nullptr, *c2 = nullptr;
        int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        cout << "*****1" << endl;
        a = (int *)malloc(sizeof(int) * LENVECTOR);
        b = (int *)malloc(sizeof(int) * LENVECTOR);
        c1 = (int *)malloc(sizeof(int) * LENVECTOR);
        c2 = (int *)malloc(sizeof(int) * LENVECTOR);

        cout << "*****2" << endl;
        int lower_bound = 1;
        int upper_bound = 100;

        for (int i = 0; i < LENVECTOR; i++)
        {
                a[i] = std::rand() % (upper_bound - lower_bound + 1) + lower_bound;
                b[i] = std::rand() % (upper_bound - lower_bound + 1) + lower_bound;
        }
        cout << "*****3" << endl;

        cudaMalloc((void **)&d_a, sizeof(int) * LENVECTOR);
        cudaMalloc((void **)&d_b, sizeof(int) * LENVECTOR);
        cudaMalloc((void **)&d_c, sizeof(int) * LENVECTOR);
        cout << "*****4" << endl;

        cudaMemcpy(d_a, a, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cout << "*****5" << endl;

        vectorAdd<<<1, THREADSPERBLOCK>>>(d_a, d_b, d_c);
        cout << "*****6" << endl;

        cudaMemcpy(c1, d_c, sizeof(int) * LENVECTOR, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cout << "*****7" << endl;
        for (int i = 0; i < LENVECTOR; i++)
        {
                c2[i] = a[i] + b[i];
                cout << "a[" << i << "] : " << a[i] << "       "
                     << "b[" << i << "] : " << b[i] << "      gpu res c1[" << i << "] : " << c1[i] << "       "
                     << "cpu res c2[" << i << "] : " << c2[i] << endl;
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
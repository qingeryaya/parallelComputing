#include <iostream>
#include <string>

using namespace std;

__global__ void add(int *d_a, int *d_b, int *d_c)
{
        *d_c = *d_a + *d_b;
}

int main(int argc, char const *argv[])
{

        int a = 10, b = 20, c;
        int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

        cudaMalloc((void **)&d_a, sizeof(int));
        cudaMalloc((void **)&d_b, sizeof(int));
        cudaMalloc((void **)&d_c, sizeof(int));

        cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

        add<<<1, 1>>>(d_a, d_b, d_c);
        cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cout << a << " + " << b << " = " << c << endl;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return 0;
}
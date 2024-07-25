#include "cudaMsg.cuh"

int main(int argc, char const *argv[])
{
        std::string cudaMsgJsonData;
        cudaMsgJsonData = getGpusMsg();
        std::cout<<cudaMsgJsonData<<std::endl;
}
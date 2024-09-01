cmake_minimum_required(VERSION 3.10)
project(matrixMultil)

# 查找 CUDA 包
find_package(CUDA REQUIRED)


# 设置适用于 Ampere 架构 GPU 的 CUDA 架构标志
set(CUDA_ARCH_BIN "8.6")
set(CUDA_ARCH_PTX "")


# 设置 NVCC 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_86,code=sm_86")

# 包含目录
include_directories(${CUDA_INCLUDE_DIRS})

# 指定源文件
set(SOURCES
    matrixMultil.cu
)

set(SOURCES_v2
    matrixMultil_v2.cu
)

# 添加可执行文件
cuda_add_executable(matrixMultil ${SOURCES})

cuda_add_executable(matrixMultil_v2 ${SOURCES_v2})


# 链接必要的库
target_link_libraries(matrixMultil ${CUDA_LIBRARIES} jsoncpp )
target_link_libraries(matrixMultil_v2 ${CUDA_LIBRARIES} jsoncpp cublas)


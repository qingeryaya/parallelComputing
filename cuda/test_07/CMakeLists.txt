# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(trtConvert VERSION 1.0)
set(CMAKE_BUILD_TYPE Debug)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-error=deprecated-declarations")
add_compile_options(-Wno-unused-parameter)


# 包含头文件目录
include_directories(
    /usr/local/cuda-11.6/TensorRT-8.4.3.1/include
)

link_directories(
    /usr/local/cuda-11.6/TensorRT-8.4.3.1/lib
    /usr/local/cuda-11.6/lib64
)

# 创建可执行文件
add_executable(
    ${PROJECT_NAME}
    commonDef.cpp
    generateTrt.cpp
    main.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    -lboost_filesystem
    -lboost_system
    -lnvinfer
    -lnvonnxparser
    -lcudart
)
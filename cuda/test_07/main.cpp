#include "generateTrt.h"
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;

// // 加载 TensorRT 引擎
// ICudaEngine* loadEngine(const std::string engineFilePath)
// {
//     // 读取序列化的引擎文件
//     std::ifstream file(engineFilePath, std::ios::binary);
//     if (!file)
//     {
//         std::cerr << "Failed to open engine file: " << engineFilePath << std::endl;
//         return nullptr;
//     }

//     // 获取文件长度
//     file.seekg(0, file.end);
//     size_t fileSize = file.tellg();
//     file.seekg(0, file.beg);

//     // 分配内存并读取文件内容
//     std::vector<char> engineData(fileSize);
//     file.read(engineData.data(), fileSize);
//     file.close();

//      P_Common::MyLogger gLogger;
//     // 创建 Runtime 和引擎
//     IRuntime* runtime = createInferRuntime(gLogger); // 使用你自己的 logger 实例
//     if (!runtime)
//     {
//         std::cerr << "Failed to create TensorRT runtime." << std::endl;
//         return nullptr;
//     }

//     // 反序列化引擎
//     ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
//     runtime->destroy(); // 销毁 runtime 实例
//     if (!engine)
//     {
//         std::cerr << "Failed to deserialize CUDA engine." << std::endl;
//         return nullptr;
//     }

//     return engine;
// }

// // 查看模型的输入层和输出层
// void printModelInfo(ICudaEngine* engine)
// {
//     if (!engine)
//     {
//         std::cerr << "Invalid engine." << std::endl;
//         return;
//     }

//     std::cout << "Model has " << engine->getNbBindings() << " bindings (inputs and outputs)." << std::endl;

//     for (int i = 0; i < engine->getNbBindings(); ++i)
//     {
//         Dims dims = engine->getBindingDimensions(i);
//         const char* name = engine->getBindingName(i);
//         std::cout << "Binding " << i << ": " << name << " (";

//         // 打印维度信息
//         for (int j = 0; j < dims.nbDims; ++j)
//         {
//             std::cout << dims.d[j];
//             if (j < dims.nbDims - 1)
//                 std::cout << "x";
//         }

//         std::cout << ") - " << (engine->bindingIsInput(i) ? "Input" : "Output") << std::endl;
//     }
// }

int main(int argc, char const *argv[])
{
    TRTBuilderTools trtBuildertools;
    std::string srcModelPath = "../waigaiSeg.onnx";
    std::string dstModelPath = "../waigaiSeg.trt";

    int res = trtBuildertools.buildTrt(srcModelPath, dstModelPath, modelType::eONNX);
    std::cout << "build TRT res:" << res << std::endl;

    res = trtBuildertools.deserializeTest(dstModelPath);
    std::cout << "反序列化 TRT测试 res:" << res << std::endl;
    return 0;
}

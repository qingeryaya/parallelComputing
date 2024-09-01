#include "generateTrt.h"
#include "errCode.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace nvinfer1;

TRTBuilderTools::TRTBuilderTools(/* args */)
{
}

TRTBuilderTools::~TRTBuilderTools()
{
}

int TRTBuilderTools::buildTrt(std::string &srcModelPath, std::string &dstModelPath, modelType srcModelType)
{
    P_Common::MyLogger nvLogger;
    const char *msg = "test logger!";
    nvLogger.log(nvinfer1::ILogger::Severity::kINFO, msg);

    // 创建TensorRT的builder
    auto builder = P_Common::SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nvLogger));
    if (!builder)
    {
        return BUILDER_CREATE_ERR; // 返回builder创建错误码
    }

    // 设置显式批次标志
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 创建网络定义
    auto network = P_Common::SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return NETWORK_CREATE_ERR; // 返回网络创建错误码
    }

    // 创建Builder配置
    auto config = P_Common::SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return CONFIG_CREATE_ERR; // 返回配置创建错误码
    }

    // 创建ONNX解析器
    auto parser = P_Common::SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, nvLogger));
    if (!parser)
    {
        return PARSER_CREATE_ERR; // 返回解析器创建错误码
    }

    // 解析ONNX模型文件
    if (!parser->parseFromFile(srcModelPath.c_str(), 1))
    {
        return PARSER_PARSE_ERR; // 返回模型解析错误码
    }

    // 创建CUDA流，用于设置配置文件流
    auto profileStream = P_Common::makeCudaStream();
    if (!profileStream)
    {
        return MAKE_CUDA_STREAM_ERR; // 返回CUDA流创建错误码
    }

    // 设置配置文件流
    config->setProfileStream(*profileStream);

    builder->setMaxBatchSize(MAX_BATCH_SIZE);

    // 构建引擎
    auto engine = P_Common::SampleUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine)
    {
        return ENGINE_BUILD_ERR; // 返回引擎构建错误码
    }

    // 序列化引擎
    auto serializedModel = P_Common::SampleUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
    if (!serializedModel)
    {
        return SERIALIZE_MODEL_ERR; // 返回模型序列化错误码
    }

    // 将序列化后的模型写入文件
    std::ofstream outFile(dstModelPath, std::ios::binary);
    if (!outFile)
    {
        std::cerr << "无法打开输出文件: " << dstModelPath << std::endl;
        return FILE_OPEN_ERR; // 返回文件打开错误码
    }
    outFile.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());
    outFile.close();

    return PASS; // 返回成功标志
}

nvinfer1::ICudaEngine *TRTBuilderTools::loadEngine(const std::string &engineFilePath)
{
    // 读取序列化的引擎文件
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open engine file: " << engineFilePath << std::endl;
        return nullptr;
    }

    // 获取文件长度
    file.seekg(0, file.end);
    size_t fileSize = file.tellg();
    file.seekg(0, file.beg);

    // 分配内存并读取文件内容
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    P_Common::MyLogger gLogger;
    // 创建 Runtime 和引擎
    IRuntime *runtime = createInferRuntime(gLogger); // 使用你自己的 logger 实例
    if (!runtime)
    {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return nullptr;
    }

    // 反序列化引擎
    ICudaEngine *engine = runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
    runtime->destroy(); // 销毁 runtime 实例
    if (!engine)
    {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        return nullptr;
    }

    return engine;
}

// 查看模型的输入层和输出层
void TRTBuilderTools::printModelInfo(nvinfer1::ICudaEngine *engine)
{
    if (!engine)
    {
        std::cerr << "Invalid engine." << std::endl;
        return;
    }

    std::cout << "Model has " << engine->getNbBindings() << " bindings (inputs and outputs)." << std::endl;

    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims = engine->getBindingDimensions(i);
        const char *name = engine->getBindingName(i);
        std::cout << "Binding " << i << ": " << name << " (";

        // 打印维度信息
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1)
                std::cout << "x";
        }

        std::cout << ") - " << (engine->bindingIsInput(i) ? "Input" : "Output") << std::endl;
    }
}

int TRTBuilderTools::deserializeTest(std::string &trtModelPath)
{
    if (!boost::filesystem::exists(trtModelPath))
        return FILE_NOT_EXIST_ERR;
    auto engine = loadEngine(trtModelPath);
    if (engine == nullptr)
        return LOAD_ENGIN_ERR;
    printModelInfo(engine);
    engine->destroy();
    return PASS;
}

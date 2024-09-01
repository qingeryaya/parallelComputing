#ifndef __GENERATE_TRT_H__
#define __GENERATE_TRT_H__
#include <iostream>
#include "commonDef.h"
#include "NvInferRuntime.h"



enum modelType
{
    eONNX
};

class TRTBuilderTools
{
private:
    /* data */
public:
    TRTBuilderTools(/* args */);
    ~TRTBuilderTools();
    int buildTrt(std::string &srcModelPath, std::string &dstModelPath, modelType srcModelType);
    nvinfer1::ICudaEngine* loadEngine(const std::string &engineFilePath);
    void printModelInfo(nvinfer1::ICudaEngine* engine);

    int deserializeTest(std::string &trtModelPath);


};

#endif
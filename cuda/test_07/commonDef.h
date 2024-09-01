#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__
#include "NvInferRuntimeCommon.h"
#include <iostream>
#include <memory>

namespace P_Common
{

    class MyLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) noexcept override;
    };

    struct InferDeleter
    {
        template <typename T>
        void operator()(T *obj) const
        {
            delete obj;
        }
    };

    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

    static auto StreamDeleter = [](cudaStream_t *pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

    inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
    {
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
        if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
        {
            pStream.reset(nullptr);
        }

        return pStream;
    }
}

#endif
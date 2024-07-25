#include "cudaMsg.cuh"

std::string getGpusMsg()
{
        std::string jsonMsg = "{}";
        Json::Value root;
        Json::CharReaderBuilder reader;
        std::istringstream jsonStream("");
        Json::parseFromStream(reader, jsonStream, &root, nullptr);

        // 统计一共有多少块GPU
        int gpuNums;
        cudaGetDeviceCount(&gpuNums);
        std::cout << "共发现了" << gpuNums << "块GPU" << std::endl;

        cudaDeviceProp prop;
        for (int i = 0; i < gpuNums; i++)
        {

                cudaGetDeviceProperties(&prop, i);
                Json::Value Node;
                Json::Value jsonArray;
                Node["name"] = prop.name;
                Node["totalGlobalMem(全局内存大小)"] = static_cast<long long>(prop.totalGlobalMem);
                Node["canMapHostMemory"] = prop.canMapHostMemory;
                Node["deviceOverlap"] = prop.deviceOverlap;
                Node["warpSize"] = prop.warpSize;
                Node["multiProcessorCount(SM数量)"] = prop.multiProcessorCount;
                Node["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
                Node["maxThreadsPerMultiProcessor"] = prop.maxThreadsPerMultiProcessor;

                jsonArray.clear();
                jsonArray.append(prop.maxThreadsDim[0]);
                jsonArray.append(prop.maxThreadsDim[1]);
                jsonArray.append(prop.maxThreadsDim[2]);
                Node["maxThreadsDimesions"] = jsonArray;
                jsonArray.clear();
                jsonArray.append(prop.maxGridSize[0]);
                jsonArray.append(prop.maxGridSize[1]);
                jsonArray.append(prop.maxGridSize[2]);
                Node["maxGridSize"] = jsonArray;
                root["No." + std::to_string(i)] = Node;
        }
        Json::StyledWriter styledWriter;
        jsonMsg = styledWriter.write(root);
        return jsonMsg;
}

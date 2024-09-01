#include "commonDef.h"

void P_Common::MyLogger::log(Severity severity, const char *msg) noexcept
{
    // 根据严重级别处理日志消息
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL ERROR: " << msg << std::endl;
        break;
    case Severity::kERROR:
        std::cerr << "ERROR: " << msg << std::endl;
        break;
    case Severity::kWARNING:
        std::cout << "WARNING: " << msg << std::endl;
        break;
    case Severity::kINFO:
        std::cout << "INFO: " << msg << std::endl;
        break;
    case Severity::kVERBOSE:
        std::cout << "VERBOSE: " << msg << std::endl;
        break;
    }
}

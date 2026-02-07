#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace orcore
{

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expr
            << " | code=" << static_cast<int>(status)
            << " (" << cudaGetErrorString(status) << ")"
            << " | at " << file << ":" << line;
    throw std::runtime_error(message.str());
}

inline void cuda_check_last_error(const char* file, int line)
{
    cuda_check(cudaGetLastError(), "cudaGetLastError()", file, line);
}

}  // namespace orcore

#define OR_CUDA_CHECK(expr) ::orcore::cuda_check((expr), #expr, __FILE__, __LINE__)
#define OR_CUDA_CHECK_LAST() ::orcore::cuda_check_last_error(__FILE__, __LINE__)

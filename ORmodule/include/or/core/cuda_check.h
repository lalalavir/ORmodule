#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace orcore
{

// Throw a detailed std::runtime_error when a CUDA runtime call fails.
//
// `expr` is the original expression string (for example, "cudaMalloc(...)")
// and file/line identify the call site.
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

// Check and report the latest CUDA launch/runtime error.
//
// This helper is typically used right after a kernel launch.
inline void cuda_check_last_error(const char* file, int line)
{
    cuda_check(cudaGetLastError(), "cudaGetLastError()", file, line);
}

}  // namespace orcore

// Wrap a CUDA runtime expression and attach source location automatically.
#define OR_CUDA_CHECK(expr) ::orcore::cuda_check((expr), #expr, __FILE__, __LINE__)

// Check the latest CUDA error with source location context.
#define OR_CUDA_CHECK_LAST() ::orcore::cuda_check_last_error(__FILE__, __LINE__)

#pragma once

#include "or/core/device_buffer.h"

#include <cuda_runtime.h>

namespace orcore
{

struct ReductionStatsI32
{
    int min_value = 0;
    int max_value = 0;
    long long sum_value = 0;
    double mean_value = 0.0;
};

ReductionStatsI32 reduce_stats_i32(const DeviceBuffer<int>& input, cudaStream_t stream = 0);

}  // namespace orcore

#pragma once

#include "or/core/device_buffer.h"

#include <cuda_runtime.h>

namespace orcore {

void reduce_minmax_i32(const DeviceBuffer<int>& input,
                      int* host_min,
                      int* host_max,
                      cudaStream_t stream = 0);

double reduce_mean_i32(const DeviceBuffer<int>& input, cudaStream_t stream = 0);

}  // namespace orcore


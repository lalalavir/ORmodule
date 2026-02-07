#pragma once

#include "or/core/device_buffer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace orcore
{

void init_rng_states(DeviceBuffer<curandState>& states, unsigned long long seed, cudaStream_t stream = 0);

void fill_uniform_u32(DeviceBuffer<curandState>& states,
                      DeviceBuffer<unsigned int>& output,
                      cudaStream_t stream = 0);

}  // namespace orcore

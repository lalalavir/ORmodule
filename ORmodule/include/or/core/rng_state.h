#pragma once

#include "or/core/device_buffer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace orcore
{

// Initialize one curandState per element in `states`.
//
// The sequence index is derived from thread/global index, while `seed`
// controls reproducibility across runs.
void init_rng_states(DeviceBuffer<curandState>& states, unsigned long long seed, cudaStream_t stream = 0);

// Fill `output` with uniformly distributed 32-bit random values.
//
// `output` is resized when needed to match `states.size()`.
void fill_uniform_u32(DeviceBuffer<curandState>& states,
                      DeviceBuffer<unsigned int>& output,
                      cudaStream_t stream = 0);

}  // namespace orcore

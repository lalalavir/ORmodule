#include "or/core/rng_state.h"
#include "or/core/cuda_check.h"

namespace
{

// Initialize curand states in parallel.
//
// Each thread owns one state and uses its global index as sequence number,
// ensuring independent random streams for subsequent draws.
__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }

    curand_init(seed, static_cast<unsigned long long>(index), 0ULL, &states[index]);
}

// Generate one random uint32 value per state.
//
// The state is updated in place by curand().
__global__ void kernel_fill_uniform_u32(curandState* states, unsigned int* output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }

    output[index] = curand(&states[index]);
}

}  // namespace

namespace orcore
{

// Host launcher for RNG state initialization.
// Safe no-op when `states` is empty.
void init_rng_states(DeviceBuffer<curandState>& states, unsigned long long seed, cudaStream_t stream)
{
    if (states.size() == 0)
    {
        return;
    }

    int n = static_cast<int>(states.size());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_init_rng<<<blocks, threads, 0, stream>>>(states.data(), seed, n);
    OR_CUDA_CHECK_LAST();
}

// Host launcher for uniform uint32 generation.
// Resizes `output` to match `states` when needed.
void fill_uniform_u32(DeviceBuffer<curandState>& states,
                      DeviceBuffer<unsigned int>& output,
                      cudaStream_t stream)
{
    if (states.size() == 0)
    {
        return;
    }

    if (output.size() != states.size())
    {
        output.allocate(states.size());
    }

    int n = static_cast<int>(states.size());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_fill_uniform_u32<<<blocks, threads, 0, stream>>>(states.data(), output.data(), n);
    OR_CUDA_CHECK_LAST();
}

}  // namespace orcore

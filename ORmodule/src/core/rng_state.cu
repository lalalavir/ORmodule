#include "or/core/rng_state.h"
#include "or/core/cuda_check.h"

namespace {

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    curand_init(seed, static_cast<unsigned long long>(index), 0ULL, &states[index]);
}

__global__ void kernel_fill_uniform_u32(curandState* states, unsigned int* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    output[index] = curand(&states[index]);
}

}  // namespace

namespace orcore {

void init_rng_states(DeviceBuffer<curandState>& states, unsigned long long seed, cudaStream_t stream) {
    if (states.size() == 0) {
        return;
    }

    int n = static_cast<int>(states.size());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_init_rng<<<blocks, threads, 0, stream>>>(states.data(), seed, n);
    OR_CUDA_CHECK_LAST();
}

void fill_uniform_u32(DeviceBuffer<curandState>& states,
                      DeviceBuffer<unsigned int>& output,
                      cudaStream_t stream) {
    if (states.size() == 0) {
        return;
    }

    if (output.size() != states.size()) {
        output.allocate(states.size());
    }

    int n = static_cast<int>(states.size());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_fill_uniform_u32<<<blocks, threads, 0, stream>>>(states.data(), output.data(), n);
    OR_CUDA_CHECK_LAST();
}

}  // namespace orcore


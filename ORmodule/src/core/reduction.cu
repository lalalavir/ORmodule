#include "or/core/reduction.h"
#include "or/core/cuda_check.h"

#include <climits>
#include <limits>
#include <vector>

namespace {

__global__ void kernel_block_reduce_minmax_i32(const int* input,
                                               int n,
                                               int* block_min,
                                               int* block_max,
                                               long long* block_sum) {
    extern __shared__ int shared_i32[];
    int* smin = shared_i32;
    int* smax = shared_i32 + blockDim.x;
    long long* ssum = reinterpret_cast<long long*>(shared_i32 + 2 * blockDim.x);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_value = 0;

    if (index < n) {
        local_value = input[index];
        smin[threadIdx.x] = local_value;
        smax[threadIdx.x] = local_value;
        ssum[threadIdx.x] = static_cast<long long>(local_value);
    } else {
        smin[threadIdx.x] = INT_MAX;
        smax[threadIdx.x] = INT_MIN;
        ssum[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + stride]);
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_min[blockIdx.x] = smin[0];
        block_max[blockIdx.x] = smax[0];
        block_sum[blockIdx.x] = ssum[0];
    }
}

}  // namespace

namespace orcore {

void reduce_minmax_i32(const DeviceBuffer<int>& input,
                      int* host_min,
                      int* host_max,
                      cudaStream_t stream) {
    if (input.size() == 0) {
        *host_min = 0;
        *host_max = 0;
        return;
    }

    const int threads = 256;
    const int n = static_cast<int>(input.size());
    const int blocks = (n + threads - 1) / threads;

    DeviceBuffer<int> block_min(blocks);
    DeviceBuffer<int> block_max(blocks);
    DeviceBuffer<long long> block_sum(blocks);

    size_t shared_bytes = static_cast<size_t>(2 * threads * sizeof(int) + threads * sizeof(long long));
    kernel_block_reduce_minmax_i32<<<blocks, threads, shared_bytes, stream>>>(
        input.data(), n, block_min.data(), block_max.data(), block_sum.data());
    OR_CUDA_CHECK_LAST();

    std::vector<int> host_block_min = block_min.copy_to_vector(stream);
    std::vector<int> host_block_max = block_max.copy_to_vector(stream);

    int min_value = std::numeric_limits<int>::max();
    int max_value = std::numeric_limits<int>::min();

    for (int value : host_block_min) {
        if (value < min_value) {
            min_value = value;
        }
    }

    for (int value : host_block_max) {
        if (value > max_value) {
            max_value = value;
        }
    }

    *host_min = min_value;
    *host_max = max_value;
}

double reduce_mean_i32(const DeviceBuffer<int>& input, cudaStream_t stream) {
    if (input.size() == 0) {
        return 0.0;
    }

    const int threads = 256;
    const int n = static_cast<int>(input.size());
    const int blocks = (n + threads - 1) / threads;

    DeviceBuffer<int> block_min(blocks);
    DeviceBuffer<int> block_max(blocks);
    DeviceBuffer<long long> block_sum(blocks);

    size_t shared_bytes = static_cast<size_t>(2 * threads * sizeof(int) + threads * sizeof(long long));
    kernel_block_reduce_minmax_i32<<<blocks, threads, shared_bytes, stream>>>(
        input.data(), n, block_min.data(), block_max.data(), block_sum.data());
    OR_CUDA_CHECK_LAST();

    std::vector<long long> host_block_sum = block_sum.copy_to_vector(stream);
    long long total_sum = 0;
    for (long long value : host_block_sum) {
        total_sum += value;
    }

    return static_cast<double>(total_sum) / static_cast<double>(input.size());
}

}  // namespace orcore

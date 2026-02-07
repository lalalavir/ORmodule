#include "or/core/reduction.h"
#include "or/core/cuda_check.h"

#include <climits>
#include <utility>

namespace
{

struct DeviceReductionStatsI32
{
    int min_value;
    int max_value;
    long long sum_value;
};

// First-pass reduction: read int32 input and output one partial stats triple per block.
__global__ void kernel_block_reduce_i32_to_stats(const int* input,
                                                  int n,
                                                  DeviceReductionStatsI32* block_stats)
{
    extern __shared__ int shared_i32[];
    int* smin = shared_i32;
    int* smax = shared_i32 + blockDim.x;
    long long* ssum = reinterpret_cast<long long*>(shared_i32 + 2 * blockDim.x);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        int local_value = input[index];
        smin[threadIdx.x] = local_value;
        smax[threadIdx.x] = local_value;
        ssum[threadIdx.x] = static_cast<long long>(local_value);
    }
    else
    {
        smin[threadIdx.x] = INT_MAX;
        smax[threadIdx.x] = INT_MIN;
        ssum[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + stride]);
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        block_stats[blockIdx.x].min_value = smin[0];
        block_stats[blockIdx.x].max_value = smax[0];
        block_stats[blockIdx.x].sum_value = ssum[0];
    }
}

// Next-pass reduction: read partial stats triples and reduce again.
__global__ void kernel_block_reduce_stats_to_stats(const DeviceReductionStatsI32* input,
                                                    int n,
                                                    DeviceReductionStatsI32* block_stats)
{
    extern __shared__ int shared_i32[];
    int* smin = shared_i32;
    int* smax = shared_i32 + blockDim.x;
    long long* ssum = reinterpret_cast<long long*>(shared_i32 + 2 * blockDim.x);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        smin[threadIdx.x] = input[index].min_value;
        smax[threadIdx.x] = input[index].max_value;
        ssum[threadIdx.x] = input[index].sum_value;
    }
    else
    {
        smin[threadIdx.x] = INT_MAX;
        smax[threadIdx.x] = INT_MIN;
        ssum[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + stride]);
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        block_stats[blockIdx.x].min_value = smin[0];
        block_stats[blockIdx.x].max_value = smax[0];
        block_stats[blockIdx.x].sum_value = ssum[0];
    }
}

}  // namespace

namespace orcore
{

ReductionStatsI32 reduce_stats_i32(const DeviceBuffer<int>& input, cudaStream_t stream)
{
    ReductionStatsI32 stats;

    if (input.size() == 0)
    {
        return stats;
    }

    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(2 * threads * sizeof(int) + threads * sizeof(long long));

    int current_count = static_cast<int>(input.size());
    int blocks = (current_count + threads - 1) / threads;

    DeviceBuffer<DeviceReductionStatsI32> partial_a(blocks);
    DeviceBuffer<DeviceReductionStatsI32> partial_b;

    kernel_block_reduce_i32_to_stats<<<blocks, threads, shared_bytes, stream>>>(
        input.data(), current_count, partial_a.data());
    OR_CUDA_CHECK_LAST();

    current_count = blocks;

    //usually 1 time
    while (current_count > 1)
    {
        blocks = (current_count + threads - 1) / threads;
        partial_b.allocate(blocks);

        kernel_block_reduce_stats_to_stats<<<blocks, threads, shared_bytes, stream>>>(
            partial_a.data(), current_count, partial_b.data());
        OR_CUDA_CHECK_LAST();

        partial_a = std::move(partial_b);
        current_count = blocks;
    }

    DeviceReductionStatsI32 device_stats;

    //copy to the host and synchronize
    OR_CUDA_CHECK(cudaMemcpyAsync(&device_stats,
                                  partial_a.data(),
                                  sizeof(DeviceReductionStatsI32),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    OR_CUDA_CHECK(cudaStreamSynchronize(stream));

    stats.min_value = device_stats.min_value;
    stats.max_value = device_stats.max_value;
    stats.sum_value = device_stats.sum_value;
    stats.mean_value = static_cast<double>(stats.sum_value) / static_cast<double>(input.size());
    return stats;
}

}  // namespace orcore

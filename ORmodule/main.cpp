#include "or/core/cuda_check.h"
#include "or/core/device_buffer.h"
#include "or/core/reduction.h"
#include "or/core/rng_state.h"

#include <cuda_runtime.h>

#include <ctime>
#include <iostream>

int main() 
{
    try
    {
        int device_count = 0;
        OR_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) 
        {
            std::cerr << "[ORmodule] No CUDA device found." << std::endl;
            return 1;
        }

        OR_CUDA_CHECK(cudaSetDevice(0));

        constexpr size_t sample_count = 1 << 16;
        orcore::DeviceBuffer<curandState> rng_states(sample_count);
        orcore::DeviceBuffer<unsigned int> random_values(sample_count);
        orcore::DeviceBuffer<int> sample_values(sample_count);

        const unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
        orcore::init_rng_states(rng_states, seed);
        orcore::fill_uniform_u32(rng_states, random_values);

        OR_CUDA_CHECK(cudaMemcpy(sample_values.data(),
                                 random_values.data(),
                                 sample_count * sizeof(unsigned int),
                                 cudaMemcpyDeviceToDevice));

        orcore::ReductionStatsI32 stats = orcore::reduce_stats_i32(sample_values);

        OR_CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "[ORmodule/Core Smoke Test]" << std::endl;
        std::cout << "  sample_count: " << sample_count << std::endl;
        std::cout << "  min: " << stats.min_value << std::endl;
        std::cout << "  max: " << stats.max_value << std::endl;
        std::cout << "  mean: " << stats.mean_value << std::endl;
        std::cout << "  status: OK" << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "[ORmodule] Fatal error: " << error.what() << std::endl;
        return 1;
    }
}

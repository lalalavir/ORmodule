#include "or/core/cuda_check.h"
#include "or/core/device_buffer.h"
#include "or/core/reduction.h"
#include "or/core/rng_state.h"
#include "or/engine/solver_engine.h"
#include "or/problem/mock_problem.h"
#include "or/problem/problem_ops.h"

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

        orengine::RunConfig config;
        config.max_generations = 6;
        config.population_size = 16;
        config.enable_crossover = true;
        config.enable_repair = true;

        orproblem::MockProblem::Instance mock_instance;
        mock_instance.initial_objective = 128;
        mock_instance.feasible_count = config.population_size;

        orengine::SolverEngine<orproblem::MockProblem> engine(config);
        orengine::RunStats engine_stats = engine.run(mock_instance);

        std::cout << "[ORmodule/Engine Mock Test]" << std::endl;
        std::cout << "  generations: " << engine_stats.generations_executed << std::endl;
        std::cout << "  best: " << engine_stats.best_objective << std::endl;
        std::cout << "  mean: " << engine_stats.mean_objective << std::endl;
        std::cout << "  feasible_count: " << engine_stats.feasible_count << std::endl;

        orproblem::MockProblem::Population mock_population;
        orproblem::MockProblem::Scratch mock_scratch;
        orproblem::ProblemOps<orproblem::MockProblem>::random_init(mock_instance, mock_population, nullptr);
        orproblem::ProblemOps<orproblem::MockProblem>::propose_move(mock_instance, mock_population, mock_scratch, nullptr);
        orproblem::ProblemOps<orproblem::MockProblem>::evaluate(mock_instance, mock_population, mock_scratch);

        std::cout << "[ORmodule/ProblemOps Mock Test]" << std::endl;
        std::cout << "  best_after_one_step: " << mock_population.best_objective << std::endl;
        std::cout << "  mean_after_one_step: " << mock_population.mean_objective << std::endl;

        std::cout << "  status: OK" << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "[ORmodule] Fatal error: " << error.what() << std::endl;
        return 1;
    }
}

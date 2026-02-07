#pragma once

#include "or/core/device_buffer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstddef>

namespace orproblem
{

struct MockProblem
{
    struct Instance
    {
        int initial_objective = 1000;
        size_t feasible_count = 1;
    };

    struct Population
    {
        int best_objective = 0;
        double mean_objective = 0.0;
        size_t feasible_count = 0;
    };

    struct Scratch
    {
        int generation_index = 0;
    };

    static void random_init(const Instance& instance,
                            Population& population,
                            curandState* /*rng_states*/,
                            cudaStream_t /*stream*/ = 0)
    {
        population.best_objective = instance.initial_objective;
        population.mean_objective = static_cast<double>(instance.initial_objective) + 10.0;
        population.feasible_count = instance.feasible_count;
    }

    static void evaluate(const Instance& /*instance*/,
                         Population& population,
                         Scratch& scratch,
                         cudaStream_t /*stream*/ = 0)
    {
        int improvement = std::max(1, scratch.generation_index / 2 + 1);
        population.best_objective = std::max(0, population.best_objective - improvement);
        population.mean_objective = static_cast<double>(population.best_objective) + 6.0;
    }

    static void propose_move(const Instance& /*instance*/,
                             Population& /*population*/,
                             Scratch& scratch,
                             curandState* /*rng_states*/,
                             cudaStream_t /*stream*/ = 0)
    {
        ++scratch.generation_index;
    }

    static void apply_move(Population& /*population*/,
                           Scratch& /*scratch*/,
                           cudaStream_t /*stream*/ = 0)
    {
    }

    static void repair(const Instance& /*instance*/,
                       Population& /*population*/,
                       Scratch& /*scratch*/,
                       cudaStream_t /*stream*/ = 0)
    {
    }

    static void crossover(const Instance& /*instance*/,
                          const Population& parents,
                          Population& offspring,
                          curandState* /*rng_states*/,
                          cudaStream_t /*stream*/ = 0)
    {
        offspring = parents;
    }

    static void distance(const Population& /*population*/,
                         orcore::DeviceBuffer<int>& out_dist,
                         cudaStream_t stream = 0)
    {
        out_dist.allocate(1);
        out_dist.memset_zero(stream);
    }

    static int best_objective(const Population& population)
    {
        return population.best_objective;
    }

    static double mean_objective(const Population& population)
    {
        return population.mean_objective;
    }

    static size_t feasible_count(const Population& population)
    {
        return population.feasible_count;
    }
};

}  // namespace orproblem

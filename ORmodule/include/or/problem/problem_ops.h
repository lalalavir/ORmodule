#pragma once

#include "or/core/device_buffer.h"
#include "or/problem/problem_traits.h"

#include <curand_kernel.h>

namespace orproblem
{

// ProblemOps is the stable adapter surface between SolverEngine and
// concrete problem plugins.
// - initialization: random_init
// - search step: propose_move / apply_move / evaluate
// - optional operators: repair / crossover / distance
// - statistics accessors: best_objective / mean_objective / feasible_count
template <class Problem>
struct ProblemOps
{
    using Traits = ProblemTraits<Problem>;
    using Instance = typename Traits::Instance;
    using Population = typename Traits::Population;
    using Scratch = typename Traits::Scratch;

    static void random_init(const Instance& instance,
                            Population& population,
                            curandState* rng_states,
                            cudaStream_t stream = 0)
    {
        Problem::random_init(instance, population, rng_states, stream);
    }

    static void evaluate(const Instance& instance,
                         Population& population,
                         Scratch& scratch,
                         cudaStream_t stream = 0)
    {
        Problem::evaluate(instance, population, scratch, stream);
    }

    static void propose_move(const Instance& instance,
                             Population& population,
                             Scratch& scratch,
                             curandState* rng_states,
                             cudaStream_t stream = 0)
    {
        Problem::propose_move(instance, population, scratch, rng_states, stream);
    }

    static void apply_move(Population& population,
                           Scratch& scratch,
                           cudaStream_t stream = 0)
    {
        Problem::apply_move(population, scratch, stream);
    }

    static void repair(const Instance& instance,
                       Population& population,
                       Scratch& scratch,
                       cudaStream_t stream = 0)
    {
        Problem::repair(instance, population, scratch, stream);
    }

    static void crossover(const Instance& instance,
                          const Population& parents,
                          Population& offspring,
                          curandState* rng_states,
                          cudaStream_t stream = 0)
    {
        Problem::crossover(instance, parents, offspring, rng_states, stream);
    }

    static void distance(const Population& population,
                         orcore::DeviceBuffer<int>& out_dist,
                         cudaStream_t stream = 0)
    {
        Problem::distance(population, out_dist, stream);
    }

    static int best_objective(const Population& population)
    {
        return Problem::best_objective(population);
    }

    static double mean_objective(const Population& population)
    {
        return Problem::mean_objective(population);
    }

    static size_t feasible_count(const Population& population)
    {
        return Problem::feasible_count(population);
    }
};

}  // namespace orproblem

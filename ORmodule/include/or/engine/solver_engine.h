#pragma once

#include "or/engine/run_config.h"
#include "or/engine/run_stats.h"
#include "or/problem/problem_ops.h"

#include <chrono>

namespace orengine
{

template <class Problem>
class SolverEngine
{
public:
    using Ops = orproblem::ProblemOps<Problem>;
    using Instance = typename Ops::Instance;
    using Population = typename Ops::Population;
    using Scratch = typename Ops::Scratch;

public:
    explicit SolverEngine(RunConfig config)
        : config_(config)
    {
    }

    
    // Current phases:
    // 1) random_init + first evaluate
    // 2) repeated local iterations: propose_move -> apply_move -> evaluate
    // 3) optional repair + evaluate
    // 4) stop by generation limit or time limit
    // 5) collect objective/feasibility statistics from the population
    RunStats run(const Instance& instance)
    {
        const auto start_time = std::chrono::steady_clock::now();

        Population population;
        Scratch scratch;

        Ops::random_init(instance, population, nullptr);
        Ops::evaluate(instance, population, scratch);

        int generations_target = config_.max_generations;
        if (generations_target < 0)
        {
            generations_target = 0;
        }

        int local_iters = config_.local_search_iters;
        if (local_iters <= 0)
        {
            local_iters = 1;
        }

        int generations_executed = 0;
        bool reached_time_limit = false;

        for (int generation = 0; generation < generations_target; ++generation)
        {
            for (int local_iter = 0; local_iter < local_iters; ++local_iter)
            {
                Ops::propose_move(instance, population, scratch, nullptr);
                Ops::apply_move(population, scratch);
                Ops::evaluate(instance, population, scratch);
            }

            if (config_.enable_repair)
            {
                Ops::repair(instance, population, scratch);
                Ops::evaluate(instance, population, scratch);
            }

            ++generations_executed;

            if (config_.time_limit_seconds > 0.0)
            {
                const auto current_time = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(current_time - start_time).count();
                if (elapsed >= config_.time_limit_seconds)
                {
                    reached_time_limit = true;
                    break;
                }
            }
        }

        const auto end_time = std::chrono::steady_clock::now();

        RunStats stats;
        stats.generations_executed = generations_executed;
        stats.reached_time_limit = reached_time_limit;
        stats.reached_generation_limit = (generations_target > 0 && generations_executed == generations_target && !reached_time_limit);
        stats.best_objective = Ops::best_objective(population);
        stats.mean_objective = Ops::mean_objective(population);
        stats.feasible_count = Ops::feasible_count(population);
        stats.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return stats;
    }

    const RunConfig& config() const
    {
        return config_;
    }

private:
    RunConfig config_;
};

}  // namespace orengine

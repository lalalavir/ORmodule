#pragma once

#include <cstddef>

namespace orengine
{

struct RunConfig
{
    size_t population_size = 0;
    int max_generations = 0;
    int local_search_iters = 0;
    double time_limit_seconds = 0.0;
    bool enable_crossover = false;
    bool enable_repair = false;
    bool enable_restart = false;
    unsigned long long random_seed = 0ULL;
};

}  // namespace orengine


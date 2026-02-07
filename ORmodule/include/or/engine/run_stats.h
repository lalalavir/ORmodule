#pragma once

#include <cstddef>

namespace orengine
{

struct RunStats
{
    int generations_executed = 0;
    bool reached_time_limit = false;
    bool reached_generation_limit = false;
    int best_objective = 0;
    double mean_objective = 0.0;
    double elapsed_seconds = 0.0;
    size_t feasible_count = 0;
};

}  // namespace orengine


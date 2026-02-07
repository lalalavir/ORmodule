#pragma once

namespace orproblem
{

template <class Problem>
struct ProblemTraits
{
    using Instance = typename Problem::Instance;
    using Population = typename Problem::Population;
    using Scratch = typename Problem::Scratch;
};

}  // namespace orproblem


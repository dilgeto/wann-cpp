#pragma once
#include <vector>

namespace wann {

// Non-dominated sorting (NSGA-II style).
// objVals[i] = {obj0, obj1, ...}  — MAXIMISATION assumed (higher is better).
// Returns rank[i] = position of individual i in the sorted population
// (0 = best). Individuals in the same Pareto front are ranked by
// crowding distance (larger distance = better = lower rank number).

std::vector<int> nsga_sort(const std::vector<std::vector<double>>& objVals);

// Internal helpers (exposed for testing).
std::vector<std::vector<int>> getFronts(const std::vector<std::vector<double>>& objVals);
std::vector<double>           getCrowdingDist(const std::vector<double>& objVector);

} // namespace wann

#include "../include/wann/NsgaSort.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace wann {

// -------------------------------------------------------------------------
// getFronts – fast non-dominated sort on 2 objectives.
// Exactly mirrors Python nsga_sort.py::getFronts.
// -------------------------------------------------------------------------
std::vector<std::vector<int>>
getFronts(const std::vector<std::vector<double>>& objVals)
{
    const int n = static_cast<int>(objVals.size());
    // For now we use the first two objectives (WANN always passes 2).
    auto v1 = [&](int i){ return objVals[i][0]; };
    auto v2 = [&](int i){ return objVals[i][1]; };

    // S[p] = set of individuals dominated by p.
    std::vector<std::vector<int>> S(n);
    // n[p] = number of individuals that dominate p.
    std::vector<int> domCount(n, 0);
    std::vector<int> rank(n, 0);

    std::vector<std::vector<int>> fronts(1);

    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) continue;
            // p dominates q (maximisation)?
            bool pDomQ = (v1(p) >  v1(q) && v2(p) >  v2(q))
                      || (v1(p) >= v1(q) && v2(p) >  v2(q))
                      || (v1(p) >  v1(q) && v2(p) >= v2(q));
            // q dominates p?
            bool qDomP = (v1(q) >  v1(p) && v2(q) >  v2(p))
                      || (v1(q) >= v1(p) && v2(q) >  v2(p))
                      || (v1(q) >  v1(p) && v2(q) >= v2(p));

            if (pDomQ) {
                S[p].push_back(q);
            } else if (qDomP) {
                ++domCount[p];
            }
        }
        if (domCount[p] == 0) {
            rank[p] = 0;
            fronts[0].push_back(p);
        }
    }

    int i = 0;
    while (!fronts[i].empty()) {
        std::vector<int> Q;
        for (int p : fronts[i]) {
            for (int q : S[p]) {
                --domCount[q];
                if (domCount[q] == 0) {
                    rank[q] = i + 1;
                    Q.push_back(q);
                }
            }
        }
        ++i;
        fronts.push_back(Q);
    }
    fronts.pop_back();  // remove trailing empty front
    return fronts;
}

// -------------------------------------------------------------------------
// getCrowdingDist – crowding distance within a single front.
// Mirrors Python nsga_sort.py::getCrowdingDist.
// -------------------------------------------------------------------------
std::vector<double> getCrowdingDist(const std::vector<double>& objVector) {
    const int m = static_cast<int>(objVector.size());
    if (m == 0) return {};

    // Sort by objective value.
    std::vector<int> key(m);
    std::iota(key.begin(), key.end(), 0);
    std::sort(key.begin(), key.end(),
              [&](int a, int b){ return objVector[a] < objVector[b]; });

    std::vector<double> sorted(m);
    for (int i = 0; i < m; ++i) sorted[i] = objVector[key[i]];

    const double Inf = std::numeric_limits<double>::infinity();

    // Distance to neighbours (edges are Inf).
    std::vector<double> crowd(m);
    for (int i = 0; i < m; ++i) {
        double prev = (i > 0)   ? sorted[i-1] : Inf;
        double next = (i < m-1) ? sorted[i+1] : Inf;
        double prevDist = std::isinf(prev) ? Inf : std::abs(sorted[i] - prev);
        double nextDist = std::isinf(next) ? Inf : std::abs(sorted[i] - next);
        crowd[i] = prevDist + nextDist;
    }

    // Normalise by objective range.
    double range = sorted[m-1] - sorted[0];
    if (range > 0.0) {
        double inv = std::abs(1.0 / range);
        for (double& c : crowd) if (!std::isinf(c)) c *= inv;
    }

    // Restore original order.
    std::vector<double> dist(m);
    for (int i = 0; i < m; ++i) dist[key[i]] = crowd[i];
    return dist;
}

// -------------------------------------------------------------------------
// nsga_sort – returns rank for each individual (0 = best).
// -------------------------------------------------------------------------
std::vector<int> nsga_sort(const std::vector<std::vector<double>>& objVals) {
    auto fronts = getFronts(objVals);

    // Within each front, rank by crowding distance (descending = better).
    for (auto& front : fronts) {
        const int fSize = static_cast<int>(front.size());
        std::vector<double> x1(fSize), x2(fSize);
        for (int i = 0; i < fSize; ++i) {
            x1[i] = objVals[front[i]][0];
            x2[i] = objVals[front[i]][1];
        }
        std::vector<double> crowd = getCrowdingDist(x1);
        {
            auto crowd2 = getCrowdingDist(x2);
            for (int i = 0; i < fSize; ++i) crowd[i] += crowd2[i];
        }
        // Sort front indices by decreasing crowding distance.
        std::vector<int> order(fSize);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return crowd[a] > crowd[b]; });
        std::vector<int> sorted(fSize);
        for (int i = 0; i < fSize; ++i) sorted[i] = front[order[i]];
        front = sorted;
    }

    // Flatten fronts into a single ordering.
    std::vector<int> tmp;
    tmp.reserve(objVals.size());
    for (const auto& front : fronts)
        for (int ind : front) tmp.push_back(ind);

    // rank[individual] = its position in the flat ordering.
    std::vector<int> rank(tmp.size());
    for (int pos = 0; pos < static_cast<int>(tmp.size()); ++pos)
        rank[tmp[pos]] = pos;
    return rank;
}

} // namespace wann

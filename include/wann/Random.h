#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace wann {

// Global Mersenne-Twister, seeded once from main().
inline std::mt19937& rng() {
    static std::mt19937 gen(42);
    return gen;
}

inline void seedRng(uint32_t seed) { rng().seed(seed); }

// Uniform real in [lo, hi)
inline double randDouble(double lo = 0.0, double hi = 1.0) {
    return std::uniform_real_distribution<double>(lo, hi)(rng());
}

// Uniform integer in [lo, hi] (inclusive)
inline int randInt(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng());
}

template<typename T>
inline void shuffle(std::vector<T>& v) {
    std::shuffle(v.begin(), v.end(), rng());
}

} // namespace wann

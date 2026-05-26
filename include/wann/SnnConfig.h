#pragma once
#include <stdexcept>
#include <string>

namespace wann {

enum class SnnEncoder { CURRENT, POISSON, RATE, TTFS, TTFS_LOG, SMALL, LARGE };
enum class SnnDecoder { SPIKE_COUNT, RATE, FIRST_SPIKE, VOTING, WTA };

inline SnnEncoder parseEncoder(const std::string& s) {
    if (s == "current")  return SnnEncoder::CURRENT;
    if (s == "poisson")  return SnnEncoder::POISSON;
    if (s == "rate")     return SnnEncoder::RATE;
    if (s == "ttfs")     return SnnEncoder::TTFS;
    if (s == "ttfs_log") return SnnEncoder::TTFS_LOG;
    if (s == "small")    return SnnEncoder::SMALL;
    if (s == "large")    return SnnEncoder::LARGE;
    throw std::runtime_error("Unknown snn_encoder: \"" + s +
        "\" (valid: current, poisson, rate, ttfs, ttfs_log, small, large)");
}

inline SnnDecoder parseDecoder(const std::string& s) {
    if (s == "spike_count") return SnnDecoder::SPIKE_COUNT;
    if (s == "rate")        return SnnDecoder::RATE;
    if (s == "first_spike") return SnnDecoder::FIRST_SPIKE;
    if (s == "voting")      return SnnDecoder::VOTING;
    if (s == "wta")         return SnnDecoder::WTA;
    throw std::runtime_error("Unknown snn_decoder: \"" + s +
                             "\" (valid: spike_count, rate, first_spike, voting, wta)");
}

} // namespace wann

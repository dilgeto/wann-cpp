#include "../../include/wann/hw/OdinNetwork.h"

#include <algorithm>
#include <stdexcept>

namespace wann::hw {

OdinNetwork::OdinNetwork(const wann::OdinNetworkConfig& cfg, OdinDriver& driver)
    : driver_(driver)
{
    driver_.loadConfig(cfg);
    for (const auto& n : cfg.neurons) {
        if (n.role == "input")  inputAddrs_.push_back(n.addr);
        if (n.role == "output") outputAddrs_.push_back(n.addr);
    }
    lastOutputSpikes_.assign(outputAddrs_.size(), false);
}

void OdinNetwork::fastReset() {
    currentTime_ = 0.0;
    std::fill(lastOutputSpikes_.begin(), lastOutputSpikes_.end(), false);
    // ODIN has no exposed "reset membrane state" register in this driver yet
    // (see OdinDriver) — behaviourally this only resets host-side timing.
}

void OdinNetwork::applyInputSpikes(const std::vector<std::vector<double>>& encoded_spikes,
                                    double currentTime)
{
    const int t = static_cast<int>(currentTime);
    for (std::size_t ch = 0; ch < encoded_spikes.size() && ch < inputAddrs_.size(); ++ch) {
        if (t >= 0 && t < static_cast<int>(encoded_spikes[ch].size()) &&
            encoded_spikes[ch][static_cast<std::size_t>(t)] > 0.0) {
            driver_.sendSpike(inputAddrs_[ch]);
        }
    }
}

void OdinNetwork::setInputCurrents(const std::vector<double>& /*currents*/) {
    throw std::logic_error(
        "OdinNetwork::setInputCurrents: ODIN only takes AER spike events, not "
        "analog current injection — pick a spike-train encoder (ttfs/poisson/"
        "rate), not current/small/large, for hardware deployment.");
}

void OdinNetwork::step(double /*sharedWeight*/) {
    // TODO: pick a per-step polling budget consistent with SIM_WINDOW_MS —
    // ODIN is event-driven with no exposed clock, so "one software timestep"
    // has no exact hardware equivalent (see plan's timing-fidelity note).
    constexpr int STEP_TIMEOUT_US = 1000;  // placeholder, needs empirical tuning
    auto fired = driver_.pollOutputs(STEP_TIMEOUT_US);

    std::fill(lastOutputSpikes_.begin(), lastOutputSpikes_.end(), false);
    for (int addr : fired) {
        auto it = std::find(outputAddrs_.begin(), outputAddrs_.end(), addr);
        if (it != outputAddrs_.end())
            lastOutputSpikes_[static_cast<std::size_t>(it - outputAddrs_.begin())] = true;
    }
    currentTime_ += 1.0;
}

std::vector<bool> OdinNetwork::getOutputSpikes() const {
    return lastOutputSpikes_;
}

} // namespace wann::hw

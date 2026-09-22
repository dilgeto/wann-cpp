#include "../../include/wann/hw/OdinNetwork.h"

#include <algorithm>
#include <stdexcept>

namespace wann::hw {

OdinNetwork::OdinNetwork(const wann::OdinNetworkConfig& cfg, OdinDriver& driver)
    : driver_(driver)
{
    driver_.loadConfig(cfg);
    // loadConfig() deliberately leaves the chip gated (GATE_ACTIVITY=1) so
    // programming neurons/synapses is safe — nothing else in this path
    // called start() before this fix, so the chip sat stopped for every
    // hardware run so far: every sendVirtual() would time out waiting for an
    // ACK that a gated chip never sends (see the aer_in handshake failures).
    driver_.start();
    // Only walk the non-twin roles, in address order — that reproduces the
    // exact channel/output ordering SnnCarOdinTask expects (0=bias,
    // 1..nInput, then outputs at fixed indices). For each, also grab its
    // twin's address (if any, see buildOdinConfig's Dale's-law splitting)
    // so both physical copies get stimulated/read together.
    for (const auto& n : cfg.neurons) {
        if (n.role == "input") {
            std::vector<int> addrs{n.addr};
            if (n.twinAddr >= 0) addrs.push_back(n.twinAddr);
            inputAddrs_.push_back(std::move(addrs));
        }
        if (n.role == "output") {
            std::vector<int> addrs{n.addr};
            if (n.twinAddr >= 0) addrs.push_back(n.twinAddr);
            outputAddrs_.push_back(std::move(addrs));
        }
    }
    lastOutputSpikes_.assign(outputAddrs_.size(), false);
}

void OdinNetwork::fastReset() {
    currentTime_ = 0.0;
    std::fill(lastOutputSpikes_.begin(), lastOutputSpikes_.end(), false);
    // ODIN has no exposed "reset membrane state" register in this driver yet
    // (see OdinDriver) — behaviourally this only resets host-side timing.
    // aerBusReset() IS a real hardware operation though: clears any AER
    // handshake left mid-transaction by the previous episode before this one
    // starts sending new events into it.
    driver_.aerBusReset();
}

void OdinNetwork::applyInputSpikes(const std::vector<std::vector<double>>& encoded_spikes,
                                    double currentTime)
{
    const int t = static_cast<int>(currentTime);
    for (std::size_t ch = 0; ch < encoded_spikes.size() && ch < inputAddrs_.size(); ++ch) {
        if (t >= 0 && t < static_cast<int>(encoded_spikes[ch].size()) &&
            encoded_spikes[ch][static_cast<std::size_t>(t)] > 0.0) {
            // Weight 7 (max drive) matches the convention already validated
            // in the user's working Iris pipeline (odin.py's run_sample) —
            // input stimulation bypasses the synaptic crossbar entirely via
            // a virtual event, it isn't a real synapse.
            for (int addr : inputAddrs_[ch]) {
                int rc = driver_.sendVirtual(addr, /*weight=*/7);
                if (rc != 0) {
                    throw std::runtime_error(
                        "OdinNetwork::applyInputSpikes: aer_in handshake failed "
                        "(rc=" + std::to_string(rc) + ") sending to addr " +
                        std::to_string(addr) + " — AER bus likely desynced, "
                        "stopping instead of continuing into a possibly worse "
                        "hardware state.");
                }
            }
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
    auto fired = driver_.drainSpikes(STEP_TIMEOUT_US);

    std::fill(lastOutputSpikes_.begin(), lastOutputSpikes_.end(), false);
    for (int addr : fired) {
        for (std::size_t k = 0; k < outputAddrs_.size(); ++k) {
            const auto& addrs = outputAddrs_[k];
            if (std::find(addrs.begin(), addrs.end(), addr) != addrs.end()) {
                lastOutputSpikes_[k] = true;
                break;
            }
        }
    }
    currentTime_ += 1.0;
}

std::vector<bool> OdinNetwork::getOutputSpikes() const {
    return lastOutputSpikes_;
}

} // namespace wann::hw

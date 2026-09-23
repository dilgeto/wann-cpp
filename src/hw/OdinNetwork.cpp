#include "../../include/wann/hw/OdinNetwork.h"

#include <algorithm>
#include <stdexcept>

namespace wann::hw {

OdinNetwork::OdinNetwork(const wann::OdinNetworkConfig& cfg, OdinDriver& driver)
    : driver_(driver)
{
    driver_.loadConfig(cfg);
    // init() (called inside loadConfig()) zeroes every global register,
    // including BURST_TIMEREF/AER_SRC_CTRL — set them unconditionally
    // rather than only when this genome happens to use a bursting
    // NeuronType (CHATTERING/INTRINSICALLY_BURSTING), since a mixed genome
    // could combine those with non-bursting types and this has no downside
    // for the ones that don't burst.
    driver_.setBurstTimeref(1023);
    driver_.setAerSrcCtrl(true);
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

namespace {
// Matches odin.py's run_sample() default drain_between_us — the window
// given to AER_OUT after each stimulated input, before moving on to the
// next one.
constexpr int kDrainBetweenUs = 3000;
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

void OdinNetwork::recordFired(const std::vector<int>& fired) {
    for (int addr : fired) {
        if (addr >= 0 && addr < 256) ++addrSpikes_[addr];
        for (std::size_t k = 0; k < outputAddrs_.size(); ++k) {
            const auto& addrs = outputAddrs_[k];
            if (std::find(addrs.begin(), addrs.end(), addr) != addrs.end()) {
                lastOutputSpikes_[k] = true;
                break;
            }
        }
    }
}

void OdinNetwork::applyInputSpikes(const std::vector<std::vector<double>>& encoded_spikes,
                                    double currentTime)
{
    // Cleared once per tick, here (the first thing SnnCarOdinTask::runEpisode
    // calls each tick) rather than in step() — step() now only accumulates,
    // since spikes fired during input injection (see below) must survive
    // into step()'s own drain instead of being wiped by it.
    std::fill(lastOutputSpikes_.begin(), lastOutputSpikes_.end(), false);

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
            // Drain AER_OUT right after this input's virtual event(s), same
            // as odin.py's run_sample() — without this, a spike triggered by
            // this input sits unread in AER_OUT and backpressures the chip's
            // AER_IN handshake for the next input (exactly the "works for
            // the first input, times out on the second" failure mode odin.py's
            // docstring already warns about).
            recordFired(driver_.drainSpikes(kDrainBetweenUs));
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

    // A hardware "step" is applyInputSpikes() (already done for this tick)
    // plus one tref — without this, IZH neurons never advance: refractory
    // periods never expire (a neuron fires once and then stays silent
    // forever) and latency/rebound behaviours never reach the delay they
    // need to fire at all. See OdinDriver::sendTrefAll's doc comment.
    int rc = driver_.sendTrefAll();
    if (rc != 0) {
        throw std::runtime_error(
            "OdinNetwork::step: tref handshake failed (rc=" + std::to_string(rc) +
            ") — AER bus likely desynced.");
    }

    // Accumulates onto whatever applyInputSpikes() already recorded this
    // tick via its own interleaved drains — does NOT clear lastOutputSpikes_
    // (that happens once per tick, at the top of applyInputSpikes()).
    recordFired(driver_.drainSpikes(STEP_TIMEOUT_US));
    currentTime_ += 1.0;
}

std::vector<bool> OdinNetwork::getOutputSpikes() const {
    return lastOutputSpikes_;
}

} // namespace wann::hw

#pragma once
#include "../OdinExport.h"
#include "OdinDriver.h"

#include <vector>

namespace wann::hw {

// Hardware-backed stand-in for snn-simulator's `Network`, used by
// SnnCarOdinTask instead of the software Izhikevich integration. Unlike
// `Network`, the topology isn't built incrementally — it's already fixed by
// the OdinNetworkConfig loaded into the chip once at startup
// (OdinDriver::loadConfig, from wann_car_odin_export's output). Each
// `step()` sends one AER event per input channel that has a spike this
// timestep and collects whatever fires back within the step's time budget.
class OdinNetwork {
public:
    OdinNetwork(const wann::OdinNetworkConfig& cfg, OdinDriver& driver);

    void fastReset();

    // encoded_spikes[channel][t] > 0 means a spike at relative time t for
    // that input channel — same shape TTFSEncoder/etc. already produce in
    // SnnCarTask.cpp. Only the spikes at `currentTime` are sent this call.
    void applyInputSpikes(const std::vector<std::vector<double>>& encoded_spikes,
                          double currentTime);

    // ODIN only accepts discrete AER events, not analog current injection —
    // current-injection encoders (SnnEncoder::CURRENT/SMALL/LARGE) aren't
    // deployable on this backend. Throws std::logic_error if called.
    void setInputCurrents(const std::vector<double>& currents);

    // sharedWeight is accepted for API parity with Network::step but ignored
    // — the weight is already baked into the config loaded onto the chip.
    void step(double sharedWeight);

    std::vector<bool> getOutputSpikes() const;
    double getCurrentTime() const { return currentTime_; }

private:
    OdinDriver& driver_;
    // channel/output index -> physical ODIN address(es). A logical
    // input/output can have two physical addresses if buildOdinConfig had
    // to split it for Dale's-law sign consistency (see OdinExport.h) — both
    // must be stimulated/read together so the twins stay in lockstep.
    std::vector<std::vector<int>> inputAddrs_;
    std::vector<std::vector<int>> outputAddrs_;
    double            currentTime_ = 0.0;
    std::vector<bool> lastOutputSpikes_;
};

} // namespace wann::hw

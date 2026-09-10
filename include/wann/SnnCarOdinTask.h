#pragma once

#include "Hyperparams.h"
#include "OdinExport.h"
#include "hw/OdinDriver.h"
#include "hw/OdinNetwork.h"

#include <utility>
#include <vector>

namespace wann {

// Hardware counterpart to SnnCarTask (src/SnnCarTask.cpp), for running an
// already-evolved, already-exported (wann_car_odin_export) car network on
// the real ODIN core instead of the software Izhikevich simulator.
//
// Deliberately NOT a template/refactor of SnnCarTask — this is a separate,
// narrower implementation so the actively-evolving SnnCarTask.cpp (used by
// screening/evolution) is never put at risk by hardware-specific code.
// Scope is intentionally narrow: only the ttfs encoder / first_spike
// decoder combination is supported, since that's what every car model
// exported so far uses — ODIN takes AER spike events only, so
// current-injection encoders (current/small/large) aren't supported at all
// (see OdinNetwork::setInputCurrents). If SnnCarTask.cpp's normalization
// constants (BOUND, VX_MAX, VY_MAX, OMEGA_MAX) or observation wiring change,
// mirror the change here too.
//
// simWindowMs must match the SIM_WINDOW_MS the source model was actually
// trained/evolved with (car_snn.json has no such field — it's a compile-time
// constant in SnnCarTask.h, WANN_CAR_SIM_WINDOW_MS, default 40 — some
// exported models, e.g. pruned ones, may have been trained at a different
// window, so this must be passed explicitly rather than assumed).
class SnnCarOdinTask {
public:
    SnnCarOdinTask(const Hyperparams& hyp, const OdinNetworkConfig& cfg,
                  hw::OdinDriver& driver, double simWindowMs);

    void setEpisodeSteps(int n) { episodeSteps_ = n; }

    // Runs one episode on real hardware, returns total reward.
    double runEpisode(long long episodeSeed);

    // Runs nEpisodes, returns the reward of each (shaped == original, as in
    // SnnCarTask — Car has no reward shaping).
    std::vector<double> evalEpisodes(int nEpisodes, int baseSeed);

private:
    int    nInput_;
    int    episodeSteps_;
    double simWindowMs_;
    hw::OdinNetwork network_;
};

} // namespace wann

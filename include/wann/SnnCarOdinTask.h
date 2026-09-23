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
// simWindowMs must match the snn_window_ms the source model was actually
// trained/evolved with (Hyperparams field, car_snn.json's snn_window_ms —
// used to be a compile-time constant, WANN_CAR_SIM_WINDOW_MS, before that
// field existed). Some exported models, e.g. pruned ones, may have been
// trained at a different window, so this must be passed explicitly rather
// than assumed/read from the config.
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

    // Wall-clock breakdown accumulated across every runEpisode() call so
    // far — answers "is the bottleneck the RL environment or the ODIN
    // hardware path" empirically on the actual board, since OdinNetwork's
    // per-tick polling window (see OdinNetwork::step) is a fixed real-time
    // cost that doesn't show up in a normal software-simulator profile.
    struct Timings {
        double envUs = 0.0;      // rl-tools observe/step/reward/terminated
        double encodeUs = 0.0;   // TTFSEncoder::encode
        double aerSendUs = 0.0;  // OdinNetwork::applyInputSpikes (virtual events out)
        double aerDrainUs = 0.0; // OdinNetwork::step (AER_OUT polling wait)
        double decodeUs = 0.0;   // RLDecoder::decodeContinuousAction
        long long nEnvSteps = 0;
        long long nTicks = 0;    // inner window ticks (nEnvSteps * window_steps)
    };
    const Timings& timings() const { return timings_; }
    const hw::OdinNetwork& network() const { return network_; }

private:
    int    nInput_;
    int    episodeSteps_;
    double simWindowMs_;
    hw::OdinNetwork network_;
    Timings timings_;
};

} // namespace wann

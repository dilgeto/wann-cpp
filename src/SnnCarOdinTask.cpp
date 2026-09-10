#include "../include/wann/SnnCarOdinTask.h"

#include <decoding/rlDecoder.hpp>
#include <encoding/ttfsEncoder.hpp>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/car/operations_generic.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rlt = rl_tools;

namespace {
using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using CarSpec   = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 50>;
using Env       = rlt::rl::environments::CarTrack<CarSpec>;
using RNG       = typename rlt::devices::random::CPU::ENGINE<>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 9, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;

// Mirrors SnnCarTask.h (VX_MAX/VY_MAX/OMEGA_MAX/BOUND) — keep in sync.
constexpr double VX_MAX    = 3.0;
constexpr double VY_MAX    = 2.0;
constexpr double OMEGA_MAX = 10.0;
constexpr double BOUND     = CarSpec::TRACK_SCALE * 100 / 2.0;
constexpr double DT = 1.0;
} // namespace

namespace wann {

SnnCarOdinTask::SnnCarOdinTask(const Hyperparams& hyp, const OdinNetworkConfig& cfg,
                                hw::OdinDriver& driver, double simWindowMs)
    : nInput_(hyp.ann_nInput)
    , episodeSteps_(1000)
    , simWindowMs_(simWindowMs)
    , network_(cfg, driver)
{
    if (hyp.snn_encoder != "ttfs" || hyp.snn_decoder != "first_spike") {
        throw std::runtime_error(
            "SnnCarOdinTask only supports snn_encoder=ttfs / snn_decoder=first_spike "
            "(got encoder=" + hyp.snn_encoder + " decoder=" + hyp.snn_decoder + ") — "
            "this is the narrow hardware path for car_ttfs_first_spike_40ms, not a "
            "general ODIN backend for every encoder/decoder combination.");
    }
}

double SnnCarOdinTask::runEpisode(long long episodeSeed) {
    network_.fastReset();
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::init(device, env);
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::car::ObservationCarTrack<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(simWindowMs_);
    const int n_channels   = nInput_ + 1;  // bias + observations

    TTFSEncoder enc(TTFSEncoder::Mapping::LINEAR, 1e-9);
    RLDecoder rl_decoder(RLDecoder::DecodingType::FIRST_SPIKE, simWindowMs_);

    double total_reward = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        std::vector<double> norm(n_channels, 0.0);
        norm[0] = 1.0;
        if (nInput_ >= 1) norm[1] = (rlt::get(obs_mat, 0, 0) + BOUND) / (2.0 * BOUND);
        if (nInput_ >= 2) norm[2] = (rlt::get(obs_mat, 0, 1) + BOUND) / (2.0 * BOUND);
        if (nInput_ >= 3) norm[3] = (rlt::get(obs_mat, 0, 2) + M_PI) / (2.0 * M_PI);
        if (nInput_ >= 4) norm[4] = (rlt::get(obs_mat, 0, 3) + VX_MAX) / (2.0 * VX_MAX);
        if (nInput_ >= 5) norm[5] = (rlt::get(obs_mat, 0, 4) + VY_MAX) / (2.0 * VY_MAX);
        if (nInput_ >= 6) norm[6] = (rlt::get(obs_mat, 0, 5) + OMEGA_MAX) / (2.0 * OMEGA_MAX);
        if (nInput_ >= 7) norm[7] = rlt::get(obs_mat, 0, 6);
        if (nInput_ >= 8) norm[8] = rlt::get(obs_mat, 0, 7);
        if (nInput_ >= 9) norm[9] = rlt::get(obs_mat, 0, 8);

        std::vector<std::vector<double>> spike_trains(n_channels);
        for (int ch = 0; ch < n_channels; ++ch) {
            double v = std::clamp(norm[ch], 0.0, 1.0);
            spike_trains[ch] = enc.encode(v, simWindowMs_, DT);
        }

        std::vector<double> out_spikes0, out_spikes1;
        for (int t = 0; t < window_steps; ++t) {
            network_.applyInputSpikes(spike_trains, network_.getCurrentTime());
            network_.step(0.0);  // shared weight already baked into the loaded config
            const auto& out = network_.getOutputSpikes();
            if (out.size() > 0 && out[0]) out_spikes0.push_back(static_cast<double>(t));
            if (out.size() > 1 && out[1]) out_spikes1.push_back(static_cast<double>(t));
        }

        double throttle = rl_decoder.decodeContinuousAction(out_spikes0) * 2.0 - 1.0;
        double steering = rl_decoder.decodeContinuousAction(out_spikes1) * 2.0 - 1.0;
        throttle = std::clamp(throttle, -1.0, 1.0);
        steering = std::clamp(steering, -1.0, 1.0);

        rlt::set(action_mat, 0, 0, throttle);
        rlt::set(action_mat, 0, 1, steering);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return total_reward;
}

std::vector<double> SnnCarOdinTask::evalEpisodes(int nEpisodes, int baseSeed) {
    std::vector<double> rewards(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i)
        rewards[i] = runEpisode(baseSeed + i);
    return rewards;
}

} // namespace wann

#include "../include/wann/SnnCarTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>
#include <memory>

#include <rl_tools/operations/cpu.h>
// Use operations_generic (not operations_cpu) to avoid the BMP file loader.
// The generic init() fills the track from the hardcoded default_track in track.h.
#include <rl_tools/rl/environments/car/operations_generic.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;

// Track: 100×100 pixels, 50mm per pixel → 5m × 5m arena.
using CarSpec   = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 50>;
using Env       = rlt::rl::environments::CarTrack<CarSpec>;
using RNG       = typename rlt::devices::random::CPU::ENGINE<>;
// 9-dim observation: x, y, mu, vx, vy, omega + 3 lidar distances
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 9, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;

namespace {
std::unique_ptr<Encoder> makeEncoder(wann::SnnEncoder type, uint32_t seed) {
    constexpr double MAX_RATE   = 100.0;
    constexpr double REF_PERIOD = 2.0;
    switch (type) {
        case wann::SnnEncoder::POISSON: {
            auto e = std::make_unique<PoissonEncoder>(MAX_RATE, true, REF_PERIOD);
            e->reseed(seed);
            return e;
        }
        case wann::SnnEncoder::RATE: {
            auto e = std::make_unique<RateEncoder>(MAX_RATE);
            e->reseed(seed);
            return e;
        }
        case wann::SnnEncoder::TTFS:
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LINEAR, 1e-9);
        case wann::SnnEncoder::TTFS_LOG:
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LOGARITHMIC, 1e-9);
        case wann::SnnEncoder::SMALL:
        case wann::SnnEncoder::LARGE:
            return nullptr;  // current injection, not spike trains
        default:
            return nullptr;  // CURRENT handled separately
    }
}
} // anonymous namespace

namespace wann {

const double SnnCarTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 2.0, 3.0, 5.0, 8.0
};

SnnCarTask::SnnCarTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnCarTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

Network SnnCarTask::buildNetwork(const Ind& ind) const
{
    Network net(1.0, true);
    std::unordered_map<int, int> snn_id;
    snn_id.reserve(ind.nodes.size());

    for (const auto& ng : ind.nodes) {
        NeuronType nt = wannActToNeuronType(ng.activation);
        int sid;
        switch (ng.type) {
            case 4: sid = net.addInputNeuron(nt);  break;
            case 1: sid = net.addInputNeuron(nt);  break;
            case 3: sid = net.addHiddenNeuron(nt); break;
            case 2: sid = net.addOutputNeuron(nt); break;
            default: continue;
        }
        snn_id[ng.id] = sid;
    }

    for (const auto& cg : ind.conns) {
        if (!cg.enabled) continue;
        auto it_src = snn_id.find(cg.src);
        auto it_dst = snn_id.find(cg.dst);
        if (it_src == snn_id.end() || it_dst == snn_id.end()) continue;
        net.addSynapse(it_src->second, it_dst->second, cg.excitatory);
    }

    return net;
}

Network SnnCarTask::buildNetwork(const std::vector<double>& wVec,
                                 const std::vector<int>&    aVec) const
{
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    Network net(1.0, true);
    std::vector<int> snn_id(N, -1);

    snn_id[0] = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    for (int i = 1; i <= nInput_; ++i)
        snn_id[i] = net.addInputNeuron(wannActToNeuronType(aVec[i]));
    for (int i = nInput_ + 1; i < N - nOutput_; ++i)
        snn_id[i] = net.addHiddenNeuron(wannActToNeuronType(aVec[i]));
    for (int i = N - nOutput_; i < N; ++i)
        snn_id[i] = net.addOutputNeuron(wannActToNeuronType(aVec[i]));

    for (int i = 0; i < N; ++i) {
        if (snn_id[i] < 0) continue;
        for (int j = 0; j < N; ++j) {
            if (snn_id[j] < 0 || i == j) continue;
            if (wVec[i * N + j] != 0.0)
                net.addSynapse(snn_id[i], snn_id[j], wVec[i * N + j] > 0.0);
        }
    }

    return net;
}

// Decode two output spike trains into (throttle, steering) ∈ [-1, 1].
// FIRST_SPIKE: latency → [0,1] → [-1,1].
// SPIKE_COUNT: min(1, n/10)*2-1 (RLDecoder handles saturation).
// default (RATE/TTFS/POISSON/RATE_ARGMAX/VOTING): n_spikes / max_spikes → [-1,1].
static std::pair<double,double> decodeActions(
    const std::vector<double>& spikes0,
    const std::vector<double>& spikes1,
    SnnDecoder decoder,
    const RLDecoder& rl_decoder,
    double max_spikes)
{
    double t, s;
    switch (decoder) {
        case SnnDecoder::FIRST_SPIKE:
            t = rl_decoder.decodeContinuousAction(spikes0) * 2.0 - 1.0;
            s = rl_decoder.decodeContinuousAction(spikes1) * 2.0 - 1.0;
            break;
        case SnnDecoder::SPIKE_COUNT:
            t = rl_decoder.decodeContinuousAction(spikes0);
            s = rl_decoder.decodeContinuousAction(spikes1);
            break;
        default:
            t = static_cast<double>(spikes0.size()) / max_spikes * 2.0 - 1.0;
            s = static_cast<double>(spikes1.size()) / max_spikes * 2.0 - 1.0;
            break;
    }
    return { std::clamp(t, -1.0, 1.0), std::clamp(s, -1.0, 1.0) };
}

std::pair<double,double> SnnCarTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    net.fastReset();
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));

    // Fill track bitmap from hardcoded default_track (generic init, not BMP loader).
    rlt::init(device, env);
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::car::ObservationCarTrack<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    // Track bounds (TRACK_SCALE=0.05, WIDTH=HEIGHT=100).
    constexpr double BOUND = CarSpec::TRACK_SCALE * 100 / 2.0;  // ±2.5 m

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;  // bias + 9 observations

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    const RLDecoder::DecodingType dec_type =
        (decoder_ == SnnDecoder::FIRST_SPIKE) ? RLDecoder::DecodingType::FIRST_SPIKE :
        (decoder_ == SnnDecoder::SPIKE_COUNT) ? RLDecoder::DecodingType::SPIKE_COUNT :
                                                RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    double total_reward = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (resetBetweenSteps_) net.fastReset();

        std::vector<double> out_spikes0, out_spikes1;

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            // Position
            if (nInput_ >= 1) norm[1] = (rlt::get(obs_mat, 0, 0) + BOUND) / (2.0 * BOUND);
            if (nInput_ >= 2) norm[2] = (rlt::get(obs_mat, 0, 1) + BOUND) / (2.0 * BOUND);
            // Heading
            if (nInput_ >= 3) norm[3] = (rlt::get(obs_mat, 0, 2) + M_PI) / (2.0 * M_PI);
            // Velocities
            if (nInput_ >= 4) norm[4] = (rlt::get(obs_mat, 0, 3) + VX_MAX) / (2.0 * VX_MAX);
            if (nInput_ >= 5) norm[5] = (rlt::get(obs_mat, 0, 4) + VY_MAX) / (2.0 * VY_MAX);
            if (nInput_ >= 6) norm[6] = (rlt::get(obs_mat, 0, 5) + OMEGA_MAX) / (2.0 * OMEGA_MAX);
            // Lidar (already in [0,1])
            if (nInput_ >= 7) norm[7] = rlt::get(obs_mat, 0, 6);
            if (nInput_ >= 8) norm[8] = rlt::get(obs_mat, 0, 7);
            if (nInput_ >= 9) norm[9] = rlt::get(obs_mat, 0, 8);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                if (out.size() > 0 && out[0]) out_spikes0.push_back(static_cast<double>(t));
                if (out.size() > 1 && out[1]) out_spikes1.push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;

            if (encoder_ == SnnEncoder::SMALL || encoder_ == SnnEncoder::LARGE) {
                RLEncoder rl_enc(encoder_ == SnnEncoder::SMALL
                                 ? RLEncoder::EncodingType::SMALL
                                 : RLEncoder::EncodingType::LARGE,
                                 100.0, 5, static_cast<size_t>(neuronsPerVar_));
                const int n_obs = (encoder_ == SnnEncoder::SMALL)
                                  ? nInput_ / 2
                                  : nInput_ / neuronsPerVar_;
                const double raw_obs[9] = {
                    rlt::get(obs_mat, 0, 0), rlt::get(obs_mat, 0, 1), rlt::get(obs_mat, 0, 2),
                    rlt::get(obs_mat, 0, 3), rlt::get(obs_mat, 0, 4), rlt::get(obs_mat, 0, 5),
                    rlt::get(obs_mat, 0, 6), rlt::get(obs_mat, 0, 7), rlt::get(obs_mat, 0, 8),
                };
                std::vector<double> obs_vals(raw_obs, raw_obs + std::min(n_obs, 9));
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    constexpr std::array<std::pair<double,double>, 9> OBS_LIMITS = {{
                        {-BOUND,     BOUND},
                        {-BOUND,     BOUND},
                        {-M_PI,      M_PI},
                        {-VX_MAX,    VX_MAX},
                        {-VY_MAX,    VY_MAX},
                        {-OMEGA_MAX, OMEGA_MAX},
                        {0.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 1.0},
                    }};
                    const std::vector<std::pair<double,double>> limits(
                        OBS_LIMITS.begin(), OBS_LIMITS.begin() + obs_vals.size());
                    enc_currents = rl_enc.encodeObservationLarge(obs_vals, limits);
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: normalized injection
                if (nInput_ >= 1) currents[1] = (rlt::get(obs_mat, 0, 0) + BOUND) / (2.0 * BOUND) * 20.0;
                if (nInput_ >= 2) currents[2] = (rlt::get(obs_mat, 0, 1) + BOUND) / (2.0 * BOUND) * 20.0;
                if (nInput_ >= 3) currents[3] = (rlt::get(obs_mat, 0, 2) + M_PI) / (2.0 * M_PI) * 20.0;
                if (nInput_ >= 4) currents[4] = (rlt::get(obs_mat, 0, 3) + VX_MAX) / (2.0 * VX_MAX) * 20.0;
                if (nInput_ >= 5) currents[5] = (rlt::get(obs_mat, 0, 4) + VY_MAX) / (2.0 * VY_MAX) * 20.0;
                if (nInput_ >= 6) currents[6] = (rlt::get(obs_mat, 0, 5) + OMEGA_MAX) / (2.0 * OMEGA_MAX) * 20.0;
                if (nInput_ >= 7) currents[7] = rlt::get(obs_mat, 0, 6) * 20.0;
                if (nInput_ >= 8) currents[8] = rlt::get(obs_mat, 0, 7) * 20.0;
                if (nInput_ >= 9) currents[9] = rlt::get(obs_mat, 0, 8) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                if (out.size() > 0 && out[0]) out_spikes0.push_back(static_cast<double>(t));
                if (out.size() > 1 && out[1]) out_spikes1.push_back(static_cast<double>(t));
            }
        }

        auto [throttle, steering] = decodeActions(
            out_spikes0, out_spikes1, decoder_, rl_decoder, max_spikes);

        rlt::set(action_mat, 0, 0, throttle);
        rlt::set(action_mat, 0, 1, steering);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return {total_reward, total_reward};
}

std::vector<double> SnnCarTask::evaluate(const Ind& ind, int seed)
{
    Network net = buildNetwork(ind);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            int episodeSeed = (seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed).first;
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

std::vector<double> SnnCarTask::getDistFitness(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        int seed)
{
    Network net = buildNetwork(wVec, aVec);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            int episodeSeed = (seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed).first;
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

std::pair<std::vector<double>,std::vector<double>> SnnCarTask::evalEpisodes(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        double weight, int nEpisodes, int baseSeed) const
{
    Network net = buildNetwork(wVec, aVec);
    std::vector<double> shaped(nEpisodes), original(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i) {
        auto [s, o] = runEpisode(net, weight, baseSeed + i);
        shaped[i]   = s;
        original[i] = o;
    }
    return {shaped, original};
}

void SnnCarTask::exportTrajectory(const std::vector<double>& wVec,
                                   const std::vector<int>&    aVec,
                                   int bestWi, int evalSeed,
                                   const std::string& outFile,
                                   bool directSeed) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,"
           "throttle,steering,reward\n";

    Network net = buildNetwork(wVec, aVec);

    int    episodeSeed;
    if (directSeed) {
        episodeSeed = evalSeed;
    } else {
        episodeSeed = evalSeed * 10000 + bestWi * 100 + 0;
    }
    const double weight = WEIGHT_VALS[bestWi];

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

    constexpr double BOUND = CarSpec::TRACK_SCALE * 100 / 2.0;
    const int  window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int  n_channels   = nInput_ + 1;

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    const RLDecoder::DecodingType dec_type =
        (decoder_ == SnnDecoder::FIRST_SPIKE) ? RLDecoder::DecodingType::FIRST_SPIKE :
        (decoder_ == SnnDecoder::SPIKE_COUNT) ? RLDecoder::DecodingType::SPIKE_COUNT :
                                                RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        double x       = rlt::get(obs_mat, 0, 0);
        double y       = rlt::get(obs_mat, 0, 1);
        double mu      = rlt::get(obs_mat, 0, 2);
        double vx      = rlt::get(obs_mat, 0, 3);
        double vy      = rlt::get(obs_mat, 0, 4);
        double omega   = rlt::get(obs_mat, 0, 5);
        double lidar_l = rlt::get(obs_mat, 0, 6);
        double lidar_c = rlt::get(obs_mat, 0, 7);
        double lidar_r = rlt::get(obs_mat, 0, 8);

        if (resetBetweenSteps_) net.fastReset();

        std::vector<double> out_spikes0, out_spikes1;

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (x + BOUND) / (2.0 * BOUND);
            if (nInput_ >= 2) norm[2] = (y + BOUND) / (2.0 * BOUND);
            if (nInput_ >= 3) norm[3] = (mu + M_PI) / (2.0 * M_PI);
            if (nInput_ >= 4) norm[4] = (vx + VX_MAX) / (2.0 * VX_MAX);
            if (nInput_ >= 5) norm[5] = (vy + VY_MAX) / (2.0 * VY_MAX);
            if (nInput_ >= 6) norm[6] = (omega + OMEGA_MAX) / (2.0 * OMEGA_MAX);
            if (nInput_ >= 7) norm[7] = lidar_l;
            if (nInput_ >= 8) norm[8] = lidar_c;
            if (nInput_ >= 9) norm[9] = lidar_r;

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }
            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(weight);
                const auto& out = net.getOutputSpikes();
                if (out.size() > 0 && out[0]) out_spikes0.push_back(static_cast<double>(t));
                if (out.size() > 1 && out[1]) out_spikes1.push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;

            if (encoder_ == SnnEncoder::SMALL || encoder_ == SnnEncoder::LARGE) {
                RLEncoder rl_enc(encoder_ == SnnEncoder::SMALL
                                 ? RLEncoder::EncodingType::SMALL
                                 : RLEncoder::EncodingType::LARGE,
                                 100.0, 5, static_cast<size_t>(neuronsPerVar_));
                const int n_obs = (encoder_ == SnnEncoder::SMALL)
                                  ? nInput_ / 2
                                  : nInput_ / neuronsPerVar_;
                const double raw_obs[9] = { x, y, mu, vx, vy, omega, lidar_l, lidar_c, lidar_r };
                std::vector<double> obs_vals(raw_obs, raw_obs + std::min(n_obs, 9));
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    constexpr std::array<std::pair<double,double>, 9> OBS_LIMITS = {{
                        {-BOUND,     BOUND},
                        {-BOUND,     BOUND},
                        {-M_PI,      M_PI},
                        {-VX_MAX,    VX_MAX},
                        {-VY_MAX,    VY_MAX},
                        {-OMEGA_MAX, OMEGA_MAX},
                        {0.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 1.0},
                    }};
                    const std::vector<std::pair<double,double>> limits(
                        OBS_LIMITS.begin(), OBS_LIMITS.begin() + obs_vals.size());
                    enc_currents = rl_enc.encodeObservationLarge(obs_vals, limits);
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: normalized injection
                if (nInput_ >= 1) currents[1] = (x + BOUND) / (2.0 * BOUND) * 20.0;
                if (nInput_ >= 2) currents[2] = (y + BOUND) / (2.0 * BOUND) * 20.0;
                if (nInput_ >= 3) currents[3] = (mu + M_PI) / (2.0 * M_PI) * 20.0;
                if (nInput_ >= 4) currents[4] = (vx + VX_MAX) / (2.0 * VX_MAX) * 20.0;
                if (nInput_ >= 5) currents[5] = (vy + VY_MAX) / (2.0 * VY_MAX) * 20.0;
                if (nInput_ >= 6) currents[6] = (omega + OMEGA_MAX) / (2.0 * OMEGA_MAX) * 20.0;
                if (nInput_ >= 7) currents[7] = lidar_l * 20.0;
                if (nInput_ >= 8) currents[8] = lidar_c * 20.0;
                if (nInput_ >= 9) currents[9] = lidar_r * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(weight);
                const auto& out = net.getOutputSpikes();
                if (out.size() > 0 && out[0]) out_spikes0.push_back(static_cast<double>(t));
                if (out.size() > 1 && out[1]) out_spikes1.push_back(static_cast<double>(t));
            }
        }

        auto [throttle, steering] = decodeActions(
            out_spikes0, out_spikes1, decoder_, rl_decoder, max_spikes);

        rlt::set(action_mat, 0, 0, throttle);
        rlt::set(action_mat, 0, 1, steering);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        double reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        csv << step    << ','
            << x       << ',' << y       << ',' << mu    << ','
            << vx      << ',' << vy      << ',' << omega << ','
            << lidar_l << ',' << lidar_c << ',' << lidar_r << ','
            << throttle << ',' << steering << ',' << reward << '\n';

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

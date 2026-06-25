#include "../include/wann/SnnMountainCarTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>
#include <memory>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/mountain_car/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace rlt = rl_tools;

using T       = double;
using TI      = size_t;
using DEVICE  = rlt::devices::DefaultCPU;
using MCSpec  = rlt::rl::environments::mountain_car::Specification<T, TI>;
using Env     = rlt::rl::environments::MountainCarContinuous<MCSpec>;
using RNG     = typename rlt::devices::random::CPU::ENGINE<>;
// Observation: [position, velocity]
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;

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

const double SnnMountainCarTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 1.5, 2.0, 5.0, 8.0
};

SnnMountainCarTask::SnnMountainCarTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , shapingScale_(hyp.reward_shaping_scale)
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnMountainCarTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        case 7:  return NeuronType::FAST_SPIKING;
        case 8:  return NeuronType::REGULAR_SPIKING;
        case 9:  return NeuronType::REGULAR_SPIKING;
        case 10: return NeuronType::CHATTERING;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

Network SnnMountainCarTask::buildNetwork(const Ind& ind) const
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

Network SnnMountainCarTask::buildNetwork(const std::vector<double>& wVec,
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

std::pair<double,double> SnnMountainCarTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    net.fastReset();
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mountain_car::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    // Normalisation ranges (from DefaultParameters)
    constexpr double POS_MIN = -1.2,  POS_MAX =  0.6;   // range = 1.8
    constexpr double VEL_MAX =  0.07;                    // symmetric ±0.07

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;   // bias + observations

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    // Decoder
    const bool use_rldecoder = (decoder_ != SnnDecoder::SPIKE_COUNT);
    const RLDecoder::DecodingType dec_type = (decoder_ == SnnDecoder::FIRST_SPIKE)
        ? RLDecoder::DecodingType::FIRST_SPIKE
        : RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    double total_shaped   = 0.0;
    double total_original = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (resetBetweenSteps_) net.fastReset();
        std::vector<double> output_spikes;

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            // position ∈ [POS_MIN, POS_MAX] → [0, 1]
            // velocity ∈ [-VEL_MAX, VEL_MAX] → [0, 1]
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;  // bias
            if (nInput_ >= 1)
                norm[1] = (rlt::get(obs_mat, 0, 0) - POS_MIN) / (POS_MAX - POS_MIN);
            if (nInput_ >= 2)
                norm[2] = (rlt::get(obs_mat, 0, 1) + VEL_MAX) / (2.0 * VEL_MAX);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                if (net.getOutputSpikes()[0])
                    output_spikes.push_back(static_cast<double>(t));
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
                const double all_obs[2] = {
                    rlt::get(obs_mat, 0, 0),  // position
                    rlt::get(obs_mat, 0, 1),  // velocity
                };
                std::vector<double> obs_vals;
                for (int i = 0; i < std::min(n_obs, 2); ++i)
                    obs_vals.push_back(all_obs[i]);
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    const std::vector<std::pair<double,double>> all_limits = {
                        {POS_MIN, POS_MAX}, {-VEL_MAX, VEL_MAX},
                    };
                    enc_currents = rl_enc.encodeObservationLarge(
                        obs_vals, std::vector<std::pair<double,double>>(
                            all_limits.begin(), all_limits.begin() + obs_vals.size()));
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: map to [0, 20] mA
                if (nInput_ >= 1)
                    currents[1] = (rlt::get(obs_mat, 0, 0) - POS_MIN) / (POS_MAX - POS_MIN) * 20.0;
                if (nInput_ >= 2)
                    currents[2] = (rlt::get(obs_mat, 0, 1) + VEL_MAX) / (2.0 * VEL_MAX) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                if (net.getOutputSpikes()[0])
                    output_spikes.push_back(static_cast<double>(t));
            }
        }

        double action;
        if (use_rldecoder) {
            double val = rl_decoder.decodeContinuousAction(output_spikes);
            action = val * 2.0 - 1.0;   // [0,1] → [-1,1]
        } else {
            action = (static_cast<double>(output_spikes.size()) / max_spikes - 1.0);
        }
        action = std::clamp(action, -MAX_ACTION, MAX_ACTION);

        rlt::set(action_mat, 0, 0, action);
        rlt::step(device, env, params, state, action_mat, next_state, rng);
        double step_r = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        total_original += step_r;
        total_shaped   += step_r;

        if (shapingScale_ != 0.0)
            total_shaped += shapingScale_ * (std::sin(3.0 * next_state.position)
                                           - std::sin(3.0 * state.position));

        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return {total_shaped, total_original};
}

std::vector<double> SnnMountainCarTask::evaluate(const Ind& ind, int seed)
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

std::vector<double> SnnMountainCarTask::getDistFitness(
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

std::pair<std::vector<double>,std::vector<double>> SnnMountainCarTask::evalEpisodes(
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

void SnnMountainCarTask::exportTrajectory(const std::vector<double>& wVec,
                                          const std::vector<int>&    aVec,
                                          int bestWi, int evalSeed,
                                          const std::string& outFile,
                                          bool directSeed) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,position,velocity,action,reward\n";

    Network net = buildNetwork(wVec, aVec);

    int episodeSeed;
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
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mountain_car::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    constexpr double POS_MIN = -1.2,  POS_MAX =  0.6;
    constexpr double VEL_MAX =  0.07;

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    const bool use_rldecoder = (decoder_ != SnnDecoder::SPIKE_COUNT);
    const RLDecoder::DecodingType dec_type = (decoder_ == SnnDecoder::FIRST_SPIKE)
        ? RLDecoder::DecodingType::FIRST_SPIKE
        : RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        double pos = rlt::get(obs_mat, 0, 0);
        double vel = rlt::get(obs_mat, 0, 1);

        if (resetBetweenSteps_) net.fastReset();
        std::vector<double> output_spikes;

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (pos - POS_MIN) / (POS_MAX - POS_MIN);
            if (nInput_ >= 2) norm[2] = (vel + VEL_MAX) / (2.0 * VEL_MAX);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }
            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(weight);
                if (net.getOutputSpikes()[0])
                    output_spikes.push_back(static_cast<double>(t));
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
                const double all_obs[2] = { pos, vel };
                std::vector<double> obs_vals;
                for (int i = 0; i < std::min(n_obs, 2); ++i)
                    obs_vals.push_back(all_obs[i]);
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    const std::vector<std::pair<double,double>> all_limits = {
                        {POS_MIN, POS_MAX}, {-VEL_MAX, VEL_MAX},
                    };
                    enc_currents = rl_enc.encodeObservationLarge(
                        obs_vals, std::vector<std::pair<double,double>>(
                            all_limits.begin(), all_limits.begin() + obs_vals.size()));
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: map to [0, 20] mA
                if (nInput_ >= 1) currents[1] = (pos - POS_MIN) / (POS_MAX - POS_MIN) * 20.0;
                if (nInput_ >= 2) currents[2] = (vel + VEL_MAX) / (2.0 * VEL_MAX) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(weight);
                if (net.getOutputSpikes()[0])
                    output_spikes.push_back(static_cast<double>(t));
            }
        }

        double action;
        if (use_rldecoder) {
            action = rl_decoder.decodeContinuousAction(output_spikes) * 2.0 - 1.0;
        } else {
            action = static_cast<double>(output_spikes.size()) / max_spikes - 1.0;
        }
        action = std::clamp(action, -MAX_ACTION, MAX_ACTION);

        rlt::set(action_mat, 0, 0, action);
        rlt::step(device, env, params, state, action_mat, next_state, rng);
        double reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        if (shapingScale_ != 0.0)
            reward += shapingScale_ * (std::sin(3.0 * next_state.position)
                                     - std::sin(3.0 * state.position));
        state = next_state;

        csv << step   << ','
            << pos    << ',' << vel    << ','
            << action << ',' << reward << '\n';

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

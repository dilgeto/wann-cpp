#include "../include/wann/SnnPendulumTask.h"

// SNN simulator
#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>
#include <memory>

// rl-tools pendulum
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>

#include <algorithm>
#include <cmath>

namespace rlt = rl_tools;

// rl-tools type aliases (matching pendulum_eval.cpp)
using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using PendSpec  = rlt::rl::environments::pendulum::Specification<T, TI>;
using Env       = rlt::rl::environments::Pendulum<PendSpec>;
using RNG       = typename rlt::devices::random::CPU::ENGINE<>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 3, false>>;
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

const double SnnPendulumTask::WEIGHT_VALS[N_WEIGHTS] = {
    1.0, 2.0, 5.0, 10.0
};

SnnPendulumTask::SnnPendulumTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnPendulumTask::wannActToNeuronType(int actId) {
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

// -------------------------------------------------------------------------
// buildNetwork (from raw genome)
//
// Builds the SNN Network directly from the WANN genome, preserving the
// excitatory/inhibitory polarity stored in each ConnGene.
//
// Node type mapping:
//   type 4 (bias)   → addInputNeuron   (driven by constant BIAS_CURRENT)
//   type 1 (input)  → addInputNeuron
//   type 3 (hidden) → addHiddenNeuron
//   type 2 (output) → addOutputNeuron
//
// Only enabled connections are added as synapses.
// The insertion order of input neurons must match the currents vector in
// runEpisode: bias first, then inputs 1..nInput_ in NodeGene order.
// -------------------------------------------------------------------------
Network SnnPendulumTask::buildNetwork(const Ind& ind) const
{
    Network net(1.0 /*dt=1ms*/, true /*allow_recurrent*/);

    // Map node gene ID → SNN neuron ID
    std::unordered_map<int,int> snn_id;
    snn_id.reserve(ind.nodes.size());

    for (const auto& ng : ind.nodes) {
        NeuronType nt = wannActToNeuronType(ng.activation);
        int sid;
        switch (ng.type) {
            case 4: sid = net.addInputNeuron(nt);  break;  // bias
            case 1: sid = net.addInputNeuron(nt);  break;  // observation input
            case 3: sid = net.addHiddenNeuron(nt); break;  // hidden
            case 2: sid = net.addOutputNeuron(nt); break;  // action output
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

// -------------------------------------------------------------------------
// buildNetwork (from wVec/aVec – ITask fallback, all synapses excitatory)
// -------------------------------------------------------------------------
Network SnnPendulumTask::buildNetwork(const std::vector<double>& wVec,
                                      const std::vector<int>&    aVec) const
{
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));

    Network net(1.0 /*dt=1ms*/, true /*allow_recurrent*/);

    std::vector<int> snn_id(N, -1);
    snn_id[0] = net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // bias
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
                net.addSynapse(snn_id[i], snn_id[j], true);
        }
    }

    return net;
}

double SnnPendulumTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::pendulum::ObservationFourier<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    constexpr double MAX_THETA_DOT = 8.0;
    const int        window_steps  = static_cast<int>(SIM_WINDOW_MS);
    const int        n_channels    = nInput_ + 1;  // bias + observations

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    // --- Decoder setup ---
    const bool use_rldecoder = (decoder_ != SnnDecoder::SPIKE_COUNT);
    const RLDecoder::DecodingType dec_type = (decoder_ == SnnDecoder::FIRST_SPIKE)
        ? RLDecoder::DecodingType::FIRST_SPIKE
        : RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    double total_reward = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (resetBetweenSteps_) net.fastReset();
        std::vector<double> output_spikes;

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (rlt::get(obs_mat, 0, 0) + 1.0) * 0.5;
            if (nInput_ >= 2) norm[2] = (rlt::get(obs_mat, 0, 1) + 1.0) * 0.5;
            if (nInput_ >= 3) norm[3] = (rlt::get(obs_mat, 0, 2) / MAX_THETA_DOT + 1.0) * 0.5;

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
                std::vector<double> obs_vals;
                for (int i = 0; i < std::min(n_obs, 3); ++i)
                    obs_vals.push_back(rlt::get(obs_mat, 0, i));
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    const std::vector<std::pair<double,double>> all_limits = {
                        {-1.0, 1.0}, {-1.0, 1.0}, {-MAX_THETA_DOT, MAX_THETA_DOT},
                    };
                    enc_currents = rl_enc.encodeObservationLarge(
                        obs_vals, std::vector<std::pair<double,double>>(
                            all_limits.begin(), all_limits.begin() + obs_vals.size()));
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: map observations to [0, 20] mA
                if (nInput_ >= 1) currents[1] = (rlt::get(obs_mat, 0, 0) + 1.0) * 10.0;
                if (nInput_ >= 2) currents[2] = (rlt::get(obs_mat, 0, 1) + 1.0) * 10.0;
                if (nInput_ >= 3) currents[3] = (rlt::get(obs_mat, 0, 2) / MAX_THETA_DOT + 1.0) * 10.0;
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
            action = (val * 2.0 - 1.0) * MAX_TORQUE;
        } else {
            action = (static_cast<double>(output_spikes.size()) / max_spikes - 1.0) * MAX_TORQUE;
        }
        action = std::clamp(action, -MAX_TORQUE, MAX_TORQUE);

        rlt::set(action_mat, 0, 0, action);
        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;
    }

    return total_reward;
}

// -------------------------------------------------------------------------
// evaluate – preferred entry point.  Uses the raw genome so that
// ConnGene.excitatory is correctly reflected as synapse polarity.
// Thread-safe: all objects are function-local.
// -------------------------------------------------------------------------
std::vector<double> SnnPendulumTask::evaluate(const Ind& ind, int seed)
{
    Network net = buildNetwork(ind);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            int episodeSeed = (seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed);
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

// -------------------------------------------------------------------------
// getDistFitness – ITask fallback.  Builds without polarity info (all
// synapses excitatory).  Provided for interface compatibility only.
// -------------------------------------------------------------------------
std::vector<double> SnnPendulumTask::getDistFitness(
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
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed);
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

} // namespace wann

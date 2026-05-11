#include "../include/wann/SnnAntTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace rlt = rl_tools;

using T       = double;
using TI      = size_t;
using DEVICE  = rlt::devices::DefaultCPU;
using RNG     = typename rlt::devices::random::CPU::ENGINE<>;

using AntSpec = rlt::rl::environments::mujoco::ant::Specification<
                    T, TI,
                    rlt::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
using Env     = rlt::rl::environments::mujoco::Ant<AntSpec>;

// Observation: 27 dims = q[2..14] (13) + q_dot[0..13] (14)
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 27, false>>;
// Action: 8 dims, joint torques ∈ [-1,1]
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1,  8, false>>;

namespace wann {

const double SnnAntTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 1.5, 2.0
};

SnnAntTask::SnnAntTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnAntTask::wannActToNeuronType(int actId) {
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

Network SnnAntTask::buildNetwork(const Ind& ind) const
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

Network SnnAntTask::buildNetwork(const std::vector<double>& wVec,
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
                net.addSynapse(snn_id[i], snn_id[j], true);
        }
    }

    return net;
}

double SnnAntTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    DEVICE device;
    Env env;
    rlt::malloc(device, env);

    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mujoco::ant::Observation<AntSpec> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;  // bias + observations

    constexpr double MAX_RATE   = 100.0;
    constexpr double REF_PERIOD = 2.0;
    constexpr double DT         = 1.0;
    std::mt19937 enc_rng(static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const bool use_rldecoder = (decoder_ != SnnDecoder::SPIKE_COUNT);
    const RLDecoder::DecodingType dec_type = (decoder_ == SnnDecoder::FIRST_SPIKE)
        ? RLDecoder::DecodingType::FIRST_SPIKE
        : RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    // Normalize one observation dimension to [0, 1] given its expected range.
    auto normObs = [](double x, double lo, double hi) {
        return std::clamp((x - lo) / (hi - lo), 0.0, 1.0);
    };

    double total_reward = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (resetBetweenSteps_) net.fastReset();

        // Normalize observation (27 dims) to [0, 1]:
        //   obs[0]    = q[2]  : torso z-height, range [0.0, 1.5]
        //   obs[1..4] = q[3..6]: quaternion, range [-1, 1]
        //   obs[5..12]= q[7..14]: joint angles, range [-π, π]
        //   obs[13..18]= q_dot[0..5]: linear+angular velocity, range [-10, 10]
        //   obs[19..26]= q_dot[6..13]: joint velocities, range [-20, 20]
        std::vector<double> norm(n_channels, 0.0);
        norm[0] = 1.0;  // bias
        if (nInput_ >= 1)
            norm[1] = normObs(rlt::get(obs_mat, 0, 0), 0.0, 1.5);
        for (int i = 1; i <= 4 && i + 1 < n_channels; ++i)
            norm[i + 1] = normObs(rlt::get(obs_mat, 0, i), -1.0, 1.0);
        for (int i = 5; i <= 12 && i + 1 < n_channels; ++i)
            norm[i + 1] = normObs(rlt::get(obs_mat, 0, i), -M_PI, M_PI);
        for (int i = 13; i <= 18 && i + 1 < n_channels; ++i)
            norm[i + 1] = normObs(rlt::get(obs_mat, 0, i), -10.0, 10.0);
        for (int i = 19; i <= 26 && i + 1 < n_channels; ++i)
            norm[i + 1] = normObs(rlt::get(obs_mat, 0, i), -20.0, 20.0);

        // Per-output spike time lists (one per action dimension).
        std::vector<std::vector<double>> out_spikes(nOutput_);

        if (encoder_ == SnnEncoder::POISSON) {
            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double rate    = std::clamp(norm[ch], 0.0, 1.0) * MAX_RATE;
                double sp_prob = (rate / 1000.0) * DT;
                double last_t  = -REF_PERIOD - 1.0;
                for (int t = 0; t < window_steps; ++t) {
                    double ct = static_cast<double>(t);
                    if ((ct - last_t) >= REF_PERIOD && udist(enc_rng) < sp_prob) {
                        spike_trains[ch].push_back(ct);
                        last_t = ct;
                    }
                }
            }
            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                const auto& fired = net.getOutputSpikes();
                for (int o = 0; o < nOutput_; ++o)
                    if (fired[o]) out_spikes[o].push_back(static_cast<double>(t));
            }
        } else {
            // CURRENT encoder: map [0,1] → [0, 20] mA
            std::vector<double> currents(n_channels);
            currents[0] = BIAS_CURRENT;
            for (int ch = 1; ch < n_channels; ++ch)
                currents[ch] = norm[ch] * 20.0;

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                const auto& fired = net.getOutputSpikes();
                for (int o = 0; o < nOutput_; ++o)
                    if (fired[o]) out_spikes[o].push_back(static_cast<double>(t));
            }
        }

        // Decode each output neuron independently → action ∈ [-1, 1]
        for (int o = 0; o < nOutput_; ++o) {
            double action;
            if (use_rldecoder) {
                action = rl_decoder.decodeContinuousAction(out_spikes[o]) * 2.0 - 1.0;
            } else {
                action = static_cast<double>(out_spikes[o].size()) / max_spikes - 1.0;
            }
            rlt::set(action_mat, 0, o, std::clamp(action, -1.0, 1.0));
        }

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    rlt::free(device, env);
    return total_reward;
}

std::vector<double> SnnAntTask::evaluate(const Ind& ind, int seed)
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

std::vector<double> SnnAntTask::getDistFitness(
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

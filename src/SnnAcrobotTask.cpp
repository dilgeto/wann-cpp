#include "../include/wann/SnnAcrobotTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace rlt = rl_tools;

using T         = double;
using TI        = size_t;
using DEVICE    = rlt::devices::DefaultCPU;
using AcrobotSpec = rlt::rl::environments::acrobot::Specification<T, TI>;
using Env       = rlt::rl::environments::Acrobot<AcrobotSpec>;
using RNG       = typename rlt::devices::random::CPU::ENGINE<>;
// 6-dimensional observation: cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ₁_dot, θ₂_dot
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 6, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;

namespace wann {

const double SnnAcrobotTask::WEIGHT_VALS[N_WEIGHTS] = {
    1.0, 2.0, 5.0, 10.0, 15.0, 20.0
};

SnnAcrobotTask::SnnAcrobotTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
{}

NeuronType SnnAcrobotTask::wannActToNeuronType(int actId) {
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
// buildNetwork (from raw genome – preserves excitatory/inhibitory polarity)
// -------------------------------------------------------------------------
Network SnnAcrobotTask::buildNetwork(const Ind& ind) const
{
    Network net(1.0, true);

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
Network SnnAcrobotTask::buildNetwork(const std::vector<double>& wVec,
                                     const std::vector<int>&    aVec) const
{
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));

    Network net(1.0, true);

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

double SnnAcrobotTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::acrobot::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    constexpr double MAX_VEL_1  = 4.0 * M_PI;
    constexpr double MAX_VEL_2  = 9.0 * M_PI;
    const int        window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int        n_channels   = nInput_ + 1;  // bias + observations

    // --- Poisson encoder setup (only used when encoder_ == POISSON) ---
    constexpr double MAX_RATE   = 100.0;  // Hz
    constexpr double REF_PERIOD = 2.0;   // ms
    constexpr double DT         = 1.0;   // ms
    std::mt19937 enc_rng(static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    // --- Decoder setup ---
    // RLDecoder is used for RATE and FIRST_SPIKE; SPIKE_COUNT uses raw count.
    const bool use_rldecoder = (decoder_ != SnnDecoder::SPIKE_COUNT);
    const RLDecoder::DecodingType dec_type = (decoder_ == SnnDecoder::FIRST_SPIKE)
        ? RLDecoder::DecodingType::FIRST_SPIKE
        : RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    double total_reward = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        net.fastReset();
        std::vector<double> output_spikes;

        if (encoder_ == SnnEncoder::POISSON) {
            // Normalize to [0,1]
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (rlt::get(obs_mat, 0, 0) + 1.0) * 0.5;
            if (nInput_ >= 2) norm[2] = (rlt::get(obs_mat, 0, 1) + 1.0) * 0.5;
            if (nInput_ >= 3) norm[3] = (rlt::get(obs_mat, 0, 2) + 1.0) * 0.5;
            if (nInput_ >= 4) norm[4] = (rlt::get(obs_mat, 0, 3) + 1.0) * 0.5;
            if (nInput_ >= 5) norm[5] = (rlt::get(obs_mat, 0, 4) / MAX_VEL_1 + 1.0) * 0.5;
            if (nInput_ >= 6) norm[6] = (rlt::get(obs_mat, 0, 5) / MAX_VEL_2 + 1.0) * 0.5;

            // Poisson spike trains (same algorithm as PoissonEncoder::encode)
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
                if (net.getOutputSpikes()[0])
                    output_spikes.push_back(static_cast<double>(t));
            }
        } else {
            // CURRENT: map observations to [0, 20] mA and inject directly
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;
            if (nInput_ >= 1) currents[1] = (rlt::get(obs_mat, 0, 0) + 1.0) * 10.0;
            if (nInput_ >= 2) currents[2] = (rlt::get(obs_mat, 0, 1) + 1.0) * 10.0;
            if (nInput_ >= 3) currents[3] = (rlt::get(obs_mat, 0, 2) + 1.0) * 10.0;
            if (nInput_ >= 4) currents[4] = (rlt::get(obs_mat, 0, 3) + 1.0) * 10.0;
            if (nInput_ >= 5) currents[5] = (rlt::get(obs_mat, 0, 4) / MAX_VEL_1 + 1.0) * 10.0;
            if (nInput_ >= 6) currents[6] = (rlt::get(obs_mat, 0, 5) / MAX_VEL_2 + 1.0) * 10.0;

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

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return total_reward;
}

// -------------------------------------------------------------------------
// evaluate – preferred entry point (full genome, with polarity)
// -------------------------------------------------------------------------
std::vector<double> SnnAcrobotTask::evaluate(const Ind& ind, int seed)
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
// getDistFitness – ITask fallback (all synapses excitatory)
// -------------------------------------------------------------------------
std::vector<double> SnnAcrobotTask::getDistFitness(
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

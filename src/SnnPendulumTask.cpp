#include "../include/wann/SnnPendulumTask.h"

// SNN simulator
#include <core/simulator.hpp>

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

namespace wann {

const double SnnPendulumTask::WEIGHT_VALS[N_WEIGHTS] = {
    1.0, 2.0, 5.0, 10.0, 15.0, 20.0
};

SnnPendulumTask::SnnPendulumTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
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

// -------------------------------------------------------------------------
// runEpisode
//
// Runs one Pendulum episode using the SNN as the controller.
// For each env step:
//   1. Encode observation as input currents.
//   2. Simulate SNN for SIM_WINDOW_MS steps (dt=1ms).
//   3. Decode output spike rate → torque action.
//   4. Step the environment and accumulate reward.
//
// The bias neuron (snn input index 0) receives BIAS_CURRENT at every SNN
// step so it fires continuously, replicating WANN's bias node value of 1.0.
// -------------------------------------------------------------------------
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

    const int    window_steps  = static_cast<int>(SIM_WINDOW_MS); // dt=1ms
    // Rough max spikes per window (Izhikevich REGULAR_SPIKING can fire ~1 spike per 3-5 ms
    // when strongly driven; window_steps/2 is a conservative upper bound)
    const double max_spikes    = static_cast<double>(window_steps) / 2.0;

    // Input current vector: [bias_current, obs[0..nInput_-1]]
    std::vector<double> currents(nInput_ + 1, 0.0);
    currents[0] = BIAS_CURRENT;

    double total_reward = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        // --- Observe ---
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        // Map Pendulum observation to SNN input currents [0, 20] mA
        //   cos(θ)  ∈ [-1, 1]  →  [0, 20]
        //   sin(θ)  ∈ [-1, 1]  →  [0, 20]
        //   θ̇      ∈ [-8, 8]  →  [0, 20]
        if (nInput_ >= 1)
            currents[1] = (rlt::get(obs_mat, 0, 0) + 1.0) * 10.0;
        if (nInput_ >= 2)
            currents[2] = (rlt::get(obs_mat, 0, 1) + 1.0) * 10.0;
        if (nInput_ >= 3)
            currents[3] = (rlt::get(obs_mat, 0, 2) / 8.0 + 1.0) * 10.0;

        // --- Simulate SNN ---
        net.fastReset();
        int spike_count = 0;
        for (int t = 0; t < window_steps; ++t) {
            // Inject bias + observation currents at every step so the bias neuron
            // keeps firing and input currents are sustained (conductance decays with tau_exc=5ms)
            net.setInputCurrents(currents);
            net.step(sharedWeight);
            if (net.getOutputSpikes()[0]) ++spike_count;
        }

        // --- Decode spikes → torque ---
        // Map [0, max_spikes] → [-MAX_TORQUE, +MAX_TORQUE]
        double action = (spike_count / max_spikes - 1.0) * MAX_TORQUE;
        action = std::clamp(action, -MAX_TORQUE, MAX_TORQUE);

        // --- Step environment ---
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

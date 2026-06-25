#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>   // Network, NeuronType

#include <vector>

namespace wann {

// ITask implementation connecting a WANN genome to the SNN simulator and the
// rl-tools Pendulum-v1 environment.
//
// WANN genome layout after express():
//   wVec index 0        = bias node (injected BIAS_CURRENT so it fires every step)
//   indices 1..nInput   = observation inputs  (cos θ, sin θ, θ̇)
//   indices nInput+1..  = hidden neurons (if any)
//   last nOutput        = output neuron (torque ∈ [-2, 2])
//
// Shared weight values are positive SNN conductances; 6 values are tested to
// match the default alg_nVals=6 in the hyperparameters.
class SnnPendulumTask : public ITask {
public:
    static constexpr int    N_WEIGHTS       = 4;
    // Positive SNN conductance values (replaces WANN's standard {-2..2} range)
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT    = 50.0;  // mA — keeps bias neuron spiking every step
    static constexpr double SIM_WINDOW_MS   = 20.0;  // SNN sim duration per env step (ms, dt=1ms)
    static constexpr double MAX_TORQUE      = 2.0;   // pendulum action range

    explicit SnnPendulumTask(const Hyperparams& hyp);

    // Full evaluation using the raw genome (preserves excitatory/inhibitory polarity).
    // This is the preferred entry point from main_snn.cpp.
    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    // ITask compatibility: builds without polarity info (all synapses excitatory).
    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

private:
    int        nInput_;
    int        nOutput_;
    int        nReps_;
    int        neuronsPerVar_;
    SnnEncoder encoder_;
    SnnDecoder decoder_;
    bool       resetBetweenSteps_;

    static NeuronType wannActToNeuronType(int actId);

    // Build an SNN Network directly from the WANN genome.
    // Uses ConnGene.excitatory to set each synapse's polarity.
    Network buildNetwork(const Ind& ind) const;

    // Build without polarity (ITask fallback — all synapses excitatory).
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    // Run one rl-tools Pendulum episode with the given shared weight.
    // Returns the total undiscounted reward.
    double runEpisode(Network& net, double sharedWeight, int episodeSeed) const;
};

} // namespace wann

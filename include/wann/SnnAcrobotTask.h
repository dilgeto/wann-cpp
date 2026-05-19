#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <vector>

namespace wann {

// ITask implementation connecting a WANN genome to the SNN simulator and the
// rl-tools Acrobot environment.
//
// Observation (6 dims):
//   cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ₁_dot, θ₂_dot
//
// Action (1 dim): continuous torque ∈ [-1, 1] N·m (same scale as Gymnasium Acrobot-v1)
//
// Episode ends at step 500 or when rlt::terminated() is true (tip above pivot).
//
// WANN genome layout after express():
//   index 0          = bias node (constant BIAS_CURRENT each SNN step)
//   indices 1..6     = observation inputs
//   indices 7..N-2   = hidden neurons (if any)
//   index  N-1       = output neuron (torque)
class SnnAcrobotTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;   // mA
    static constexpr double SIM_WINDOW_MS = 20.0;   // SNN sim duration per env step
    static constexpr double MAX_TORQUE    = 1.0;    // Acrobot torque range ±1 N·m (Gymnasium scale)

    explicit SnnAcrobotTask(const Hyperparams& hyp);

    // Preferred entry point: uses ConnGene.excitatory for synapse polarity.
    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    // ITask fallback: all synapses excitatory.
    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Run nEpisodes with the given shared weight; returns per-episode rewards.
    std::vector<double> evalEpisodes(const std::vector<double>& wVec,
                                     const std::vector<int>&    aVec,
                                     double weight, int nEpisodes,
                                     int baseSeed) const;

    // Columns: step,cos_th1,sin_th1,cos_th2,sin_th2,dth1,dth2,action,reward
    // bestWi: index into WEIGHT_VALS used for the logged episode.
    // evalSeed: the seed passed to evaluate() for this individual (not the episode seed).
    void exportTrajectory(const std::vector<double>& wVec,
                          const std::vector<int>&    aVec,
                          int bestWi, int evalSeed,
                          const std::string& outFile) const;

private:
    int        nInput_;
    int        nOutput_;
    int        nReps_;
    SnnEncoder encoder_;
    SnnDecoder decoder_;
    double     shapingScale_;
    bool       resetBetweenSteps_;

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const Ind& ind) const;
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    double runEpisode(Network& net, double sharedWeight, int episodeSeed) const;
};

} // namespace wann

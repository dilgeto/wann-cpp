#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <vector>

namespace wann {

// ITask implementation: WANN + SNN simulator + rl-tools MountainCar discrete
// (matches Gymnasium MountainCar-v0 exactly).
//
// Observation (2 dims):
//   position ∈ [-1.2, 0.6]
//   velocity ∈ [-0.07, 0.07]
//
// Action (3 discrete): 0 = push left, 1 = no push, 2 = push right
//   Decoded via winner-takes-all on 3 output neurons.
//
// Physics: force = (winner - 1) * power,  power = 0.001  (Gymnasium discrete)
// Reward:  -1 per step (no action cost, no goal bonus)
// Terminal: position >= 0.5 && velocity >= 0, or step 200
class SnnDiscMCTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;
    static constexpr double SIM_WINDOW_MS = 20.0;
    static constexpr int    N_ACTIONS     = 3;

    explicit SnnDiscMCTask(const Hyperparams& hyp);

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Columns: step,position,velocity,action,reward
    // bestWi: index into WEIGHT_VALS for the logged episode.
    // evalSeed: the seed passed to evaluate() for this individual.
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
    bool       resetBetweenSteps_;
    double     rewardShapingScale_;

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const Ind& ind) const;
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    double runEpisode(Network& net, double sharedWeight, int episodeSeed) const;
};

} // namespace wann

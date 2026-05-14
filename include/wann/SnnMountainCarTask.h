#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <vector>

namespace wann {

// ITask implementation: WANN + SNN simulator + rl-tools MountainCarContinuous-v0.
//
// Observation (2 dims):
//   position ∈ [-1.2, 0.6]
//   velocity ∈ [-0.07, 0.07]
//
// Action (1 dim): force ∈ [-1, 1]
//
// Episode ends at step 999 or when position >= 0.45 && velocity >= 0.
// Reward: -0.1 * action² each step, +100 on reaching the goal.
class SnnMountainCarTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;   // mA
    static constexpr double SIM_WINDOW_MS = 20.0;   // SNN sim duration per env step
    static constexpr double MAX_ACTION    = 1.0;    // force range ±1

    explicit SnnMountainCarTask(const Hyperparams& hyp);

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Columns: step,position,velocity,action,reward
    void exportTrajectory(const std::vector<double>& wVec,
                          const std::vector<int>&    aVec,
                          double weight, int seed,
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

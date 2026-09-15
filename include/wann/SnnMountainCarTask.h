#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <utility>
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

    // Build the network topology for one individual. Independent of weight
    // value (the shared scalar is applied at simulation time in
    // Network::step(), not baked into synapses here), so callers can build
    // this once per individual and reuse (copy) it across weight values
    // instead of re-parsing the genome for each one.
    Network buildNetwork(const Ind& ind) const;

    // Evaluate a single (individual, weight-value) pair from a pre-built
    // topology — the finer-grained unit evalPop() dispatches on so idle
    // threads always have work near the end of a generation, instead of
    // only per full individual. Copies templateNet internally so concurrent
    // calls sharing the same template are safe.
    double evaluateWeight(const Network& templateNet, int wi, int seed = -1) const;

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Run nEpisodes with the given shared weight.
    // Returns {shaped_rewards, original_rewards}; shaped includes the potential-based bonus.
    std::pair<std::vector<double>,std::vector<double>>
    evalEpisodes(const std::vector<double>& wVec,
                 const std::vector<int>&    aVec,
                 double weight, int nEpisodes,
                 int baseSeed) const;

    // Columns: step,position,velocity,action,reward
    // bestWi: index into WEIGHT_VALS used for the logged episode.
    // evalSeed: training evaluate() seed (directSeed=false) or direct episode seed (directSeed=true).
    void exportTrajectory(const std::vector<double>& wVec,
                          const std::vector<int>&    aVec,
                          int bestWi, int evalSeed,
                          const std::string& outFile,
                          bool directSeed = false) const;

private:
    int        nInput_;
    int        nOutput_;
    int        nReps_;
    int        neuronsPerVar_;
    SnnEncoder encoder_;
    SnnDecoder decoder_;
    double     shapingScale_;
    bool       resetBetweenSteps_;

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    std::pair<double,double> runEpisode(Network& net, double sharedWeight, int episodeSeed) const;
};

} // namespace wann

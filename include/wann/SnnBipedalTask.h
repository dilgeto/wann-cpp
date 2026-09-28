#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace wann {

// ITask implementation: WANN + SNN simulator + BipedalWalker(Hardcore)-v3,
// the Box2D-free C++ port living in snn-simulator
// (include/rl/environments/bipedal_walker/). Hardcore by default; plain
// BipedalWalker with bipedal_hardcore=false.
//
// Observation (24 dims, Gymnasium order):
//   0..3   hull angle, angular velocity, vx, vy (already scaled by the env)
//   4..8   leg 1: hip angle, hip speed, knee angle+1, knee speed, contact
//   9..13  leg 2: same
//   14..23 lidar, 10 rays, in [0,1]
//
// Action (4 dims): hip/knee torques of each leg ∈ [-1, 1].
//
// Reward: Gymnasium's (130*dx/SCALE - 5*|Δhull angle| - torque cost; -100
// on fall or leaving the terrain on the left). Terrain is resampled every
// episode from the episode seed, so fitness measures generalisation across
// terrains, not memorising one course.
class SnnBipedalTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;   // mA
    static constexpr int    N_OBS         = 24;
    static constexpr int    N_ACTIONS     = 4;

    explicit SnnBipedalTask(const Hyperparams& hyp);

    // Build the network topology for one individual. Independent of weight
    // value (the shared scalar is applied at simulation time in
    // Network::step(), not baked into synapses here), so callers can build
    // this once per individual and reuse (copy) it across weight values.
    Network buildNetwork(const Ind& ind) const;

    // Evaluate a single (individual, weight-value) pair from a pre-built
    // topology — the unit evalPop() dispatches on. Copies templateNet
    // internally so concurrent calls sharing the same template are safe.
    double evaluateWeight(const Network& templateNet, int wi, int seed = -1) const;

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Run nEpisodes with the given shared weight.
    // Returns {shaped_rewards, original_rewards}; shaped == original (no shaping).
    std::pair<std::vector<double>,std::vector<double>>
    evalEpisodes(const std::vector<double>& wVec,
                 const std::vector<int>&    aVec,
                 double weight, int nEpisodes,
                 int baseSeed) const;

    // Run one episode and write trajectory CSV.
    // Columns: step,hull_x,obs0..obs23,a0..a3,reward
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
    bool       resetBetweenSteps_;
    double     windowMs_;
    double     tauExc_;
    double     tauInh_;
    double     ttfsThreshold_;
    bool       hardcore_;
    int        maxSteps_;       // 0 = environment's own EPISODE_STEP_LIMIT

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    // Dispatches on hardcore_. csv != nullptr also logs every step to it.
    std::pair<double,double> runEpisode(Network& net, double sharedWeight,
                                        long long episodeSeed,
                                        std::ostream* csv = nullptr) const;

    // Defined (and only instantiated) in SnnBipedalTask.cpp.
    template <typename ENV>
    std::pair<double,double> runEpisodeImpl(Network& net, double sharedWeight,
                                            long long episodeSeed,
                                            std::ostream* csv) const;
};

} // namespace wann

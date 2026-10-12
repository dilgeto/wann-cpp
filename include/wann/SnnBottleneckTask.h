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

// ITask implementation: WANN + SNN simulator + rl-tools multi-agent
// Bottleneck (rl_tools::rl::environments::multi_agent::Bottleneck). Uses the
// same Parameters as rl-tools' own benchmark config for this environment
// (rl::zoo::bottleneck_v0::ENVIRONMENT_FACTORY in
// src/rl/zoo/bottleneck-v0/environment.h, trained there with PPO), not the
// bare DefaultParameters: N_AGENTS=5, LIDAR_FOV=90° (vs 20°),
// BOTTLENECK_WIDTH=5 (vs 2), AGENT_MAX_SPEED=4/AGENT_MAX_ANGULAR_VELOCITY=4
// (vs 2/2), AGENT_MAX_ACCELERATION=20/AGENT_MAX_ANGULAR_ACCELERATION=20 (vs
// 2/2) — see BenchmarkParameters in SnnBottleneckTask.cpp. LIDAR_RESOLUTION=5,
// 20x20 arena and EPISODE_STEP_LIMIT=200 are unchanged from DefaultParameters
// (the zoo config re-states them but to the same values).
//
// One WANN/SNN topology with shared weights is instantiated as N_AGENTS
// independent copies (parameter sharing, decentralized execution — same
// scheme rl-tools' own PPO config uses for this environment via
// ConfigApproximatorsSequentialMultiAgent) so each agent keeps its own
// membrane state across time steps without cross-contaminating the others
// within a step. Per step: every agent's local observation is encoded and
// run through its own copy of the network to produce its 2 actions; the
// joint action vector is then applied with one rlt::step()/reward() call.
// rlt::reward() already returns one scalar summed over agents (with a
// one-time death penalty and a cross-the-bottleneck bonus), so it plugs
// directly into ITask's single-fitness-per-weight interface.
//
// Per-agent observation (12 dims, PER_AGENT_OBS_DIM = 7 + LIDAR_RESOLUTION):
//   position.x, position.y, orientation, velocity.x, velocity.y,
//   angular_velocity, alive∈{-1,1}, lidar[0..4] (distance; rl-tools' own
//   update_lidar() never actually clamps this to LIDAR_RANGE despite the
//   parameter's name — see LIDAR_MAX_DIST in SnnBottleneckTask.cpp)
//
// Per-agent action (2 dims): linear acceleration, angular acceleration,
// both ∈ [-1, 1].
//
// Episode ends at step 200 (EPISODE_STEP_LIMIT) or when every agent is dead.
class SnnBottleneckTask : public ITask {
public:
    static constexpr int    N_WEIGHTS           = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT        = 50.0;   // mA
    static constexpr int    N_AGENTS            = 5;
    static constexpr int    PER_AGENT_OBS_DIM   = 12;      // 7 + LIDAR_RESOLUTION(5)
    static constexpr int    PER_AGENT_ACTION_DIM = 2;

    explicit SnnBottleneckTask(const Hyperparams& hyp);

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
    // Columns: step, per-agent obs (agent*12..+11), per-agent action (agent*2..+1), reward
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

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    // csv != nullptr also logs every step to it.
    std::pair<double,double> runEpisode(const Network& templateNet, double sharedWeight,
                                        long long episodeSeed,
                                        std::ostream* csv = nullptr) const;
};

} // namespace wann

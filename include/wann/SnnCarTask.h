#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <utility>
#include <vector>

// Overridable at compile time via -DWANN_CAR_SIM_WINDOW_MS=<value> if a
// future experiment needs a different simulation window without editing
// this default.
#ifndef WANN_CAR_SIM_WINDOW_MS
#define WANN_CAR_SIM_WINDOW_MS 40.0
#endif

namespace wann {

// ITask implementation: WANN + SNN simulator + rl-tools Car (CarTrack variant).
//
// Observation (9 dims):
//   x, y           – car position in track frame (m)
//   mu             – heading angle (rad)
//   vx, vy         – longitudinal / lateral velocity (m/s)
//   omega          – yaw rate (rad/s)
//   lidar_L, lidar_C, lidar_R – normalised distance to track boundary
//                    (0 = wall immediately, 1 = full lidar range ≈ 1.25 m)
//
// Action (2 dims):
//   throttle_break ∈ [-1, 1]
//   delta (steering angle) ∈ [-1, 1]
//
// Reward:   vx (forward velocity, dense)
// Terminal: car leaves the track or exceeds EPISODE_STEPS
class SnnCarTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;    // mA
    static constexpr double SIM_WINDOW_MS = WANN_CAR_SIM_WINDOW_MS;  // ms per env step
    static constexpr int    EPISODE_STEPS = 1000;     // max env steps per episode
    static constexpr double VX_MAX        = 3.0;     // m/s – normalisation bound
    static constexpr double VY_MAX        = 2.0;     // m/s
    static constexpr double OMEGA_MAX     = 10.0;    // rad/s

    explicit SnnCarTask(const Hyperparams& hyp);

    void setEpisodeSteps(int n) { episodeSteps_ = n; }

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Run nEpisodes with the given shared weight.
    // Returns {shaped_rewards, original_rewards}; shaped == original (no shaping for Car).
    std::pair<std::vector<double>,std::vector<double>>
    evalEpisodes(const std::vector<double>& wVec,
                 const std::vector<int>&    aVec,
                 double weight, int nEpisodes,
                 int baseSeed) const;

    // Run one episode and write trajectory CSV.
    // Columns: step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,throttle,steering,reward
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
    int        episodeSteps_ = EPISODE_STEPS;
    SnnEncoder encoder_;
    SnnDecoder decoder_;
    bool       resetBetweenSteps_;

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const Ind& ind) const;
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    std::pair<double,double> runEpisode(Network& net, double sharedWeight, long long episodeSeed) const;
};

} // namespace wann

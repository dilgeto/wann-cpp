#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <utility>
#include <vector>

namespace wann {

// ITask implementation connecting a WANN genome to the SNN simulator and the
// rl-tools L2F ("Learning to Fly") Crazyflie quadrotor environment.
//
// Task: track a small Lissajous trajectory centered at the origin
// (rl-tools default: amplitudes 0.5m/1.0m/0m, period 6.5s).
//
// Observation (13 dims):
//   position error (x,y,z) relative to the trajectory target
//   orientation quaternion (w,x,y,z)
//   linear velocity error (x,y,z) relative to the trajectory target
//   angular velocity (x,y,z)
//
// Action (4 dims): normalized rotor commands ∈ [-1, 1] (rl-tools rescales
// internally to the Crazyflie's [0,1] rpm range).
//
// Episode ends at step 500 (5s @ 100Hz) or when rlt::terminated() is true
// (position/velocity error or angular velocity exceeds the MDP thresholds).
class SnnL2FTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 6;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;   // mA
    static constexpr double SIM_WINDOW_MS = 40.0;   // SNN sim duration per env step
    static constexpr double MAX_ACTION    = 1.0;    // rotor command range ±1

    explicit SnnL2FTask(const Hyperparams& hyp);

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

    // Run nEpisodes with the given shared weight.
    // Returns {shaped_rewards, original_rewards}; shaped == original here
    // (L2F's reward is already dense, no potential-based shaping applied).
    std::pair<std::vector<double>,std::vector<double>>
    evalEpisodes(const std::vector<double>& wVec,
                 const std::vector<int>&    aVec,
                 double weight, int nEpisodes,
                 int baseSeed) const;

    // Columns: step,pos_err_x,pos_err_y,pos_err_z,qw,qx,qy,qz,
    //          vel_err_x,vel_err_y,vel_err_z,wx,wy,wz,
    //          rotor0,rotor1,rotor2,rotor3,reward
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

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const Ind& ind) const;
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    std::pair<double,double> runEpisode(Network& net, double sharedWeight, long long episodeSeed) const;
};

} // namespace wann

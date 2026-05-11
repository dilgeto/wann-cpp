#pragma once

#include "Task.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "SnnConfig.h"

#include <core/network.hpp>

#include <vector>

namespace wann {

// ITask implementation: WANN + SNN simulator + rl-tools Ant-v4 (MuJoCo).
//
// Observation (27 dims):
//   q[2]     : torso z-height
//   q[3..6]  : torso quaternion (w,x,y,z)
//   q[7..14] : 8 joint angles
//   q_dot[0..2]  : torso linear velocity (x,y,z)
//   q_dot[3..5]  : torso angular velocity
//   q_dot[6..13] : 8 joint angular velocities
//
// Action (8 dims): joint torques ∈ [-1, 1]
//
// Reward per step: forward_velocity + 1.0 (healthy) - 0.5 * sum(action²)
// Episode ends at step 1000 or when ant falls (z < 0.2 or z > 1.0).
//
// Thread safety: MuJoCo env is created/destroyed per runEpisode call so
// each OpenMP thread operates on its own model+data pair.
class SnnAntTask : public ITask {
public:
    static constexpr int    N_WEIGHTS     = 4;
    static const     double WEIGHT_VALS[N_WEIGHTS];
    static constexpr double BIAS_CURRENT  = 50.0;
    static constexpr double SIM_WINDOW_MS = 20.0;

    explicit SnnAntTask(const Hyperparams& hyp);

    std::vector<double> evaluate(const Ind& ind, int seed = -1);

    std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) override;

    int numWeightVals() const override { return N_WEIGHTS; }

private:
    int        nInput_;
    int        nOutput_;
    int        nReps_;
    SnnEncoder encoder_;
    SnnDecoder decoder_;
    bool       resetBetweenSteps_;

    static NeuronType wannActToNeuronType(int actId);

    Network buildNetwork(const Ind& ind) const;
    Network buildNetwork(const std::vector<double>& wVec,
                         const std::vector<int>&    aVec) const;

    double runEpisode(Network& net, double sharedWeight, int episodeSeed) const;
};

} // namespace wann

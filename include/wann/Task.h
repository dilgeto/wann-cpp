#pragma once
#include <vector>

namespace wann {

// -------------------------------------------------------------------------
// ITask – abstract interface for task evaluation.
//
// Implement this interface to plug in any environment.  The Wann algorithm
// calls it indirectly through the evolution loop in main.cpp.
//
// The weight-agnostic evaluation pattern:
//   1. For each individual, obtain its wVec (flattened weight structure).
//   2. Test the same topology with nVals different shared weight scalars.
//   3. Return a reward matrix reward[nReps][nVals] → mean across reps.
// -------------------------------------------------------------------------
class ITask {
public:
    virtual ~ITask() = default;

    // Evaluate one network topology with a distribution of weight values.
    //
    //   wVec  : flattened N×N weight matrix from Ind::wVec
    //           (non-zero = connection exists, actual value is replaced)
    //   aVec  : activation function index per node
    //   seed  : random seed (-1 = no fixed seed)
    //
    // Returns: fitness for each weight value tested [nVals].
    virtual std::vector<double> getDistFitness(
            const std::vector<double>& wVec,
            const std::vector<int>&    aVec,
            int seed = -1) = 0;

    // Number of weight values that getDistFitness returns.
    virtual int numWeightVals() const = 0;
};

} // namespace wann

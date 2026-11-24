// wann_replay_weights_{acrobot,car,disc_mc} – loads a saved network and exports ONE episode of it (a chosen
// shared-weight index and evaluation seed) as a trajectory CSV. Evaluation only: nothing is trained.
//
// Episode seed = seed*10000 + weight_index*100 (rep 0), i.e. the first episode that wann_eval_weights_<task>
// (and training's getDistFitness) would run for that (seed, weight) pair — so the exported episode is one of
// the episodes behind the revalidated numbers in eval_results/*_best.csv. --nreps only matters for disc_mc,
// whose exportTrajectory replays the earlier episodes to reproduce the SNN state. --rep K (K >= 1) exports the K-th
// episode (0-based) of that (seed, weight) evaluation instead of the first one.
//
// The task is fixed at compile time (EVAL_TASK_*) for the same multiple-definition reason as
// main_eval_weights.cpp.
//
// Usage:
//   ./wann_replay_weights_car -f log/full_p3_<run_key>/rank00_seed00_best.out -d p/car_snn.json \
//       -p screening_full/<run_key>/p3_configs/rank00_seed00.json --weight-index 4 --seed 0 -o out.csv [--nreps 11]

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Task.h"

#if defined(EVAL_TASK_ACROBOT)
    #include "../include/wann/SnnAcrobotTask.h"
    using EvalTask = wann::SnnAcrobotTask;
#elif defined(EVAL_TASK_CAR)
    #include "../include/wann/SnnCarTask.h"
    using EvalTask = wann::SnnCarTask;
#elif defined(EVAL_TASK_DISC_MC)
    #include "../include/wann/SnnDiscMCTask.h"
    using EvalTask = wann::SnnDiscMCTask;
#else
    #error "Define EVAL_TASK_ACROBOT, EVAL_TASK_CAR or EVAL_TASK_DISC_MC"
#endif

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string netFile, configFile, overrideFile, outFile;
    int weightIndex = -1, seed = 0, nReps = -1, repIdx = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f"             && i+1 < argc) netFile      = argv[++i];
        else if (arg == "-d"             && i+1 < argc) configFile   = argv[++i];
        else if (arg == "-p"             && i+1 < argc) overrideFile = argv[++i];
        else if (arg == "-o"             && i+1 < argc) outFile      = argv[++i];
        else if (arg == "--weight-index" && i+1 < argc) weightIndex  = std::stoi(argv[++i]);
        else if (arg == "--seed"         && i+1 < argc) seed         = std::stoi(argv[++i]);
        else if (arg == "--nreps"        && i+1 < argc) nReps        = std::stoi(argv[++i]);
        else if (arg == "--rep"          && i+1 < argc) repIdx       = std::stoi(argv[++i]);
        else {
            std::cerr << "Usage: wann_replay_weights_<task> -f net.out -d config.json [-p overrides.json] "
                         "--weight-index N [--seed S] [--rep K] [--nreps N] -o traj.csv\n";
            return 1;
        }
    }
    if (netFile.empty() || configFile.empty() || outFile.empty() || weightIndex < 0) {
        std::cerr << "-f, -d, -o and --weight-index are required.\n";
        return 1;
    }

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(configFile);
        if (!overrideFile.empty()) wann::updateHyp(hyp, overrideFile);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << '\n';
        return 1;
    }
    if (nReps > 0) hyp.alg_nReps = nReps;

    auto [wVec, aVec, wKey] = wann::importNet(netFile);
    EvalTask task(hyp);
    if (weightIndex >= task.numWeightVals()) {
        std::cerr << "--weight-index out of range (0.." << task.numWeightVals() - 1 << ").\n";
        return 1;
    }
    if (repIdx > 0) {
        // episode k of the evaluation of (seed, weight): episode seed = seed*10000 + weight_index*100 + k, exactly the seed
        // evalEpisodes() gives its k-th episode. Valid when the network state does not carry over between episodes (car and
        // acrobot reset it at the start of every episode; disc_mc configs with snn_reset_between_steps=true).
        task.exportTrajectory(wVec, aVec, weightIndex, seed * 10000 + weightIndex * 100 + repIdx, outFile, /*directSeed=*/true);
    } else {
        task.exportTrajectory(wVec, aVec, weightIndex, seed, outFile);
    }
    std::cout << "wrote " << outFile << " (weight " << EvalTask::WEIGHT_VALS[weightIndex]
              << ", seed " << seed << ")\n";
    return 0;
}

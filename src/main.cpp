// wann_train – WANN evolution loop.
//
// Usage:
//   ./wann_train [-d default.json] [-p overrides.json] [-o prefix] [-s seed]
//
// Implement ITask (see include/wann/Task.h) and wire it into evalPop() below
// to connect the algorithm to any environment.

#include "wann/DataGatherer.h"
#include "wann/Hyperparams.h"
#include "wann/Ind.h"
#include "wann/Random.h"
#include "wann/Task.h"
#include "wann/Wann.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =========================================================================
// evalPop – evaluate the entire population.
//
// Returns reward[i][j] = fitness of pop[i] with j-th weight value.
//
// Replace the stub below with your ITask implementation.
// =========================================================================
static std::vector<std::vector<double>>
evalPop(const std::vector<wann::Ind>& pop,
        wann::ITask* task,
        const wann::Hyperparams& hyp,
        int seed)
{
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<double>> reward(n);

    for (int i = 0; i < n; ++i) {
        if (task) {
            reward[i] = task->getDistFitness(pop[i].wVec, pop[i].aVec, seed);
        } else {
            // ---- STUB: replace with real evaluation ----
            // Returns zeros so the algorithm runs without a task.
            reward[i].assign(hyp.alg_nVals, 0.0);
        }
    }
    return reward;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char* argv[]) {
    // --- Parse arguments ---
    std::string defaultHyp = "p/default_wan.json";
    std::string overrideHyp;
    std::string outPrefix  = "test";
    uint32_t    seed       = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-d" && i+1 < argc) { defaultHyp  = argv[++i]; }
        else if (arg == "-p" && i+1 < argc) { overrideHyp = argv[++i]; }
        else if (arg == "-o" && i+1 < argc) { outPrefix   = argv[++i]; }
        else if (arg == "-s" && i+1 < argc) { seed = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else {
            std::cerr << "Usage: wann_train [-d default.json] [-p overrides.json]"
                         " [-o prefix] [-s seed]\n";
            return 1;
        }
    }

    // --- Load hyperparameters ---
    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(defaultHyp);
        if (!overrideHyp.empty()) wann::updateHyp(hyp, overrideHyp);
    } catch (const std::exception& e) {
        std::cerr << "Error loading hyperparameters: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Task: " << hyp.task
              << "  popSize: " << hyp.popSize
              << "  maxGen: "  << hyp.maxGen << '\n';

    // --- Seed global RNG ---
    wann::seedRng(seed);

    // --- Prepare output directory ---
    fs::create_directories("log");

    // --- Wire up task (nullptr = stub) ---
    wann::ITask* task = nullptr;
    // Example:
    //   MyTask myTask(hyp);
    //   task = &myTask;

    // --- Evolution loop ---
    wann::Wann          wann(hyp);
    wann::DataGatherer  data(outPrefix, hyp);

    for (int gen = 0; gen < hyp.maxGen; ++gen) {
        auto& pop    = wann.ask();
        auto  reward = evalPop(pop, task, hyp, static_cast<int>(seed));
        wann.tell(reward);

        data.gatherData(pop);
        std::cout << gen << "\t - \t" << data.display() << '\n';

        if (gen % hyp.save_mod == 0) {
            data.save(gen);
        }
    }

    // Final save.
    data.save();
    std::cout << "Done. Results written to log/" << outPrefix << "_*\n";
    return 0;
}

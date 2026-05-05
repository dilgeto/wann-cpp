// wann_mountain_car – WANN evolution + SNN simulator + rl-tools MountainCarContinuous-v0.
//
// Usage:
//   ./wann_mountain_car [-d mountain_car_snn.json] [-p overrides.json] [-o prefix] [-s seed] [-v]

#include "../include/wann/DataGatherer.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/SnnDebug.h"
#include "../include/wann/SnnMountainCarTask.h"
#include "../include/wann/Wann.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<std::vector<double>>
evalPop(const std::vector<wann::Ind>& pop,
        wann::SnnMountainCarTask&     task,
        int                           seed)
{
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<double>> reward(n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
        reward[i] = task.evaluate(pop[i], seed * 10000 + i);

    return reward;
}

int main(int argc, char* argv[]) {
    std::string defaultHyp = "p/mountain_car_snn.json";
    std::string overrideHyp;
    std::string outPrefix  = "snn_mountain_car";
    uint32_t    seed       = 42;
    bool        debugLog   = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-d" && i+1 < argc) { defaultHyp  = argv[++i]; }
        else if (arg == "-p" && i+1 < argc) { overrideHyp = argv[++i]; }
        else if (arg == "-o" && i+1 < argc) { outPrefix   = argv[++i]; }
        else if (arg == "-s" && i+1 < argc) { seed = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg == "-v")               { debugLog = true; }
        else {
            std::cerr << "Usage: wann_mountain_car [-d default.json] [-p overrides.json]"
                         " [-o prefix] [-s seed] [-v]\n";
            return 1;
        }
    }

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(defaultHyp);
        if (!overrideHyp.empty()) wann::updateHyp(hyp, overrideHyp);
    } catch (const std::exception& e) {
        std::cerr << "Error loading hyperparameters: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Task: SNN MountainCar"
              << "  nInput="  << hyp.ann_nInput
              << "  nOutput=" << hyp.ann_nOutput
              << "  popSize=" << hyp.popSize
              << "  maxGen="  << hyp.maxGen << '\n';

    wann::seedRng(seed);
    fs::create_directories("log");

    std::ofstream dbgFile;
    if (debugLog) {
        std::string dbgPath = "log/" + outPrefix + "_debug.log";
        dbgFile.open(dbgPath);
        if (!dbgFile)
            std::cerr << "Warning: cannot open debug log " << dbgPath << '\n';
        else
            std::cout << "Debug log: " << dbgPath << '\n';
    }

    wann::SnnMountainCarTask task(hyp);
    wann::Wann               alg(hyp);
    wann::DataGatherer       data(outPrefix, hyp);

    for (int gen = 0; gen < hyp.maxGen; ++gen) {
        auto& pop    = alg.ask();
        auto  reward = evalPop(pop, task, static_cast<int>(seed));
        alg.tell(reward);

        data.gatherData(pop);
        std::cout << gen << "\t - \t" << data.display() << '\n';

        if (gen % hyp.save_mod == 0) {
            data.save(gen);
            data.savePareto(pop, gen);
            if (dbgFile) {
                dbgFile << "========== Generation " << gen << " ==========\n";
                wann::debugSnn(pop[0], dbgFile);
                dbgFile.flush();
            }
        }
    }

    data.save();
    std::cout << "Done. Results written to log/" << outPrefix << "_*\n";
    return 0;
}

// wann_acrobot – WANN evolution driven by the SNN simulator + rl-tools Acrobot.
//
// Usage:
//   ./wann_acrobot [-d acrobot_snn.json] [-p overrides.json] [-o prefix] [-s seed] [-v]

#include "../include/wann/DataGatherer.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/SnnAcrobotTask.h"
#include "../include/wann/SnnDebug.h"
#include "../include/wann/Wann.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int REPLAY_INTERVAL = 256;

static std::vector<std::vector<double>>
evalPop(const std::vector<wann::Ind>& pop,
        wann::SnnAcrobotTask&         task,
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
    std::string defaultHyp = "p/acrobot_snn.json";
    std::string overrideHyp;
    std::string outPrefix  = "snn_acrobot";
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
            std::cerr << "Usage: wann_acrobot [-d default.json] [-p overrides.json]"
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

    std::cout << "Task: SNN Acrobot"
              << "  nInput="  << hyp.ann_nInput
              << "  nOutput=" << hyp.ann_nOutput
              << "  popSize=" << hyp.popSize
              << "  maxGen="  << hyp.maxGen << '\n';

    wann::seedRng(seed);
    fs::create_directories("log");

    const std::string replayDir = "log/" + outPrefix + "_replay";
    fs::create_directories(replayDir);

    std::ofstream dbgFile;
    if (debugLog) {
        std::string dbgPath = "log/" + outPrefix + "_debug.log";
        dbgFile.open(dbgPath);
        if (!dbgFile)
            std::cerr << "Warning: cannot open debug log " << dbgPath << '\n';
        else
            std::cout << "Debug log: " << dbgPath << '\n';
    }

    wann::SnnAcrobotTask task(hyp);
    wann::Wann           alg(hyp);
    wann::DataGatherer   data(outPrefix, hyp);

    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    for (int gen = 0; gen < hyp.maxGen; ++gen) {
        auto& pop    = alg.ask();
        auto  reward = evalPop(pop, task, static_cast<int>(seed) + gen);
        alg.tell(reward);

        int eliteIdx = static_cast<int>(
            std::max_element(pop.begin(), pop.end(),
                [](const wann::Ind& a, const wann::Ind& b){ return a.fitness < b.fitness; })
            - pop.begin());

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

        if ((gen % REPLAY_INTERVAL == 0 || gen == hyp.maxGen - 1)
                && !pop[eliteIdx].wVec.empty()) {
            const auto& rw = reward[eliteIdx];
            int bestWi = static_cast<int>(
                std::max_element(rw.begin(), rw.end()) - rw.begin());

            std::ostringstream fname;
            fname << replayDir << "/gen_"
                  << std::setw(4) << std::setfill('0') << gen << ".csv";
            try {
                task.exportTrajectory(pop[eliteIdx].wVec, pop[eliteIdx].aVec,
                                      wann::SnnAcrobotTask::WEIGHT_VALS[bestWi],
                                      static_cast<int>(seed) + gen,
                                      fname.str());
            } catch (const std::exception& e) {
                std::cerr << "Warning: no se pudo guardar replay gen " << gen
                          << ": " << e.what() << '\n';
            }
        }
    }

    double total_s = std::chrono::duration<double>(Clock::now() - t_start).count();
    double per_gen = total_s / hyp.maxGen;

    std::ofstream tlog("log/" + outPrefix + "_time.log");
    tlog << std::fixed << std::setprecision(3)
         << "total_s   " << total_s    << '\n'
         << "per_gen_s " << per_gen    << '\n'
         << "maxGen    " << hyp.maxGen  << '\n'
         << "popSize   " << hyp.popSize << '\n';
    std::cout << "Time: " << total_s << " s  (" << per_gen << " s/gen)\n";

    data.save();
    std::cout << "Done. Results written to log/" << outPrefix << "_*\n";
    return 0;
}

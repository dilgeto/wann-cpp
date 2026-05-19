// wann_acrobot_eval – carga una red entrenada y evalúa N episodios.
//
// Uso:
//   ./wann_acrobot_eval
//   ./wann_acrobot_eval -f log/snn_acrobot_best.out -d p/acrobot_snn.json -n 20 -s 0
//   ./wann_acrobot_eval -w 1.5 -n 50   (fija el peso en lugar de buscarlo)

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnAcrobotTask.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string netFile    = "log/snn_acrobot_best.out";
    std::string configFile = "p/acrobot_snn.json";
    std::string weightArg  = "best";
    int         nEpisodes  = 10;
    int         seed       = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f" && i+1 < argc) { netFile    = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-w" && i+1 < argc) { weightArg  = argv[++i]; }
        else if (arg == "-n" && i+1 < argc) { nEpisodes  = std::atoi(argv[++i]); }
        else if (arg == "-s" && i+1 < argc) { seed       = std::atoi(argv[++i]); }
        else {
            std::cerr << "Uso: wann_acrobot_eval [-f red.out] [-d config.json]"
                         " [-w peso|best] [-n episodios] [-s seed]\n";
            return 1;
        }
    }

    wann::Hyperparams hyp;
    try { hyp = wann::loadHyp(configFile); }
    catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    auto [wVec, aVec, wKey] = wann::importNet(netFile);
    wann::SnnAcrobotTask task(hyp);

    double chosenWeight;
    if (weightArg == "best") {
        // Try companion file saved during training first.
        std::string wiFile = netFile.substr(0, netFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (wif) {
            int wi; wif >> wi;
            chosenWeight = wann::SnnAcrobotTask::WEIGHT_VALS[wi];
            std::cout << "Peso cargado de " << wiFile << ": " << chosenWeight
                      << "  (wi=" << wi << ")\n";
        } else {
            auto allRewards = task.getDistFitness(wVec, aVec, seed);
            int bestWi = static_cast<int>(
                std::max_element(allRewards.begin(), allRewards.end()) - allRewards.begin());
            chosenWeight = wann::SnnAcrobotTask::WEIGHT_VALS[bestWi];
            std::cout << "Pesos evaluados: ";
            for (int i = 0; i < wann::SnnAcrobotTask::N_WEIGHTS; ++i)
                std::cout << wann::SnnAcrobotTask::WEIGHT_VALS[i]
                          << "→" << std::fixed << std::setprecision(2) << allRewards[i]
                          << (i+1 < wann::SnnAcrobotTask::N_WEIGHTS ? "  " : "\n");
            std::cout << "Mejor peso: " << chosenWeight
                      << "  (reward=" << allRewards[bestWi] << ")\n";
        }
    } else {
        chosenWeight = std::stod(weightArg);
        std::cout << "Peso fijo: " << chosenWeight << '\n';
    }

    std::cout << "Evaluando " << nEpisodes << " episodios (seed base=" << seed << ")...\n\n";
    auto rewards = task.evalEpisodes(wVec, aVec, chosenWeight, nEpisodes, seed);

    const double mean = std::accumulate(rewards.begin(), rewards.end(), 0.0) / nEpisodes;
    double sq = 0;
    for (double r : rewards) sq += (r - mean) * (r - mean);
    const double std_dev = std::sqrt(sq / nEpisodes);
    const double rmin = *std::min_element(rewards.begin(), rewards.end());
    const double rmax = *std::max_element(rewards.begin(), rewards.end());

    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < nEpisodes; ++i)
        std::cout << "  ep " << std::setw(3) << i << ": " << rewards[i] << '\n';
    std::cout << "\nResumen (" << nEpisodes << " episodios):\n"
              << "  Media:  " << mean    << '\n'
              << "  StdDev: " << std_dev << '\n'
              << "  Min:    " << rmin    << '\n'
              << "  Max:    " << rmax    << '\n';
    return 0;
}

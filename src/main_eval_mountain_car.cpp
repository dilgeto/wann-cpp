// wann_mountain_car_eval – carga una red entrenada y evalúa N episodios.
//
// Uso:
//   ./wann_mountain_car_eval
//   ./wann_mountain_car_eval -f log/snn_mountain_car_best.out -d p/mountain_car_snn.json -n 20 -s 0
//   ./wann_mountain_car_eval -w 1.5 -n 50
//   ./wann_mountain_car_eval -n 20 -S best          (guarda el mejor episodio)
//   ./wann_mountain_car_eval -n 20 -S 3             (guarda el episodio 3)
//   ./wann_mountain_car_eval -n 20 -S best -o mi_replay.csv

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnMountainCarTask.h"

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
    std::string netFile    = "log/snn_mountain_car_best.out";
    std::string configFile = "p/mountain_car_snn.json";
    std::string weightArg  = "best";
    std::string saveArg;   // "best", número, o vacío (no guardar)
    std::string outFile;
    int         nEpisodes  = 10;
    int         seed       = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f" && i+1 < argc) { netFile    = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-w" && i+1 < argc) { weightArg  = argv[++i]; }
        else if (arg == "-n" && i+1 < argc) { nEpisodes  = std::atoi(argv[++i]); }
        else if (arg == "-s" && i+1 < argc) { seed       = std::atoi(argv[++i]); }
        else if (arg == "-S" && i+1 < argc) { saveArg    = argv[++i]; }
        else if (arg == "-o" && i+1 < argc) { outFile    = argv[++i]; }
        else {
            std::cerr << "Uso: wann_mountain_car_eval [-f red.out] [-d config.json]"
                         " [-w peso|best] [-n episodios] [-s seed]"
                         " [-S best|N] [-o salida.csv]\n";
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
    wann::SnnMountainCarTask task(hyp);

    int    chosenWiIdx  = 0;
    double chosenWeight = wann::SnnMountainCarTask::WEIGHT_VALS[0];

    if (weightArg == "best") {
        std::string wiFile = netFile.substr(0, netFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (wif) {
            wif >> chosenWiIdx;
            chosenWeight = wann::SnnMountainCarTask::WEIGHT_VALS[chosenWiIdx];
            std::cout << "Peso cargado de " << wiFile << ": " << chosenWeight
                      << "  (wi=" << chosenWiIdx << ")\n";
        } else {
            auto allRewards = task.getDistFitness(wVec, aVec, seed);
            chosenWiIdx = static_cast<int>(
                std::max_element(allRewards.begin(), allRewards.end()) - allRewards.begin());
            chosenWeight = wann::SnnMountainCarTask::WEIGHT_VALS[chosenWiIdx];
            std::cout << "Pesos evaluados: ";
            for (int i = 0; i < wann::SnnMountainCarTask::N_WEIGHTS; ++i)
                std::cout << wann::SnnMountainCarTask::WEIGHT_VALS[i]
                          << "→" << std::fixed << std::setprecision(2) << allRewards[i]
                          << (i+1 < wann::SnnMountainCarTask::N_WEIGHTS ? "  " : "\n");
            std::cout << "Mejor peso: " << chosenWeight
                      << "  (reward=" << allRewards[chosenWiIdx] << ")\n";
        }
    } else {
        chosenWeight = std::stod(weightArg);
        // find closest index
        double best_dist = std::abs(wann::SnnMountainCarTask::WEIGHT_VALS[0] - chosenWeight);
        for (int i = 1; i < wann::SnnMountainCarTask::N_WEIGHTS; ++i) {
            double d = std::abs(wann::SnnMountainCarTask::WEIGHT_VALS[i] - chosenWeight);
            if (d < best_dist) { best_dist = d; chosenWiIdx = i; }
        }
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

    if (!saveArg.empty()) {
        int epToSave;
        if (saveArg == "best") {
            epToSave = static_cast<int>(
                std::max_element(rewards.begin(), rewards.end()) - rewards.begin());
            std::cout << "\nGuardando episodio " << epToSave
                      << " (mejor, reward=" << rewards[epToSave] << ")...\n";
        } else {
            epToSave = std::atoi(saveArg.c_str());
            std::cout << "\nGuardando episodio " << epToSave
                      << " (reward=" << rewards[epToSave] << ")...\n";
        }

        if (outFile.empty()) {
            auto stem = netFile.substr(0, netFile.rfind('.'));
            outFile = stem + "_ep" + std::to_string(epToSave) + "_replay.csv";
        }

        task.exportTrajectory(wVec, aVec, chosenWiIdx, seed + epToSave, outFile, true);
        std::cout << "Replay guardado en: " << outFile << '\n';
    }

    return 0;
}

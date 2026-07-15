// wann_disc_mc_eval – carga una red entrenada y evalúa N episodios.
//
// Uso:
//   ./wann_disc_mc_eval
//   ./wann_disc_mc_eval -f log/snn_disc_mc_best.out -d p/disc_mc_snn.json -n 20 -s 0
//   ./wann_disc_mc_eval -w 2.0 -n 50
//   ./wann_disc_mc_eval -n 20 -S best          (guarda el mejor episodio)
//   ./wann_disc_mc_eval -n 20 -S 3             (guarda el episodio 3)
//   ./wann_disc_mc_eval -n 20 -S best -o mi_replay.csv

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnDiscMCTask.h"

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
    std::string netFile    = "log/snn_disc_mc_best.out";
    std::string configFile = "p/disc_mc_snn.json";
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
            std::cerr << "Uso: wann_disc_mc_eval [-f red.out] [-d config.json]"
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
    wann::SnnDiscMCTask task(hyp);

    int    chosenWiIdx  = 0;
    double chosenWeight = wann::SnnDiscMCTask::WEIGHT_VALS[0];

    if (weightArg == "best") {
        std::string wiFile = netFile.substr(0, netFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (wif) {
            wif >> chosenWiIdx;
            chosenWeight = wann::SnnDiscMCTask::WEIGHT_VALS[chosenWiIdx];
            std::cout << "Peso cargado de " << wiFile << ": " << chosenWeight
                      << "  (wi=" << chosenWiIdx << ")\n";
        } else {
            auto allRewards = task.getDistFitness(wVec, aVec, seed);
            chosenWiIdx = static_cast<int>(
                std::max_element(allRewards.begin(), allRewards.end()) - allRewards.begin());
            chosenWeight = wann::SnnDiscMCTask::WEIGHT_VALS[chosenWiIdx];
            std::cout << "Pesos evaluados: ";
            for (int i = 0; i < wann::SnnDiscMCTask::N_WEIGHTS; ++i)
                std::cout << wann::SnnDiscMCTask::WEIGHT_VALS[i]
                          << "→" << std::fixed << std::setprecision(2) << allRewards[i]
                          << (i+1 < wann::SnnDiscMCTask::N_WEIGHTS ? "  " : "\n");
            std::cout << "Mejor peso: " << chosenWeight
                      << "  (reward=" << allRewards[chosenWiIdx] << ")\n";
        }
    } else {
        chosenWeight = std::stod(weightArg);
        // find closest index
        double best_dist = std::abs(wann::SnnDiscMCTask::WEIGHT_VALS[0] - chosenWeight);
        for (int i = 1; i < wann::SnnDiscMCTask::N_WEIGHTS; ++i) {
            double d = std::abs(wann::SnnDiscMCTask::WEIGHT_VALS[i] - chosenWeight);
            if (d < best_dist) { best_dist = d; chosenWiIdx = i; }
        }
        std::cout << "Peso fijo: " << chosenWeight << '\n';
    }

    std::cout << "Evaluando " << nEpisodes << " episodios (seed base=" << seed << ")...\n\n";
    auto [rewards_shaped, rewards_orig] = task.evalEpisodes(wVec, aVec, chosenWeight, nEpisodes, seed);

    auto stats = [&](const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / nEpisodes;
        double sq = 0; for (double r : v) sq += (r - mean) * (r - mean);
        return std::make_tuple(mean, std::sqrt(sq / nEpisodes),
                               *std::min_element(v.begin(), v.end()),
                               *std::max_element(v.begin(), v.end()));
    };
    auto [mean_s, std_s, min_s, max_s] = stats(rewards_shaped);
    auto [mean_o, std_o, min_o, max_o] = stats(rewards_orig);

    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < nEpisodes; ++i)
        std::cout << "  ep " << std::setw(3) << i
                  << ": shaped=" << rewards_shaped[i]
                  << "  original=" << rewards_orig[i] << '\n';
    std::cout << "\nResumen (" << nEpisodes << " episodios):\n"
              << "  [Shaped]   Media=" << mean_s << "  StdDev=" << std_s
              << "  Min=" << min_s << "  Max=" << max_s << '\n'
              << "  [Original] Media=" << mean_o << "  StdDev=" << std_o
              << "  Min=" << min_o << "  Max=" << max_o << '\n';

    if (!saveArg.empty()) {
        int epToSave;
        if (saveArg == "best") {
            epToSave = static_cast<int>(
                std::max_element(rewards_shaped.begin(), rewards_shaped.end()) - rewards_shaped.begin());
            std::cout << "\nGuardando episodio " << epToSave
                      << " (mejor, shaped=" << rewards_shaped[epToSave] << ")...\n";
        } else {
            epToSave = std::atoi(saveArg.c_str());
            std::cout << "\nGuardando episodio " << epToSave
                      << " (shaped=" << rewards_shaped[epToSave] << ")...\n";
        }

        if (outFile.empty()) {
            auto stem = netFile.substr(0, netFile.rfind('.'));
            outFile = stem + "_ep" + std::to_string(epToSave) + "_replay.csv";
        }

        // Reproducir con la misma semilla que usó evalEpisodes para ese episodio.
        task.exportTrajectory(wVec, aVec, chosenWiIdx, seed + epToSave, outFile, true);
        std::cout << "Replay guardado en: " << outFile << '\n';
    }

    return 0;
}

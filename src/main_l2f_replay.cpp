// wann_l2f_replay – carga la mejor red del L2F, corre un episodio y exporta
// la trayectoria a CSV para visualización 3D.
//
// Uso:
//   ./wann_l2f_replay
//   ./wann_l2f_replay -f log/snn_l2f_best.out -d p/l2f_snn.json -w best -s 0
//   ./wann_l2f_replay -w 4   (usa el índice 4 de WEIGHT_VALS)

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnL2FTask.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string bestFile   = "log/snn_l2f_best.out";
    std::string configFile = "p/l2f_snn.json";
    std::string weightArg  = "best";   // "best" | índice de WEIGHT_VALS
    int         seed       = 0;
    std::string outFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f" && i+1 < argc) { bestFile   = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-w" && i+1 < argc) { weightArg  = argv[++i]; }
        else if (arg == "-s" && i+1 < argc) { seed       = std::atoi(argv[++i]); }
        else if (arg == "-o" && i+1 < argc) { outFile    = argv[++i]; }
        else {
            std::cerr << "Uso: wann_l2f_replay [-f best.out] [-d config.json]"
                         " [-w idx|best] [-s seed] [-o salida.csv]\n";
            return 1;
        }
    }

    if (outFile.empty()) {
        auto dot = bestFile.rfind('.');
        outFile = (dot != std::string::npos ? bestFile.substr(0, dot) : bestFile)
                  + "_replay.csv";
    }

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(configFile);
    } catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    auto [wVec, aVec, wKey] = wann::importNet(bestFile);

    wann::SnnL2FTask task(hyp);

    int chosenWi;
    if (weightArg == "best") {
        std::string wiFile = bestFile.substr(0, bestFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (wif) {
            wif >> chosenWi;
            std::cout << "Peso cargado de " << wiFile << ": "
                      << wann::SnnL2FTask::WEIGHT_VALS[chosenWi]
                      << "  (wi=" << chosenWi << ")\n";
        } else {
            auto rewards = task.getDistFitness(wVec, aVec, seed);
            chosenWi = static_cast<int>(
                std::max_element(rewards.begin(), rewards.end()) - rewards.begin());
            std::cout << "Peso seleccionado automáticamente: "
                      << wann::SnnL2FTask::WEIGHT_VALS[chosenWi]
                      << "  (reward=" << rewards[chosenWi] << ")\n";
        }
    } else {
        chosenWi = std::atoi(weightArg.c_str());
    }

    std::cout << "Corriendo episodio — peso=" << wann::SnnL2FTask::WEIGHT_VALS[chosenWi]
              << "  seed=" << seed << '\n';

    task.exportTrajectory(wVec, aVec, chosenWi, seed, outFile);
    std::cout << "Guardado: " << outFile << '\n';
    return 0;
}

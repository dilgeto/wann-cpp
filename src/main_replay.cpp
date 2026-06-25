// wann_car_replay – carga la mejor red del Car, corre un episodio y exporta
// la trayectoria a CSV para visualización.
//
// Uso:
//   ./wann_car_replay
//   ./wann_car_replay -f log/snn_car_best.out -d p/car_snn.json -w 5.0 -s 0
//   ./wann_car_replay -w best   (prueba los 6 pesos y elige el mejor)

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnCarTask.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string bestFile   = "log/snn_car_best.out";
    std::string configFile = "p/car_snn.json";
    std::string weightArg  = "best";   // "best" | float value
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
            std::cerr << "Uso: wann_car_replay [-f best.out] [-d config.json]"
                         " [-w peso|best] [-s seed] [-o salida.csv]\n";
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

    wann::SnnCarTask task(hyp);

    double chosenWeight;
    if (weightArg == "best") {
        // Evalúa todos los pesos y elige el que da mayor reward
        auto rewards = task.getDistFitness(wVec, aVec, seed);
        int  best_i  = 0;
        for (int i = 1; i < static_cast<int>(rewards.size()); ++i)
            if (rewards[i] > rewards[best_i]) best_i = i;
        chosenWeight = wann::SnnCarTask::WEIGHT_VALS[best_i];
        std::cout << "Peso seleccionado automáticamente: " << chosenWeight
                  << "  (reward=" << rewards[best_i] << ")\n";
    } else {
        chosenWeight = std::stod(weightArg);
    }

    std::cout << "Corriendo episodio — peso=" << chosenWeight
              << "  seed=" << seed << '\n';

    task.exportTrajectory(wVec, aVec, chosenWeight, seed, outFile);
    return 0;
}

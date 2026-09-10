// wann_car_odin_export – convierte una red car ya evolucionada (formato
// exportNet/.out + .wi) a una configuración lista para el core ODIN
// (mapeo de neuronas a direcciones físicas 0..255, parámetros Izhikevich
// por neurona, tabla de sinapsis con peso de 3 bits).
//
// Esta herramienta corre en la máquina de desarrollo (x86) — no necesita el
// ZCU104 ni el simulador SNN completo, solo el archivo .out/.wi ya
// entrenado. El JSON que produce lo consume el driver de hardware
// (include/wann/hw/OdinDriver.h, todavía pendiente del mapa de registros
// real del overlay).
//
// Uso:
//   ./wann_car_odin_export -f log/full_p3_car_ttfs_first_spike_40ms/rank00_seed00_best.out \
//                           -d p/car_snn.json -o odin_config.json
//   ./wann_car_odin_export -f red.out -w 3 -o odin_config.json   (fuerza wi=3)

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/OdinExport.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {
// Mirrors SnnCarTask::WEIGHT_VALS (src/SnnCarTask.h) — duplicated so this
// tool doesn't need to link the full car task/rl-tools stack. Keep in sync.
constexpr int    N_WEIGHTS   = 6;
constexpr double WEIGHT_VALS[N_WEIGHTS] = {0.5, 1.0, 2.0, 3.0, 5.0, 8.0};
constexpr double WEIGHT_MAX  = WEIGHT_VALS[N_WEIGHTS - 1];

int quantizeMagnitude3bit(double sharedWeight) {
    double frac = std::abs(sharedWeight) / WEIGHT_MAX;
    int bucket = static_cast<int>(std::lround(std::clamp(frac, 0.0, 1.0) * 7.0));
    return std::clamp(bucket, 0, 7);
}
} // namespace

int main(int argc, char* argv[]) {
    std::string netFile    = "log/snn_car_best.out";
    std::string configFile = "p/car_snn.json";
    std::string outFile    = "odin_config.json";
    std::string runKey;
    int         wiArg      = -1;  // -1 = read from .wi next to netFile

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f" && i+1 < argc) { netFile    = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-o" && i+1 < argc) { outFile    = argv[++i]; }
        else if (arg == "-w" && i+1 < argc) { wiArg      = std::atoi(argv[++i]); }
        else if (arg == "-k" && i+1 < argc) { runKey     = argv[++i]; }
        else {
            std::cerr << "Uso: wann_car_odin_export [-f red.out] [-d config.json]"
                         " [-w weight_index] [-k run_key] [-o salida.json]\n";
            return 1;
        }
    }

    wann::Hyperparams hyp;
    try { hyp = wann::loadHyp(configFile); }
    catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    std::vector<double> wVec;
    std::vector<int>    aVec;
    try {
        auto [w, a, wKey] = wann::importNet(netFile);
        wVec = std::move(w);
        aVec = std::move(a);
        (void)wKey;
    } catch (const std::exception& e) {
        std::cerr << "Error cargando red: " << e.what() << '\n';
        return 1;
    }

    int chosenWi = wiArg;
    if (chosenWi < 0) {
        std::string wiFile = netFile.substr(0, netFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (!wif) {
            std::cerr << "No encontre " << wiFile << " y no se paso -w explicito; "
                         "no puedo elegir el peso compartido ganador.\n";
            return 1;
        }
        wif >> chosenWi;
    }
    if (chosenWi < 0 || chosenWi >= N_WEIGHTS) {
        std::cerr << "weight_index fuera de rango: " << chosenWi << '\n';
        return 1;
    }
    const double sharedWeight = WEIGHT_VALS[chosenWi];
    const int    magnitude3bit = quantizeMagnitude3bit(sharedWeight);

    if (runKey.empty()) {
        // Best-effort guess from the netFile's directory name; purely cosmetic.
        auto slash = netFile.find_last_of('/');
        runKey = (slash == std::string::npos) ? "car" : netFile.substr(0, slash);
    }

    try {
        auto cfg = wann::buildOdinConfig(wVec, aVec, hyp.ann_nInput, hyp.ann_nOutput,
                                          sharedWeight, chosenWi, magnitude3bit,
                                          "car", runKey);
        wann::writeOdinConfig(outFile, cfg);
        std::cout << "Config ODIN escrita en " << outFile
                  << " (" << cfg.nNeurons << " neuronas / " << wann::ODIN_MAX_NEURONS
                  << " maximo, " << cfg.synapses.size() << " sinapsis, "
                  << "peso compartido=" << sharedWeight
                  << " -> magnitud 3 bits=" << magnitude3bit << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

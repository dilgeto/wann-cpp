// wann_car_odin_eval – corre N episodios de car sobre el core ODIN real
// (ZCU104/PYNQ), usando la configuración exportada por wann_car_odin_export.
//
// Solo se compila con -DWANN_ODIN_HW=ON, y solo puede correr de verdad en el
// ZCU104 una vez que include/wann/hw/OdinRegisters.h tenga el mapa de
// registros real (ver ese archivo) y OdinDriver.cpp implemente el protocolo
// SPI/AER real — hasta entonces construye pero falla al abrir /dev/mem o al
// mandar el primer spike, con un error explícito señalando qué falta.
//
// Uso:
//   ./wann_car_odin_eval -c odin_config.json -d p/car_snn.json -n 10 -s 0 -m 40
//
// -m/--window-ms DEBE coincidir con el SIM_WINDOW_MS con el que se entrenó
// el modelo exportado (no viene en car_snn.json — es una constante de
// compilación en SnnCarTask.h, WANN_CAR_SIM_WINDOW_MS). Un modelo podado
// (pruned) puede haberse evolucionado con una ventana distinta a la base.

#include "../include/wann/Hyperparams.h"
#include "../include/wann/OdinExport.h"
#include "../include/wann/SnnCarOdinTask.h"
#include "../include/wann/hw/OdinDriver.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

int main(int argc, char* argv[]) {
    std::string cfgFile    = "odin_config.json";
    std::string configFile = "p/car_snn.json";
    int         nEpisodes  = 10;
    int         seed       = 0;
    double      windowMs   = 40.0;
    bool        windowMsSet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-c" && i+1 < argc) { cfgFile    = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-n" && i+1 < argc) { nEpisodes  = std::atoi(argv[++i]); }
        else if (arg == "-s" && i+1 < argc) { seed       = std::atoi(argv[++i]); }
        else if ((arg == "-m" || arg == "--window-ms") && i+1 < argc) {
            windowMs = std::atof(argv[++i]); windowMsSet = true;
        } else {
            std::cerr << "Uso: wann_car_odin_eval [-c odin_config.json] "
                         "[-d config.json] [-n episodios] [-s seed] [-m window_ms]\n";
            return 1;
        }
    }
    if (!windowMsSet) {
        std::cerr << "Aviso: -m/--window-ms no especificado, usando el default "
                     "40 ms (WANN_CAR_SIM_WINDOW_MS) — confirma que coincide con "
                     "el modelo exportado, algunos (p.ej. podados) usan otra ventana.\n";
    }

    wann::Hyperparams hyp;
    try { hyp = wann::loadHyp(configFile); }
    catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    try {
        auto cfg = wann::readOdinConfig(cfgFile);
        std::cout << "Config ODIN cargada de " << cfgFile
                  << " (" << cfg.nNeurons << " neuronas, "
                  << cfg.synapses.size() << " sinapsis, run_key=" << cfg.runKey << ")\n";

        wann::hw::OdinDriver driver;
        wann::SnnCarOdinTask task(hyp, cfg, driver, windowMs);

        std::cout << "Evaluando " << nEpisodes << " episodios en hardware "
                     "(seed base=" << seed << ")...\n\n";
        auto rewards = task.evalEpisodes(nEpisodes, seed);

        double mean = std::accumulate(rewards.begin(), rewards.end(), 0.0) / nEpisodes;
        double sq = 0; for (double r : rewards) sq += (r - mean) * (r - mean);
        double stdev = std::sqrt(sq / nEpisodes);

        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < nEpisodes; ++i)
            std::cout << "  ep " << std::setw(3) << i << ": reward=" << rewards[i] << '\n';
        std::cout << "\nResumen (" << nEpisodes << " episodios en ODIN): media=" << mean
                  << "  desv=" << stdev << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

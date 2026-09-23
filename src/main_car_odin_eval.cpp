// wann_car_odin_eval – corre N episodios de car sobre el core ODIN real
// (ZCU104/PYNQ), usando la configuración exportada por wann_car_odin_export.
//
// Solo se compila con -DWANN_ODIN_HW=ON. Antes de correrlo, en cada power
// cycle / recarga del bitstream, hay que correr una vez:
//   python3 odin_bootstrap.py /ruta/a/odin.bit
// (descarga el overlay y hace que PYNQ configure la direccion de los AXI
// GPIO — ver ese script). Ademas necesita que
// include/wann/hw/OdinRegisters.h tenga las 5 direcciones fisicas reales
// (el propio odin_bootstrap.py las imprime) — hasta entonces este binario
// compila pero falla al construir OdinDriver con un error explicito.
//
// Uso:
//   ./wann_car_odin_eval -c odin_config.json -d p/car_snn.json -n 10 -s 0 -m 40
//
// -m/--window-ms DEBE coincidir con el snn_window_ms con el que se entrenó
// el modelo exportado (campo de Hyperparams — car_snn.json lo trae, pero se
// pide explícito acá porque un modelo podado puede haberse evolucionado con
// una ventana distinta a la base y no hay forma de saberlo solo del .out).
//
// --csv: imprime SOLO "episode,reward\n0,123.45\n..." a stdout (todo lo
// demás va a stderr) — pensado para que un script Python capture stdout
// directo con pd.read_csv(io.StringIO(...)), igual que hace
// bootstrap_auto_lib.eval_snn() con el binario wann_eval_weights_car
// (--episode-detail). Usado por bootstrap_compare_car_odin_auto.py.
//
// --profile: al terminar, imprime (a stderr en modo --csv) el desglose de
// tiempo de pared acumulado entre el entorno RL (rl-tools) y el camino de
// hardware ODIN (encoder, envío de eventos AER, espera de AER_OUT, decoder)
// — para diagnosticar en el propio chip si lo que domina el tiempo total es
// la red de espigas/protocolo AER o el entorno, en vez de asumirlo.

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
    // std::cout is fully buffered once piped/redirected (e.g. `| tee log`) —
    // if the process hangs or freezes the whole board mid-run, everything
    // printed so far can be sitting in that buffer and never reach the
    // pipe/file. Force a flush after every write so the log always reflects
    // exactly how far execution got, no matter how it ends. (Don't rely on
    // `stdbuf` for this from the outside: under sudo the dynamic linker
    // ignores LD_PRELOAD for privileged processes, so `stdbuf` silently has
    // no effect there.)
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    std::string cfgFile    = "odin_config.json";
    std::string configFile = "p/car_snn.json";
    int         nEpisodes  = 10;
    int         seed       = 0;
    double      windowMs   = 40.0;
    bool        windowMsSet = false;
    bool        csvMode    = false;
    bool        profile    = false;
    int         maxSteps   = 0;   // 0 = default (1000 pasos por episodio)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-c" && i+1 < argc) { cfgFile    = argv[++i]; }
        else if (arg == "-d" && i+1 < argc) { configFile = argv[++i]; }
        else if (arg == "-n" && i+1 < argc) { nEpisodes  = std::atoi(argv[++i]); }
        else if (arg == "-s" && i+1 < argc) { seed       = std::atoi(argv[++i]); }
        else if ((arg == "-m" || arg == "--window-ms") && i+1 < argc) {
            windowMs = std::atof(argv[++i]); windowMsSet = true;
        } else if (arg == "--csv") { csvMode = true;
        } else if (arg == "--profile") { profile = true;
        } else if (arg == "--steps" && i+1 < argc) { maxSteps = std::atoi(argv[++i]);
        } else {
            std::cerr << "Uso: wann_car_odin_eval [-c odin_config.json] "
                         "[-d config.json] [-n episodios] [-s seed] [-m window_ms] "
                         "[--csv] [--profile] [--steps N]\n";
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

    // In --csv mode every diagnostic line goes to stderr so stdout is pure
    // CSV for a caller to pipe into pandas.
    std::ostream& log = csvMode ? std::cerr : std::cout;

    try {
        auto cfg = wann::readOdinConfig(cfgFile);
        log << "Config ODIN cargada de " << cfgFile
            << " (" << cfg.nNeurons << " neuronas, "
            << cfg.synapses.size() << " sinapsis, run_key=" << cfg.runKey << ")\n";

        wann::hw::OdinDriver driver;
        wann::SnnCarOdinTask task(hyp, cfg, driver, windowMs);
        if (maxSteps > 0) task.setEpisodeSteps(maxSteps);

        log << "Evaluando " << nEpisodes << " episodios en hardware "
               "(seed base=" << seed << ")...\n\n";
        auto rewards = task.evalEpisodes(nEpisodes, seed);

        if (profile) {
            const auto& t = task.timings();
            double total = t.envUs + t.encodeUs + t.aerSendUs + t.aerDrainUs + t.decodeUs;
            auto pct = [&](double us) { return total > 0.0 ? 100.0 * us / total : 0.0; };
            log << std::fixed << std::setprecision(3);
            log << "\n--- Desglose de tiempo (" << t.nEnvSteps << " pasos de entorno, "
                << t.nTicks << " ticks internos) ---\n"
                << "Entorno RL (rl-tools observe/step/reward/terminated): "
                << t.envUs / 1e6 << " s  (" << std::setprecision(1) << pct(t.envUs) << "%)\n"
                << std::setprecision(3)
                << "Encoder (TTFS):                                      "
                << t.encodeUs / 1e6 << " s  (" << std::setprecision(1) << pct(t.encodeUs) << "%)\n"
                << std::setprecision(3)
                << "Envio AER (red -> ODIN, sendVirtual):                "
                << t.aerSendUs / 1e6 << " s  (" << std::setprecision(1) << pct(t.aerSendUs) << "%)\n"
                << std::setprecision(3)
                << "Espera/lectura AER_OUT (drainSpikes):                "
                << t.aerDrainUs / 1e6 << " s  (" << std::setprecision(1) << pct(t.aerDrainUs) << "%)\n"
                << std::setprecision(3)
                << "Decoder (RLDecoder):                                 "
                << t.decodeUs / 1e6 << " s  (" << std::setprecision(1) << pct(t.decodeUs) << "%)\n"
                << std::setprecision(3)
                << "Total medido:                                        " << total / 1e6 << " s\n"
                << "Promedio por tick (envio+espera AER):                "
                << (t.nTicks > 0 ? (t.aerSendUs + t.aerDrainUs) / t.nTicks : 0.0) << " us\n\n";
        }

        if (profile) {
            const auto& counts = task.network().addrSpikeCounts();
            log << "--- Spikes vistos en AER_OUT por direccion ---\n";
            bool any = false;
            for (int a = 0; a < 256; ++a)
                if (counts[a] > 0) { log << "  addr " << a << ": " << counts[a] << '\n'; any = true; }
            if (!any) log << "  (ninguno: ninguna neurona disparo)\n";
            log << "Direcciones de salida esperadas:";
            for (const auto& g : task.network().outputAddrs())
                for (int a : g) log << ' ' << a;
            log << "\n\n";
        }

        if (csvMode) {
            std::cout << "episode,reward\n" << std::fixed << std::setprecision(6);
            for (int i = 0; i < nEpisodes; ++i) std::cout << i << ',' << rewards[i] << '\n';
            return 0;
        }

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

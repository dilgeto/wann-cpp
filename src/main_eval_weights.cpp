// wann_eval_weights_{acrobot,car,disc_mc} – carga una red guardada y evalúa
// los N_WEIGHTS pesos compartidos para una lista de seeds. Imprime un CSV a
// stdout con una fila por seed; el promedio a través de seeds se calcula en
// el script Python que orquesta esta herramienta (ver eval_p3_weights.py).
//
// --reward shaped   (default) usa getDistFitness(), igual que en entrenamiento.
// --reward original usa la recompensa sin shaping. getDistFitness() sólo
//   expone la recompensa shaped, así que para "original" se reproduce el
//   mismo barrido (mismos pesos, mismos episodios/seeds — episodeSeed =
//   seed*10000 + wi*100 + rep, idéntico al de evaluate()/getDistFitness())
//   pero llamando a evalEpisodes() y promediando su componente original.
//
// La tarea concreta se fija en tiempo de compilación (una de
// EVAL_TASK_ACROBOT / EVAL_TASK_CAR / EVAL_TASK_DISC_MC) porque cada
// TaskXxx.cpp instancia símbolos no-inline de rl-tools que chocan (multiple
// definition) si dos de ellas se linkean en el mismo binario.
//
// Uso:
//   ./wann_eval_weights_acrobot -f log/.../rank00_seed00_best.out \
//       -d p/acrobot_snn.json [-p overrides.json] --seeds 0,1,2,3,4,5,6,7,8,9 \
//       [--reward shaped|original]

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Task.h"

#if defined(EVAL_TASK_ACROBOT)
    #include "../include/wann/SnnAcrobotTask.h"
    using EvalTask = wann::SnnAcrobotTask;
#elif defined(EVAL_TASK_CAR)
    #include "../include/wann/SnnCarTask.h"
    using EvalTask = wann::SnnCarTask;
#elif defined(EVAL_TASK_DISC_MC)
    #include "../include/wann/SnnDiscMCTask.h"
    using EvalTask = wann::SnnDiscMCTask;
#else
    #error "Define EVAL_TASK_ACROBOT, EVAL_TASK_CAR o EVAL_TASK_DISC_MC"
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int> parseSeeds(const std::string& csv) {
    std::vector<int> seeds;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) seeds.push_back(std::stoi(tok));
    }
    return seeds;
}

// Reproduce el barrido de getDistFitness() pero devuelve la recompensa
// original (sin shaping): mismos pesos, mismo número de repeticiones
// (nReps) y mismo esquema de seeds por episodio (seed*10000 + wi*100 + rep).
std::vector<double> getDistFitnessOriginal(EvalTask& task,
                                            const std::vector<double>& wVec,
                                            const std::vector<int>&    aVec,
                                            int seed, int nReps) {
    const int nW = task.numWeightVals();
    std::vector<double> rewards(nW, 0.0);
    for (int wi = 0; wi < nW; ++wi) {
        const int baseSeed = seed * 10000 + wi * 100;
        auto [shaped, original] = task.evalEpisodes(wVec, aVec,
                                                     EvalTask::WEIGHT_VALS[wi],
                                                     nReps, baseSeed);
        double total = 0.0;
        for (double o : original) total += o;
        rewards[wi] = total / static_cast<double>(nReps);
    }
    return rewards;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string netFile, configFile, overrideFile, seedsArg;
    std::string rewardArg = "shaped";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f"      && i+1 < argc) { netFile      = argv[++i]; }
        else if (arg == "-d"      && i+1 < argc) { configFile   = argv[++i]; }
        else if (arg == "-p"      && i+1 < argc) { overrideFile = argv[++i]; }
        else if (arg == "--seeds" && i+1 < argc) { seedsArg     = argv[++i]; }
        else if (arg == "--reward"&& i+1 < argc) { rewardArg    = argv[++i]; }
        else {
            std::cerr << "Uso: " << argv[0]
                       << " -f red.out -d config.json [-p overrides.json] "
                          "--seeds s0,s1,... [--reward shaped|original]\n";
            return 1;
        }
    }

    if (netFile.empty() || configFile.empty() || seedsArg.empty()) {
        std::cerr << "Faltan argumentos requeridos (-f, -d, --seeds).\n";
        return 1;
    }
    if (rewardArg != "shaped" && rewardArg != "original") {
        std::cerr << "--reward debe ser 'shaped' u 'original', recibido: "
                   << rewardArg << '\n';
        return 1;
    }
    const bool useOriginal = (rewardArg == "original");

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(configFile);
        if (!overrideFile.empty()) wann::updateHyp(hyp, overrideFile);
    } catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    auto [wVec, aVec, wKey] = wann::importNet(netFile);

    EvalTask task(hyp);
    const int nW = task.numWeightVals();

    const auto seeds = parseSeeds(seedsArg);

    std::cout << "seed";
    for (int w = 0; w < nW; ++w) std::cout << ",w" << w;
    std::cout << '\n';

    std::cout << std::scientific;
    for (int seed : seeds) {
        auto rewards = useOriginal
            ? getDistFitnessOriginal(task, wVec, aVec, seed, hyp.alg_nReps)
            : task.getDistFitness(wVec, aVec, seed);
        std::cout << seed;
        for (double r : rewards) std::cout << ',' << r;
        std::cout << '\n';
    }

    return 0;
}

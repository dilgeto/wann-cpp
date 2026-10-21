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
// EVAL_TASK_ACROBOT / EVAL_TASK_CAR / EVAL_TASK_DISC_MC / EVAL_TASK_BIPEDAL) porque cada
// TaskXxx.cpp instancia símbolos no-inline de rl-tools que chocan (multiple
// definition) si dos de ellas se linkean en el mismo binario.
//
// Uso:
//   ./wann_eval_weights_acrobot -f log/.../rank00_seed00_best.out \
//       -d p/acrobot_snn.json [-p overrides.json] --seeds 0,1,2,3,4,5,6,7,8,9 \
//       [--reward shaped|original]
//
// --episode-detail --weight-index N: en vez del promedio por (seed, peso),
// imprime cada uno de los nReps episodios individuales que se promediaron
// para dar ese número, para el peso N y cada seed de --seeds. Sirve para
// inspeccionar qué episodios componen un valor "eval_seed_X" de *_best.csv
// (weight_index viene de esa misma fila). CSV: seed,episode,reward.
//
// --nreps N: sobreescribe hyp.alg_nReps SOLO en memoria para esta corrida
// (no toca p/*.json, que sigue rigiendo el entrenamiento). Como
// getDistFitness() usa el nReps_ interno de la tarea (copiado de hyp al
// construirla), el override aplica igual a --reward shaped, original y
// --episode-detail.
//
// --weights v0,v1,...: en vez de los N_WEIGHTS pesos compilados (WEIGHT_VALS),
// barre los valores dados (p. ej. pesos fuera de la grilla de entrenamiento
// para medir agnosticismo fuera de muestra). Misma salida CSV (seed,w0,w1,...,
// una columna por peso dado, en ese orden); mismo esquema de seeds por episodio
// (seed*10000 + k*100 + rep, k = posición en la lista). Respeta --reward.
//
// Ablaciones post-hoc (solo evaluación; transforman wVec/aVec ya cargados,
// no tocan el archivo ni el entrenamiento). Se aplican en este orden y se
// pueden combinar, aunque el uso previsto es una por corrida:
//   --ablate-inhibitory      w<0 -> 0 (se quitan las sinapsis inhibitorias).
//   --flip-inhibitory        w<0 -> +|w| (misma topología, todas excitatorias).
//   --drop-random-exc K      quita K sinapsis excitatorias al azar (control de
//                            --ablate-inhibitory con K = nº de inhibitorias).
//   --force-neuron-type N    todas las neuronas (entradas, bias, ocultas,
//                            salidas) pasan al tipo N (1..6 = ann_actRange).
//   --shuffle-types          permuta los tipos entre neuronas (mismo
//                            histograma de tipos, distinta asignación).
//   --ablate-seed S          semilla del RNG de los dos controles aleatorios.
// Un resumen de lo modificado se imprime en stderr.

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
#elif defined(EVAL_TASK_BIPEDAL)
    #include "../include/wann/SnnBipedalTask.h"
    using EvalTask = wann::SnnBipedalTask;
#else
    #error "Define EVAL_TASK_ACROBOT, EVAL_TASK_CAR, EVAL_TASK_DISC_MC o EVAL_TASK_BIPEDAL"
#endif

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
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

struct AblationOpts {
    bool ablateInhibitory = false;
    bool flipInhibitory   = false;
    int  dropRandomExc    = -1;   // < 0 = off
    int  forceNeuronType  = -1;   // < 0 = off
    bool shuffleTypes     = false;
    unsigned ablateSeed   = 0;
    bool any() const {
        return ablateInhibitory || flipInhibitory || dropRandomExc >= 0 ||
               forceNeuronType >= 0 || shuffleTypes;
    }
};

// Aplica las ablaciones sobre la red cargada. wVec es N×N disperso (0 = sin
// conexión); el signo es la polaridad (>0 excitatoria), igual que lo lee
// buildNetwork(). Devuelve false (con mensaje) si algún parámetro es inválido.
bool applyAblation(const AblationOpts& o, std::vector<double>& wVec,
                   std::vector<int>& aVec) {
    int nExc = 0, nInh = 0;
    for (double w : wVec) { if (w > 0.0) ++nExc; else if (w < 0.0) ++nInh; }
    std::cerr << "[ablation] original: exc=" << nExc << " inh=" << nInh
              << " neuronas=" << aVec.size() << '\n';

    if (o.ablateInhibitory) {
        for (double& w : wVec) if (w < 0.0) w = 0.0;
        std::cerr << "[ablation] inhibitorias eliminadas: " << nInh << '\n';
    }
    if (o.flipInhibitory) {
        for (double& w : wVec) if (w < 0.0) w = -w;
        std::cerr << "[ablation] inhibitorias invertidas a excitatorias: "
                  << nInh << '\n';
    }
    std::mt19937 rng(o.ablateSeed);
    if (o.dropRandomExc >= 0) {
        std::vector<int> exc;
        for (int k = 0; k < static_cast<int>(wVec.size()); ++k)
            if (wVec[k] > 0.0) exc.push_back(k);
        if (o.dropRandomExc > static_cast<int>(exc.size())) {
            std::cerr << "--drop-random-exc " << o.dropRandomExc
                      << " > excitatorias disponibles (" << exc.size() << ").\n";
            return false;
        }
        std::shuffle(exc.begin(), exc.end(), rng);
        for (int k = 0; k < o.dropRandomExc; ++k) wVec[exc[k]] = 0.0;
        std::cerr << "[ablation] excitatorias eliminadas al azar: "
                  << o.dropRandomExc << '\n';
    }
    if (o.forceNeuronType >= 0) {
        if (o.forceNeuronType < 1 || o.forceNeuronType > 6) {
            std::cerr << "--force-neuron-type debe estar en 1..6.\n";
            return false;
        }
        std::fill(aVec.begin(), aVec.end(), o.forceNeuronType);
        std::cerr << "[ablation] todas las neuronas -> tipo "
                  << o.forceNeuronType << '\n';
    }
    if (o.shuffleTypes) {
        std::shuffle(aVec.begin(), aVec.end(), rng);
        std::cerr << "[ablation] tipos de neurona permutados\n";
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string netFile, configFile, overrideFile, seedsArg;
    std::string rewardArg = "shaped";
    bool episodeDetail = false;
    int weightIndex = -1;
    int nRepsOverride = -1;
    std::string weightsArg;
    AblationOpts abl;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f"            && i+1 < argc) { netFile      = argv[++i]; }
        else if (arg == "-d"            && i+1 < argc) { configFile   = argv[++i]; }
        else if (arg == "-p"            && i+1 < argc) { overrideFile = argv[++i]; }
        else if (arg == "--seeds"       && i+1 < argc) { seedsArg     = argv[++i]; }
        else if (arg == "--reward"      && i+1 < argc) { rewardArg    = argv[++i]; }
        else if (arg == "--episode-detail")             { episodeDetail = true; }
        else if (arg == "--weight-index"&& i+1 < argc) { weightIndex  = std::stoi(argv[++i]); }
        else if (arg == "--nreps"       && i+1 < argc) { nRepsOverride = std::stoi(argv[++i]); }
        else if (arg == "--weights"     && i+1 < argc) { weightsArg   = argv[++i]; }
        else if (arg == "--ablate-inhibitory")          { abl.ablateInhibitory = true; }
        else if (arg == "--flip-inhibitory")            { abl.flipInhibitory   = true; }
        else if (arg == "--drop-random-exc" && i+1 < argc) { abl.dropRandomExc = std::stoi(argv[++i]); }
        else if (arg == "--force-neuron-type" && i+1 < argc) { abl.forceNeuronType = std::stoi(argv[++i]); }
        else if (arg == "--shuffle-types")              { abl.shuffleTypes     = true; }
        else if (arg == "--ablate-seed" && i+1 < argc)  { abl.ablateSeed = static_cast<unsigned>(std::stoul(argv[++i])); }
        else {
            std::cerr << "Uso: " << argv[0]
                       << " -f red.out -d config.json [-p overrides.json] "
                          "--seeds s0,s1,... [--reward shaped|original] "
                          "[--episode-detail --weight-index N] [--nreps N] [--weights v0,v1,...] "
                          "[--ablate-inhibitory|--flip-inhibitory|--drop-random-exc K|"
                          "--force-neuron-type N|--shuffle-types] [--ablate-seed S]\n";
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
    if (episodeDetail && weightIndex < 0) {
        std::cerr << "--episode-detail requiere --weight-index N (>= 0).\n";
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
    if (nRepsOverride > 0) hyp.alg_nReps = nRepsOverride;

    auto [wVec, aVec, wKey] = wann::importNet(netFile);
    if (abl.any() && !applyAblation(abl, wVec, aVec)) return 1;

    EvalTask task(hyp);
    const int nW = task.numWeightVals();
    const auto seeds = parseSeeds(seedsArg);

    if (episodeDetail) {
        if (weightIndex >= nW) {
            std::cerr << "--weight-index " << weightIndex
                       << " fuera de rango (0.." << (nW - 1) << ").\n";
            return 1;
        }
        std::cout << "seed,episode,reward\n";
        std::cout << std::scientific;
        for (int seed : seeds) {
            const int baseSeed = seed * 10000 + weightIndex * 100;
            auto [shaped, original] = task.evalEpisodes(wVec, aVec,
                                                         EvalTask::WEIGHT_VALS[weightIndex],
                                                         hyp.alg_nReps, baseSeed);
            const auto& rewards = useOriginal ? original : shaped;
            for (int ep = 0; ep < static_cast<int>(rewards.size()); ++ep)
                std::cout << seed << ',' << ep << ',' << rewards[ep] << '\n';
        }
        return 0;
    }

    if (!weightsArg.empty()) {
        std::vector<double> wvals;
        {
            std::stringstream ss(weightsArg);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) wvals.push_back(std::stod(tok));
        }
        std::cout << "seed";
        for (size_t k = 0; k < wvals.size(); ++k) std::cout << ",w" << k;
        std::cout << '\n' << std::scientific;
        for (int seed : seeds) {
            std::cout << seed;
            for (size_t k = 0; k < wvals.size(); ++k) {
                auto [shaped, original] = task.evalEpisodes(
                    wVec, aVec, wvals[k], hyp.alg_nReps,
                    seed * 10000 + static_cast<int>(k) * 100);
                const auto& rw = useOriginal ? original : shaped;
                double total = 0.0;
                for (double r : rw) total += r;
                std::cout << ',' << total / static_cast<double>(rw.size());
            }
            std::cout << '\n';
        }
        return 0;
    }

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

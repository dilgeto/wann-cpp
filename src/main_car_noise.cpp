// wann_car_noise – evalúa la robustez de una red car ya entrenada a ruido de
// memristores (escritura/programación + lectura), en dos modos activables:
//
//   --mode shared       : un único valor ruidoso (muestreado del modelo de
//                         ruido elegido) aplicado a TODA la red por igual,
//                         exactamente como evalEpisodes()/getDistFitness() ya
//                         hacen hoy — el "chip" se trata como un solo
//                         memristor. Barato, no requiere tocar el simulador,
//                         pero no puede ver fragilidad a la dispersión
//                         sinapsis-a-sinapsis.
//   --mode per-synapse  : cada conexión habilitada recibe su propio valor
//                         ruidoso, muestreado de forma independiente — un
//                         memristor físico distinto por sinapsis.
//
// --noise-model none|simple|pcm|reram|reramcmo selecciona el modelo de
// memristor_noise.hpp (DeviceModel/PCMDeviceModel/ReRamWan2022DeviceModel/
// ReRamCMODeviceModel, valores por defecto del header; "none" no agrega ruido
// — sirve de chequeo de regresión: con --mode per-synapse --noise-model none
// el resultado debe coincidir EXACTAMENTE con --mode shared para el mismo
// --weight). "reramcmo" necesita además --reramcmo-g-max/--reramcmo-g-min
// (sin default confiable — ver memristor_noise.hpp).
//
// Solo evalúa episodios con una red ya entrenada — nunca entrena.
//
// --noise-scale escala la intensidad del modelo elegido (0=sin ruido, 1=los
// valores por defecto del header, 2=el doble). --prog-noise-scale/
// --read-noise-scale/--drift-scale la desglosan por mecanismo para ablación
// uno-a-la-vez (si no se pasan, caen en --noise-scale). ReRAM (Wan2022) no
// separa programación de deriva/lectura en su modelo — un solo valor para
// los tres; ReRamCMO sí los separa, como PCM.
//
// Cada fila también imprime weight_rmse: el RMSE de los pesos realmente
// realizados (tras el ruido) contra los nominales, normalizado por w_max —
// a diferencia de noise_scale, es la MISMA unidad en los cuatro modelos, así
// que sirve de eje común para comparar simple/pcm/reram/reramcmo entre sí.
//
// --n-levels N: cuantiza la conductancia a N niveles distinguibles al
// programar (<=1 = continuo, de siempre). No es parte de ningún dato de
// Nandakumar/Wan — mecanismo adicional para estudiar el efecto de una
// resolución de programación finita (par diferencial, cada rama por
// separado).
//
// Uso:
//   ./wann_car_noise -f red.out -d p/car_snn.json [-p overrides.json] \
//       (--weight-index N | --weight VAL) \
//       [--mode shared|per-synapse] [--noise-model none|simple|pcm|reram|reramcmo] \
//       [--n-chips N] [--reads-per-chip N] [--t-inference t1,t2,...] \
//       [--w-max V] [--noise-scale F] \
//       [--prog-noise-scale F] [--read-noise-scale F] [--drift-scale F] \
//       [--n-levels N] [--reramcmo-g-max V] [--reramcmo-g-min V] \
//       --seeds s0,s1,...

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnCarTask.h"
#include "memristor_noise.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int> parseInts(const std::string& csv) {
    std::vector<int> v;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

std::vector<double> parseDoubles(const std::string& csv) {
    std::vector<double> v;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stod(tok));
    return v;
}

double meanOf(const std::vector<double>& v) {
    double total = 0.0;
    for (double x : v) total += x;
    return v.empty() ? 0.0 : total / static_cast<double>(v.size());
}

// Conexiones habilitadas de wVec (layout N×N disperso), en el MISMO orden
// (i,j) que SnnCarTask::buildNetwork(wVec,aVec) usa para llamar addSynapse —
// debe mantenerse en sync con esa función si su bucle cambia. 'positions'
// recibe el índice plano i*N+j de cada conexión; el valor devuelto es el peso
// nominal con signo (+weightVal si wVec[i*N+j] > 0, -weightVal si < 0).
std::vector<double> nominalConnections(const std::vector<double>& wVec, double weightVal,
                                        std::vector<int>& positions) {
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    std::vector<double> nominal;
    positions.clear();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const double w = wVec[i * N + j];
            if (w != 0.0) {
                nominal.push_back(w > 0.0 ? weightVal : -weightVal);
                positions.push_back(i * N + j);
            }
        }
    return nominal;
}

// Dispersa noisyCompact (misma indexación que nominalConnections/positions)
// de vuelta a un vector N×N para alimentar
// SnnCarTask::evalEpisodesPerSynapse().
std::vector<double> scatter(const std::vector<double>& noisyCompact,
                             const std::vector<int>& positions, size_t fullSize) {
    std::vector<double> out(fullSize, 0.0);
    for (size_t k = 0; k < positions.size(); ++k) out[positions[k]] = noisyCompact[k];
    return out;
}

struct Args {
    std::string netFile, configFile, overrideFile, seedsArg;
    std::string mode = "shared";
    std::string noiseModel = "none";
    int weightIndex = -1;
    double weight = -1.0;
    int nChips = 100;
    int readsPerChip = 1;
    std::string tInferenceArg;
    double wMax = -1.0;  // -1 = sin especificar -> default más abajo
    double noiseScale = 1.0;
    // Sentinela -1 = "no se pasó esta flag, usar noiseScale". Permiten
    // ablación mecanismo-por-mecanismo (p. ej. --read-noise-scale 0
    // --drift-scale 0 para medir SOLO ruido de programación/D2D).
    double progNoiseScale = -1.0;
    double readNoiseScale = -1.0;
    double driftScaleArg  = -1.0;
    int nLevels = 0;  // <=1 (o 0) = conductancia continua, de siempre
    // reramcmo no tiene default confiable en aihwkit (ver memristor_noise.hpp)
    // — estos SÍ son valores de este binario, no de Falcone et al.; pasar los
    // propios si se tienen mediciones reales.
    double reramcmoGMax = 25.0;
    double reramcmoGMin = 1.0;
};

void printUsage(const char* argv0) {
    std::cerr << "Uso: " << argv0
              << " -f red.out -d config.json [-p overrides.json] "
                 "(--weight-index N | --weight VAL) "
                 "[--mode shared|per-synapse] [--noise-model none|simple|pcm|reram|reramcmo] "
                 "[--n-chips N] [--reads-per-chip N] [--t-inference t1,t2,...] "
                 "[--w-max V] [--noise-scale F] "
                 "[--prog-noise-scale F] [--read-noise-scale F] [--drift-scale F] "
                 "[--n-levels N] [--reramcmo-g-max V] [--reramcmo-g-min V] "
                 "--seeds s0,s1,...\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f"               && i+1 < argc) { a.netFile      = argv[++i]; }
        else if (arg == "-d"               && i+1 < argc) { a.configFile   = argv[++i]; }
        else if (arg == "-p"               && i+1 < argc) { a.overrideFile = argv[++i]; }
        else if (arg == "--seeds"          && i+1 < argc) { a.seedsArg     = argv[++i]; }
        else if (arg == "--mode"           && i+1 < argc) { a.mode         = argv[++i]; }
        else if (arg == "--noise-model"    && i+1 < argc) { a.noiseModel   = argv[++i]; }
        else if (arg == "--weight-index"   && i+1 < argc) { a.weightIndex  = std::stoi(argv[++i]); }
        else if (arg == "--weight"         && i+1 < argc) { a.weight       = std::stod(argv[++i]); }
        else if (arg == "--n-chips"        && i+1 < argc) { a.nChips       = std::stoi(argv[++i]); }
        else if (arg == "--reads-per-chip" && i+1 < argc) { a.readsPerChip = std::stoi(argv[++i]); }
        else if (arg == "--t-inference"    && i+1 < argc) { a.tInferenceArg = argv[++i]; }
        else if (arg == "--w-max"          && i+1 < argc) { a.wMax         = std::stod(argv[++i]); }
        else if (arg == "--noise-scale"    && i+1 < argc) { a.noiseScale   = std::stod(argv[++i]); }
        else if (arg == "--prog-noise-scale" && i+1 < argc) { a.progNoiseScale = std::stod(argv[++i]); }
        else if (arg == "--read-noise-scale" && i+1 < argc) { a.readNoiseScale = std::stod(argv[++i]); }
        else if (arg == "--drift-scale"      && i+1 < argc) { a.driftScaleArg  = std::stod(argv[++i]); }
        else if (arg == "--n-levels"         && i+1 < argc) { a.nLevels        = std::stoi(argv[++i]); }
        else if (arg == "--reramcmo-g-max"   && i+1 < argc) { a.reramcmoGMax   = std::stod(argv[++i]); }
        else if (arg == "--reramcmo-g-min"   && i+1 < argc) { a.reramcmoGMin   = std::stod(argv[++i]); }
        else { printUsage(argv[0]); return 1; }
    }

    if (a.netFile.empty() || a.configFile.empty() || a.seedsArg.empty()) {
        std::cerr << "Faltan argumentos requeridos (-f, -d, --seeds).\n";
        printUsage(argv[0]);
        return 1;
    }
    if (a.mode != "shared" && a.mode != "per-synapse") {
        std::cerr << "--mode debe ser 'shared' o 'per-synapse'.\n";
        return 1;
    }
    if (a.noiseModel != "none" && a.noiseModel != "simple" &&
        a.noiseModel != "pcm" && a.noiseModel != "reram" && a.noiseModel != "reramcmo") {
        std::cerr << "--noise-model debe ser 'none', 'simple', 'pcm', 'reram' o 'reramcmo'.\n";
        return 1;
    }
    if ((a.weightIndex < 0) == (a.weight < 0.0)) {
        std::cerr << "Pasar exactamente uno de --weight-index N o --weight VAL.\n";
        return 1;
    }

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(a.configFile);
        if (!a.overrideFile.empty()) wann::updateHyp(hyp, a.overrideFile);
    } catch (const std::exception& e) {
        std::cerr << "Error cargando config: " << e.what() << '\n';
        return 1;
    }

    auto [wVec, aVec, wKey] = wann::importNet(a.netFile);
    wann::SnnCarTask task(hyp);

    const double weightVal = (a.weightIndex >= 0)
        ? wann::SnnCarTask::WEIGHT_VALS[a.weightIndex]
        : a.weight;

    const double wMaxDefault = *std::max_element(
        wann::SnnCarTask::WEIGHT_VALS,
        wann::SnnCarTask::WEIGHT_VALS + wann::SnnCarTask::N_WEIGHTS);
    const double wMax = (a.wMax > 0.0) ? a.wMax : wMaxDefault;

    std::vector<double> tInference = {0.0};
    if (!a.tInferenceArg.empty()) {
        tInference = parseDoubles(a.tInferenceArg);
    } else if (a.noiseModel == "reram") {
        tInference = {1.0, 86400.0, 172800.0};
    } else if (a.noiseModel == "reramcmo") {
        // t_inference==0 es un bypass TOTAL en este modelo (ni deriva ni
        // ruido de lectura, ver memristor_noise.hpp) — 1s como default para
        // que el barrido de intensidad por defecto sí ejercite los tres
        // mecanismos, no solo programación.
        tInference = {1.0};
    }

    std::vector<int> positions;
    const std::vector<double> nominal = nominalConnections(wVec, weightVal, positions);

    // Lo que efectivamente se "programa en el chip": un solo valor (shared) o
    // uno por conexión habilitada (per-synapse). Fijo para toda la corrida;
    // lo único que cambia por chip es el ruido muestreado a partir de esto.
    const std::vector<double> baseWeights =
        (a.mode == "shared") ? std::vector<double>{weightVal} : nominal;

    const auto seeds = parseInts(a.seedsArg);

    // --noise-scale escala la intensidad de cada modelo (1.0 = valores por
    // defecto del header, ajustados por IBM/literatura; 0.0 = sin ruido;
    // 2.0 = el doble). --prog-noise-scale/--read-noise-scale/--drift-scale
    // (si se pasan) la desglosan por mecanismo, para ablación uno-a-la-vez —
    // si no se pasan, caen en --noise-scale (comportamiento de siempre).
    // ReRAM no distingue programación de deriva/lectura en su modelo (ver
    // memristor_noise.hpp): usa un solo progScale para las tres.
    const double progScale  = (a.progNoiseScale >= 0.0) ? a.progNoiseScale : a.noiseScale;
    const double readScale  = (a.readNoiseScale >= 0.0) ? a.readNoiseScale : a.noiseScale;
    const double driftScale = (a.driftScaleArg  >= 0.0) ? a.driftScaleArg  : a.noiseScale;

    memnoise::DeviceModel dmSimple;
    dmSimple.c0 *= progScale;
    dmSimple.c1 *= progScale;
    dmSimple.c2 *= progScale;
    dmSimple.r0 *= readScale;
    dmSimple.n_levels = a.nLevels;

    memnoise::PCMDeviceModel dmPcm;
    dmPcm.prog_noise_scale = progScale;
    dmPcm.read_noise_scale = readScale;
    dmPcm.drift_scale      = driftScale;
    dmPcm.n_levels         = a.nLevels;

    memnoise::ReRamWan2022DeviceModel dmReram;
    dmReram.noise_scale = progScale;
    dmReram.n_levels    = a.nLevels;

    memnoise::ReRamCMODeviceModel dmReramCmo;
    dmReramCmo.g_max = a.reramcmoGMax;
    dmReramCmo.g_min = a.reramcmoGMin;
    dmReramCmo.prog_noise_scale = progScale;
    dmReramCmo.read_noise_scale = readScale;
    dmReramCmo.drift_scale      = driftScale;
    dmReramCmo.n_levels         = a.nLevels;

    std::cout << "seed,chip,read,t_inference,mode,noise_model,weight,"
                 "prog_noise_scale,read_noise_scale,drift_scale,weight_rmse,"
                 "reward_shaped,reward_original\n";
    std::cout << std::scientific;

    for (int seed : seeds) {
        for (int chip = 0; chip < a.nChips; ++chip) {
            // Un RNG propio por (seed,chip): cada "chip" es una instancia de
            // programación independiente; (seed,chip) determina su semilla de
            // forma reproducible.
            std::mt19937_64 rng(static_cast<uint64_t>(seed) * 1000003ULL +
                                 static_cast<uint64_t>(chip));

            // Los cuatro crossbars se construyen siempre (barato: solo arma
            // vectores, no consume RNG) pero solo el del modelo elegido llega
            // a usarse (program()/read_weights()) — así no hay que repetir el
            // bucle de lectura cinco veces, uno por modelo.
            memnoise::NoisyCrossbar             simple(baseWeights, wMax, dmSimple);
            memnoise::PCMNoisyCrossbar           pcm(baseWeights, wMax, dmPcm);
            memnoise::ReRamWan2022NoisyCrossbar  reram(baseWeights, wMax, dmReram);
            memnoise::ReRamCMONoisyCrossbar      reramcmo(baseWeights, wMax, dmReramCmo);

            // readFn(t_inference) -> ruido de LECTURA fresco (nuevo cada
            // llamada); el ruido de ESCRITURA/los coeficientes de deriva se
            // fijaron una vez arriba, vía program(), para este chip.
            std::function<std::vector<double>(double)> readFn;
            if (a.noiseModel == "simple") {
                simple.program(rng);
                readFn = [&](double) {
                    std::vector<double> out;
                    simple.read_weights(rng, out);
                    return out;
                };
            } else if (a.noiseModel == "pcm") {
                pcm.program(rng);
                readFn = [&](double t) {
                    std::vector<double> out;
                    pcm.read_weights(rng, t, out);
                    return out;
                };
            } else if (a.noiseModel == "reram") {
                reram.program(rng);
                readFn = [&](double t) {
                    std::vector<double> out;
                    reram.read_weights(rng, t, out);
                    return out;
                };
            } else if (a.noiseModel == "reramcmo") {
                reramcmo.program(rng);
                readFn = [&](double t) {
                    std::vector<double> out;
                    reramcmo.read_weights(rng, t, out);
                    return out;
                };
            } else {  // "none": sin ruido, chequeo de regresión contra evalEpisodes()
                readFn = [&](double) { return baseWeights; };
            }

            for (int read = 0; read < a.readsPerChip; ++read) {
                for (double t : tInference) {
                    // Espaciado amplio: deja hueco para hyp.alg_nReps episodios
                    // internos por fila (evalEpisodes/evalEpisodesPerSynapse
                    // suman baseSeed+i para i en [0, nReps)).
                    const long long baseSeed =
                        static_cast<long long>(seed) * 10000000LL +
                        static_cast<long long>(chip) * 10000LL +
                        static_cast<long long>(read) * 100LL;

                    const std::vector<double> noisy = readFn(t);

                    // Métrica común entre modelos: RMSE de los pesos
                    // realizados vs. los nominales, normalizado por w_max —
                    // a diferencia de noise_scale (que no es la misma unidad
                    // en simple/pcm/reram), esto sí es directamente
                    // comparable entre los tres.
                    double sqErrSum = 0.0;
                    for (size_t k = 0; k < noisy.size(); ++k) {
                        const double err = noisy[k] - baseWeights[k];
                        sqErrSum += err * err;
                    }
                    const double weightRmse =
                        std::sqrt(sqErrSum / static_cast<double>(noisy.size())) / wMax;

                    double rewardShaped, rewardOriginal;
                    if (a.mode == "shared") {
                        auto [shaped, original] = task.evalEpisodes(
                            wVec, aVec, std::fabs(noisy[0]), hyp.alg_nReps, baseSeed);
                        rewardShaped = meanOf(shaped);
                        rewardOriginal = meanOf(original);
                    } else {
                        const auto synapseMagnitudes = scatter(noisy, positions, wVec.size());
                        auto [shaped, original] = task.evalEpisodesPerSynapse(
                            wVec, aVec, synapseMagnitudes, hyp.alg_nReps, baseSeed);
                        rewardShaped = meanOf(shaped);
                        rewardOriginal = meanOf(original);
                    }

                    std::cout << seed << ',' << chip << ',' << read << ',' << t << ','
                               << a.mode << ',' << a.noiseModel << ',' << weightVal << ','
                               << progScale << ',' << readScale << ',' << driftScale << ','
                               << weightRmse << ','
                               << rewardShaped << ',' << rewardOriginal << '\n';
                }
            }
        }
    }

    return 0;
}

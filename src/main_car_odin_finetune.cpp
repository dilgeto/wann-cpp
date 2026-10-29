// wann_car_odin_finetune – ajuste fino POST-entrenamiento de los pesos de una red
// car ya evolucionada, restringido a lo que ODIN puede almacenar: una magnitud
// entera de 3 bits (niveles 0..7) por sinapsis.
//
// Herramienta de búsqueda local sobre una red guardada (*.out + .wi); no toca
// ni reentrena la topología ni el algoritmo evolutivo (Wann.cpp). Sólo
// simula episodios con SnnCarTask::evalEpisodesPerSynapse.
//
// Espacio de búsqueda: un nivel entero L[k] en 0..levels-1 por conexión
// habilitada. El signo de cada conexión no cambia (lo da wVec / excitatory;
// ODIN sólo admite un signo por neurona presináptica y el export ya separa
// los nodos de signo mixto — como los signos no se tocan, la cuenta de
// neuronas del export tampoco cambia). Nivel 0 = sinapsis desconectada.
// El peso simulado del nivel L es L * w_max / (levels-1); con el w_max por
// defecto (levels-1 = 7) el peso simulado es el propio nivel entero 0..7.
// OJO: wann_car_odin_export usa otra escala (quantizeMagnitude3bit divide por 8
// y multiplica por 7), así que sus niveles no coinciden con los de aquí.
//
// Punto de partida: todas las sinapsis en el nivel de quantizeMagnitude3bit()
// del peso compartido ganador — la red que hoy exporta wann_car_odin_export.
//
// Algoritmo ((1+lambda) con aceptación sólo estricta):
//   1. Cada iteración genera --lambda hijos del padre, cada uno con 1..--max-mut
//      sinapsis cambiadas (+-1 nivel, a 0, o a un nivel al azar).
//   2. Cribado: padre e hijos se simulan con las MISMAS semillas (pareado) —
//      semillas nuevas en cada iteración, para no sobreajustar a un conjunto fijo.
//   3. Confirmación: los --confirm-top mejores del cribado con Δ>0 se
//      re-evalúan junto al padre con semillas independientes del cribado
//      (--confirm-eps episodios pareados). Se acepta uno sólo si pasa DOS
//      pruebas, igual que wann_car_neighborhood: t-test de una cola con
//      p < --alpha/K, y efecto >= --min-effect (fracción del fitness del padre).
//      Elegir el mejor de lambda hijos con una muestra ruidosa sobreestima su
//      ventaja (maldición del ganador); la confirmación con datos nuevos lo corrige.
//   4. Al final, validación con semillas held-out (--holdout-eps) de: el peso
//      compartido continuo original, la red cuantizada uniforme de partida, el
//      mejor nivel uniforme 1..levels-1 y la red ajustada.
//
// Uso:
//   ./wann_car_odin_finetune -f log/.../rank00_seed00_best.out -d p/car_snn.json \
//       [-p overrides.json] [-w weight_index] [-o prefix] [--levels 8] \
//       [--w-max 7] [--init-level L] [--iters 200] [--lambda 16] [--max-mut 3] \
//       [--screen-eps N] [--confirm-eps 10] [--confirm-top 2] [--alpha 0.05] \
//       [--min-effect 0.002] [--patience 50] [--holdout-eps 50] \
//       [--seed-offset S] [--rng-seed S]
//
//   -d/-p deben ser los MISMOS que en el entrenamiento (ventana, encoder,
//   decoder, ...); si no, se afina otra red de la que se evaluó.
//
// Salidas (prefix por defecto: log/odin_ft_<archivo sin extensión>):
//   <prefix>_levels.json  nivel 0..7 de cada conexión (src,dst del layout N×N
//                         de wVec, excitatory, level); se reescribe en cada
//                         aceptación, así que una corrida interrumpida deja lo
//                         último aceptado.
//   <prefix>_trace.csv    una fila por iteración.

#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/SnnCarTask.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using wann::SnnCarTask;

namespace {

// --- Student's t survival function P(T > t) ---------------------------------
// Copia de la de main_car_neighborhood.cpp (allí es estática de ese ejecutable).
double betacf(double a, double b, double x) {
    const int MAXIT = 200;
    const double EPS = 3e-12, FPMIN = 1e-300;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;  if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;  if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < EPS) break;
    }
    return h;
}

double regularizedIncompleteBeta(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double logBt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                 + a * std::log(x) + b * std::log(1.0 - x);
    double bt = std::exp(logBt);
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double tSurvival(double t, double df) {
    if (df <= 0.0) return 0.5;
    if (t < 0.0) return 1.0 - tSurvival(-t, df);
    double x = df / (df + t * t);
    return 0.5 * regularizedIncompleteBeta(df / 2.0, 0.5, x);
}

double meanOf(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double sdOf(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = meanOf(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

// Prueba pareada de una cola H1: mean(a-b) > 0. Devuelve la media de la
// diferencia y p.
struct PairedTest { double meanDiff; double p; };
PairedTest pairedTest(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> d(a.size());
    for (size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    const double m = meanOf(d);
    const double se = sdOf(d) / std::sqrt(static_cast<double>(d.size()));
    if (d.size() < 2) return {m, 1.0};
    if (se == 0.0) return {m, m > 0.0 ? 0.0 : 1.0};
    return {m, tSurvival(m / se, static_cast<double>(d.size() - 1))};
}

// --- Red ---------------------------------------------------------------------
struct Conn { int pos, src, dst; bool exc; };

// Conexiones habilitadas, en el MISMO orden (i,j) en que
// SnnCarTask::buildNetwork(wVec,aVec) llama addSynapse (igual que
// nominalConnections() de main_car_noise.cpp — mantener en sync).
std::vector<Conn> enabledConnections(const std::vector<double>& wVec) {
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    std::vector<Conn> conns;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const double w = wVec[i * N + j];
            if (w != 0.0) conns.push_back({i * N + j, i, j, w > 0.0});
        }
    return conns;
}

struct Quant {
    int    levels;
    double wMax;
    double weightOf(int level) const { return level * wMax / (levels - 1); }
    // Misma regla que quantizeMagnitude3bit() de main_odin_export.cpp.
    int levelOf(double w) const {
        const double frac = std::clamp(std::abs(w) / wMax, 0.0, 1.0);
        return static_cast<int>(std::lround(frac * (levels - 1)));
    }
};

// Dispersa los niveles de las conexiones a un vector N×N de magnitudes para
// evalEpisodesPerSynapse().
std::vector<double> magnitudes(const std::vector<int>& lv, const std::vector<Conn>& conns,
                               size_t fullSize, const Quant& q) {
    std::vector<double> out(fullSize, 0.0);
    for (size_t k = 0; k < conns.size(); ++k) out[conns[k].pos] = q.weightOf(lv[k]);
    return out;
}

// rewards[c][e]: episodio e (semilla baseSeed+e) del candidato c. Todos los
// candidatos usan las mismas semillas -> comparables de forma pareada. El
// paralelismo es por (candidato, episodio).
std::vector<std::vector<double>> evalBatch(const SnnCarTask& task,
                                           const std::vector<double>& wVec,
                                           const std::vector<int>& aVec,
                                           const std::vector<std::vector<double>>& mags,
                                           int nEps, int baseSeed) {
    const int nC = static_cast<int>(mags.size());
    std::vector<std::vector<double>> r(nC, std::vector<double>(nEps, 0.0));
    const long total = static_cast<long>(nC) * nEps;
    #pragma omp parallel for schedule(dynamic)
    for (long idx = 0; idx < total; ++idx) {
        const int c = static_cast<int>(idx / nEps);
        const int e = static_cast<int>(idx % nEps);
        r[c][e] = task.evalEpisodesPerSynapse(wVec, aVec, mags[c], 1, baseSeed + e).first[0];
    }
    return r;
}

void writeLevels(const std::string& path, const std::vector<int>& lv,
                 const std::vector<Conn>& conns, const Quant& q,
                 const std::string& netFile, int weightIndex, double sharedWeight, int nNodes) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("No puedo escribir: " + path);
    f << "{\n"
      << "  \"source_net\": \"" << netFile << "\",\n"
      << "  \"weight_index\": " << weightIndex << ",\n"
      << "  \"shared_weight\": " << sharedWeight << ",\n"
      << "  \"n_nodes\": " << nNodes << ",\n"
      << "  \"levels\": " << q.levels << ",\n"
      << "  \"w_max\": " << q.wMax << ",\n"
      << "  \"connections\": [\n";
    for (size_t k = 0; k < conns.size(); ++k)
        f << "    {\"src\": " << conns[k].src << ", \"dst\": " << conns[k].dst
          << ", \"excitatory\": " << (conns[k].exc ? "true" : "false")
          << ", \"level\": " << lv[k] << "}" << (k + 1 < conns.size() ? "," : "") << "\n";
    f << "  ]\n}\n";
}

struct Args {
    std::string netFile, configFile, overrideFile, prefix;
    int    weightIndex = -1;
    int    levels = 8;
    double wMax = -1.0;       // -1 = levels-1 (peso simulado = nivel)
    int    initLevel = -1;    // -1 = nivel del peso compartido ganador
    int    iters = 200;
    int    lambda = 16;
    int    maxMut = 3;
    int    screenEps = -1;    // -1 = hyp.alg_nReps
    int    confirmEps = 10;
    int    confirmTop = 2;
    double alpha = 0.05;
    double minEffect = 0.002;
    int    patience = 50;     // 0 = apagada
    int    holdoutEps = 50;
    int    seedOffset = 0;
    uint64_t rngSeed = 1;
};

void printUsage(const char* argv0) {
    std::cerr << "Uso: " << argv0
              << " -f red.out -d config.json [-p overrides.json] [-w weight_index] "
                 "[-o prefix] [--levels 8] [--w-max V] [--init-level L] [--iters N] "
                 "[--lambda N] [--max-mut N] [--screen-eps N] [--confirm-eps N] "
                 "[--confirm-top K] [--alpha A] [--min-effect F] [--patience N] "
                 "[--holdout-eps N] [--seed-offset S] [--rng-seed S]\n";
}

// Rangos de semillas (int en evalEpisodesPerSynapse): cribado, confirmación y
// held-out no se solapan entre sí para ningún --iters razonable.
constexpr int SCREEN_BASE  = 100000000;
constexpr int CONFIRM_BASE = 500000000;
constexpr int HOLDOUT_BASE = 1000000000;
constexpr int ITER_STRIDE  = 1000;

}  // namespace

int main(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f"              && i+1 < argc) { a.netFile      = argv[++i]; }
        else if (arg == "-d"              && i+1 < argc) { a.configFile   = argv[++i]; }
        else if (arg == "-p"              && i+1 < argc) { a.overrideFile = argv[++i]; }
        else if (arg == "-o"              && i+1 < argc) { a.prefix       = argv[++i]; }
        else if (arg == "-w"              && i+1 < argc) { a.weightIndex  = std::stoi(argv[++i]); }
        else if (arg == "--levels"        && i+1 < argc) { a.levels       = std::stoi(argv[++i]); }
        else if (arg == "--w-max"         && i+1 < argc) { a.wMax         = std::stod(argv[++i]); }
        else if (arg == "--init-level"    && i+1 < argc) { a.initLevel    = std::stoi(argv[++i]); }
        else if (arg == "--iters"         && i+1 < argc) { a.iters        = std::stoi(argv[++i]); }
        else if (arg == "--lambda"        && i+1 < argc) { a.lambda       = std::stoi(argv[++i]); }
        else if (arg == "--max-mut"       && i+1 < argc) { a.maxMut       = std::stoi(argv[++i]); }
        else if (arg == "--screen-eps"    && i+1 < argc) { a.screenEps    = std::stoi(argv[++i]); }
        else if (arg == "--confirm-eps"   && i+1 < argc) { a.confirmEps   = std::stoi(argv[++i]); }
        else if (arg == "--confirm-top"   && i+1 < argc) { a.confirmTop   = std::stoi(argv[++i]); }
        else if (arg == "--alpha"         && i+1 < argc) { a.alpha        = std::stod(argv[++i]); }
        else if (arg == "--min-effect"    && i+1 < argc) { a.minEffect    = std::stod(argv[++i]); }
        else if (arg == "--patience"      && i+1 < argc) { a.patience     = std::stoi(argv[++i]); }
        else if (arg == "--holdout-eps"   && i+1 < argc) { a.holdoutEps   = std::stoi(argv[++i]); }
        else if (arg == "--seed-offset"   && i+1 < argc) { a.seedOffset   = std::stoi(argv[++i]); }
        else if (arg == "--rng-seed"      && i+1 < argc) { a.rngSeed      = std::stoull(argv[++i]); }
        else { printUsage(argv[0]); return 1; }
    }
    if (a.netFile.empty() || a.configFile.empty()) {
        std::cerr << "Faltan argumentos requeridos (-f, -d).\n";
        printUsage(argv[0]);
        return 1;
    }
    if (a.levels < 2 || a.lambda < 1 || a.maxMut < 1 || a.confirmEps < 2 ||
        a.confirmTop < 1 || a.holdoutEps < 2) {
        std::cerr << "Valores inválidos: --levels>=2, --lambda>=1, --max-mut>=1, "
                     "--confirm-eps>=2, --confirm-top>=1, --holdout-eps>=2.\n";
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
    const int screenEps = (a.screenEps > 0) ? a.screenEps : hyp.alg_nReps;

    std::vector<double> wVec;
    std::vector<int>    aVec;
    try {
        auto [w, ac, wKey] = wann::importNet(a.netFile);
        wVec = std::move(w);
        aVec = std::move(ac);
        (void)wKey;
    } catch (const std::exception& e) {
        std::cerr << "Error cargando red: " << e.what() << '\n';
        return 1;
    }

    // Peso compartido ganador: -w o el .wi junto al .out (como wann_car_odin_export).
    int wi = a.weightIndex;
    if (wi < 0) {
        std::ifstream wif(a.netFile.substr(0, a.netFile.rfind('.')) + ".wi");
        if (!wif || !(wif >> wi)) {
            std::cerr << "No encontré el .wi junto a " << a.netFile << " y no se pasó -w.\n";
            return 1;
        }
    }
    if (wi < 0 || wi >= SnnCarTask::N_WEIGHTS) {
        std::cerr << "weight_index fuera de rango: " << wi << '\n';
        return 1;
    }
    const double sharedWeight = SnnCarTask::WEIGHT_VALS[wi];

    // Por defecto el peso simulado ES el nivel entero (0,1,...,7): w_max = levels-1.
    const Quant q{a.levels, (a.wMax > 0.0) ? a.wMax : static_cast<double>(a.levels - 1)};

    int initLevel = (a.initLevel >= 0) ? a.initLevel : q.levelOf(sharedWeight);
    if (initLevel < 1 && a.initLevel < 0) {
        // Un peso compartido < 0.5 cuantiza a nivel 0 (todas las sinapsis apagadas).
        // Partir de ahí no tiene sentido; se avisa y se usa 1.
        std::cout << "AVISO: el peso compartido " << sharedWeight << " cuantiza a nivel 0 "
                     "(red muerta); parto del nivel 1 (usa --init-level para fijarlo).\n";
        initLevel = 1;
    }
    if (initLevel >= q.levels) {
        std::cerr << "--init-level debe estar en [0," << q.levels - 1 << "].\n";
        return 1;
    }

    if (a.prefix.empty()) {
        std::string base = a.netFile;
        auto slash = base.find_last_of('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        base = base.substr(0, base.rfind('.'));
        a.prefix = "log/odin_ft_" + base;
    }

    SnnCarTask task(hyp);
    const std::vector<Conn> conns = enabledConnections(wVec);
    const int nNodes = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    const size_t full = wVec.size();
    if (conns.empty()) {
        std::cerr << "La red no tiene conexiones habilitadas.\n";
        return 1;
    }

    std::cout << "red: " << a.netFile << "  wi=" << wi << " (peso compartido " << sharedWeight
              << ")  conexiones=" << conns.size() << "  niveles=" << q.levels
              << "  w_max=" << q.wMax << "  nivel inicial=" << initLevel
              << " (peso sim " << q.weightOf(initLevel) << ")  hilos=" << omp_get_max_threads()
              << '\n';

    std::vector<int> parent(conns.size(), initLevel);
    const std::vector<int> start = parent;
    std::mt19937_64 rng(a.rngSeed);

    std::ofstream trace(a.prefix + "_trace.csv");
    if (!trace) {
        std::cerr << "No puedo escribir " << a.prefix << "_trace.csv (¿existe el directorio?)\n";
        return 1;
    }
    trace << "iter,parent_screen,best_screen_delta,n_confirmed_tested,accepted,"
             "conf_delta,conf_p,n_zero,n_changed_vs_start\n";
    writeLevels(a.prefix + "_levels.json", parent, conns, q, a.netFile, wi, sharedWeight, nNodes);

    auto mutate = [&](std::vector<int> lv) {
        // 1 + geométrica(0.5), truncado a maxMut: la mayoría cambia una sola sinapsis.
        int nMut = 1;
        while (nMut < a.maxMut && (rng() & 1ULL)) ++nMut;
        std::uniform_int_distribution<size_t> pick(0, lv.size() - 1);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int m = 0; m < nMut; ++m) {
            const size_t k = pick(rng);
            const int old = lv[k];
            int nw;
            const double r = u(rng);
            if (r < 0.8) {                         // paso de +-1 nivel
                nw = old + ((rng() & 1ULL) ? 1 : -1);
                if (nw < 0 || nw >= q.levels) nw = old + (old == 0 ? 1 : -1);
            } else if (r < 0.9) {                  // desconectar
                nw = 0;
            } else {                               // nivel al azar
                nw = static_cast<int>(rng() % static_cast<uint64_t>(q.levels));
            }
            if (nw == old) nw = (old + 1) % q.levels;
            lv[k] = nw;
        }
        return lv;
    };

    int sinceAccept = 0, nAccepted = 0;
    for (int it = 0; it < a.iters; ++it) {
        const int screenSeed  = SCREEN_BASE  + a.seedOffset + it * ITER_STRIDE;
        const int confirmSeed = CONFIRM_BASE + a.seedOffset + it * ITER_STRIDE;

        // [0] = padre, [1..lambda] = hijos.
        std::vector<std::vector<int>> pop;
        pop.push_back(parent);
        for (int c = 0; c < a.lambda; ++c) pop.push_back(mutate(parent));
        std::vector<std::vector<double>> mags;
        for (const auto& lv : pop) mags.push_back(magnitudes(lv, conns, full, q));

        const auto scr = evalBatch(task, wVec, aVec, mags, screenEps, screenSeed);
        const double parentScreen = meanOf(scr[0]);

        std::vector<int> order;
        for (int c = 1; c <= a.lambda; ++c)
            if (meanOf(scr[c]) - parentScreen > 0.0) order.push_back(c);
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            return meanOf(scr[x]) > meanOf(scr[y]);
        });
        if (static_cast<int>(order.size()) > a.confirmTop) order.resize(a.confirmTop);

        double bestDelta = -1e300;
        for (int c = 1; c <= a.lambda; ++c) bestDelta = std::max(bestDelta, meanOf(scr[c]) - parentScreen);

        bool accepted = false;
        double accDelta = 0.0, accP = 1.0;
        if (!order.empty()) {
            std::vector<std::vector<double>> cmags{mags[0]};
            for (int c : order) cmags.push_back(mags[c]);
            const auto cf = evalBatch(task, wVec, aVec, cmags, a.confirmEps, confirmSeed);
            const double parentConf = meanOf(cf[0]);
            const double alphaK = a.alpha / static_cast<double>(order.size());  // Bonferroni
            int bestK = -1;
            for (size_t k = 0; k < order.size(); ++k) {
                const PairedTest t = pairedTest(cf[k + 1], cf[0]);
                const bool significant = t.p < alphaK;
                const bool practical = t.meanDiff >= a.minEffect * std::max(std::abs(parentConf), 1e-9);
                if (significant && practical && (bestK < 0 || t.meanDiff > accDelta)) {
                    bestK = static_cast<int>(k);
                    accDelta = t.meanDiff;
                    accP = t.p;
                }
            }
            if (bestK >= 0) {
                parent = pop[order[bestK]];
                accepted = true;
                ++nAccepted;
                writeLevels(a.prefix + "_levels.json", parent, conns, q, a.netFile, wi,
                            sharedWeight, nNodes);
            }
        }

        int nZero = 0, nChanged = 0;
        for (size_t k = 0; k < parent.size(); ++k) {
            nZero += (parent[k] == 0);
            nChanged += (parent[k] != start[k]);
        }
        trace << it << ',' << parentScreen << ',' << bestDelta << ',' << order.size() << ','
              << (accepted ? 1 : 0) << ',' << accDelta << ',' << accP << ','
              << nZero << ',' << nChanged << '\n';
        trace.flush();

        std::cout << "it " << std::setw(4) << it << "  padre(cribado)=" << std::fixed
                  << std::setprecision(4) << parentScreen << "  mejorΔ=" << bestDelta
                  << "  a confirmar=" << order.size()
                  << (accepted ? "  ACEPTADO Δconf=" : "  -")
                  << (accepted ? std::to_string(accDelta) : std::string())
                  << "  sinapsis!=inicio=" << nChanged << "  en 0=" << nZero << '\n';

        sinceAccept = accepted ? 0 : sinceAccept + 1;
        if (a.patience > 0 && sinceAccept >= a.patience) {
            std::cout << "Sin mejoras confirmadas en " << a.patience
                      << " iteraciones seguidas; termino.\n";
            break;
        }
    }

    // --- Validación held-out ---------------------------------------------------
    // Mismas semillas para todas las configuraciones -> comparaciones pareadas.
    // Ninguna de estas semillas se usó en cribado ni confirmación.
    const int hSeed = HOLDOUT_BASE + a.seedOffset;
    struct Row { std::string name; std::vector<double> r; };
    std::vector<Row> rows;

    {   // Peso compartido continuo original (lo que el entrenamiento evaluó).
        std::vector<double> r(a.holdoutEps);
        #pragma omp parallel for schedule(dynamic)
        for (int e = 0; e < a.holdoutEps; ++e)
            r[e] = task.evalEpisodes(wVec, aVec, sharedWeight, 1, hSeed + e).first[0];
        rows.push_back({"compartido_continuo(w=" + std::to_string(sharedWeight) + ")", r});
    }
    std::vector<std::vector<double>> hm;
    std::vector<std::string> hn;
    hn.push_back("uniforme_inicial(L=" + std::to_string(initLevel) + ")");
    hm.push_back(magnitudes(start, conns, full, q));
    for (int L = 1; L < q.levels; ++L) {
        if (L == initLevel) continue;
        hn.push_back("uniforme(L=" + std::to_string(L) + ")");
        hm.push_back(magnitudes(std::vector<int>(conns.size(), L), conns, full, q));
    }
    hn.push_back("AJUSTADA");
    hm.push_back(magnitudes(parent, conns, full, q));
    const auto hr = evalBatch(task, wVec, aVec, hm, a.holdoutEps, hSeed);
    for (size_t c = 0; c < hr.size(); ++c) rows.push_back({hn[c], hr[c]});

    // Mejor nivel uniforme (excluye la fila continua y la ajustada).
    size_t bestUni = 1;
    for (size_t c = 1; c + 1 < rows.size(); ++c)
        if (meanOf(rows[c].r) > meanOf(rows[bestUni].r)) bestUni = c;
    const Row& tuned = rows.back();

    std::cout << "\n=== Held-out (" << a.holdoutEps << " episodios, semillas " << hSeed << "+) ===\n"
              << std::fixed << std::setprecision(4);
    for (const auto& row : rows)
        std::cout << std::left << std::setw(36) << row.name << " media=" << meanOf(row.r)
                  << "  sd=" << sdOf(row.r) << '\n';
    const PairedTest vsStart = pairedTest(tuned.r, rows[1].r);
    const PairedTest vsBest  = pairedTest(tuned.r, rows[bestUni].r);
    std::cout << "\najustada - uniforme_inicial : Δ=" << vsStart.meanDiff << "  p(una cola)=" << vsStart.p
              << "\najustada - mejor uniforme   : Δ=" << vsBest.meanDiff << "  p(una cola)=" << vsBest.p
              << "  (" << rows[bestUni].name << ")\n"
              << "aceptaciones: " << nAccepted << "   niveles: " << a.prefix << "_levels.json\n";
    return 0;
}

// wann_car_neighborhood – análisis de vecindad de un individuo WANN (tarea Car).
//
// Herramienta de EVALUACIÓN (no entrena): dado un individuo, enumera todos los
// genomas a una mutación de distancia (vecindad de orden 1, N1), los evalúa con
// el mismo simulador y las mismas semillas que el padre, y reporta cuáles
// mejoran, cuáles son neutros y cuáles empeoran. Sirve para responder si un
// individuo estancado es un óptimo local estricto (ningún vecino mejora) o si
// existen vecinos mejores que la evolución no encuentra.
//
// Entrada (-i):
//   *.json  snapshot de población escrito por wann_car con snapshot_interval > 0
//           (log/<prefix>_snap/gen_XXXXX.json). Trae genomas completos y las
//           semillas de entrenamiento, así que el padre se re-evalúa con la misma
//           semilla que tuvo en su generación.
//   otro    red guardada (*_best.out). Se reconstruye un genoma aproximado (ver
//           genomeFromNetFile en GenomeIO.h): sirve para corridas ya hechas, pero
//           las semillas de evaluación no coinciden con las del entrenamiento.
//
// Uso:
//   ./wann_car_neighborhood -i log/car_snap/gen_00400.json [-d p/car_snn.json]
//       [-p overrides.json] [-o prefix] [--who elite|top:K|idx:N,N|all]
//       [--seed S] [--max-per-op N] [--noise-seeds R] [--eps E]
//       [--n2-mids M] [--n2-per-mid S] [--n2-if-stuck]
//       [--climb K] [--rng-seed S] [--dry-run]
//
//   -d/-p deben ser los MISMOS que en el entrenamiento (operadores, actRange,
//   parámetros SNN); si no, la vecindad y las recompensas describen otra cosa.
//
//   --who         qué individuos del snapshot analizar (default elite = mayor
//                 fitness medio). Ignorado con una red *.out.
//   --seed S      fuerza la semilla base de evaluación (episodios de individuo i:
//                 S*10000+i) en vez de la del snapshot. Con *.out el default es 0.
//   --max-per-op  submuestrea N vecinos por operador (0 = todos). La columna
//                 `weight` del CSV (n_op/N) reescala las probabilidades para que
//                 P(mejora) siga siendo un estimador insesgado.
//   --noise-seeds re-evalúa el padre con R semillas distintas para estimar el
//                 ruido del simulador (default 4). Si no se da --eps, se usa
//                 eps = 2 * sd(fitness medio del padre entre semillas).
//   --eps E       umbral fijo de neutralidad: |d_mean| <= E es "neutro".
//   --n2-mids M   además de N1, muestrea M vecinos intermedios (con prob.
//                 proporcional a su probabilidad de aparecer por mutación) y
//                 evalúa --n2-per-mid de SUS vecinos (default 50; 0 = todos).
//                 Distingue mejoras alcanzables por un intermedio neutro de las
//                 que exigen cruzar un valle.
//   --n2-if-stuck sólo corre N2 si ningún vecino de N1 mejora.
//   --climb K     ascenso voraz: hasta K veces, pasa al mejor vecino que supere
//                 eps y repite. Cada paso re-evalúa con una semilla nueva, para
//                 no arrastrar la maldición del ganador entre pasos.
//   --dry-run     sólo enumera y cuenta vecinos por operador; no simula.
//
// Salidas (prefix por defecto: log/neighbors_<archivo>):
//   <prefix>_neighbors.csv  una fila por individuo evaluado (padre, vecino N1,
//                           nieto N2), con recompensa por peso compartido.
//   <prefix>_parents.csv    resumen por padre y paso.
//   <prefix>_ops.csv        desglose por operador.
//
// Recompensas: SnnCarTask::evaluateWeight(), la misma función que usa el
// entrenamiento; fitness = media sobre los N_WEIGHTS pesos compartidos.

#include "../include/wann/GenomeIO.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Neighborhood.h"
#include "../include/wann/Random.h"
#include "../include/wann/SnnCarTask.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace wann;

namespace {

constexpr int NOISE_SEED_STRIDE = 7919;    // separa bloques de episodios (cada uno usa < 10000 seeds)
constexpr int CLIMB_SEED_STRIDE = 104729;

struct Options {
    std::string input;
    std::string defaultHyp = "p/car_snn.json";
    std::string overrideHyp;
    std::string outPrefix;
    std::string who        = "elite";
    bool        haveSeed   = false;
    int         seed       = 0;
    int         maxPerOp   = 0;
    int         noiseSeeds = 4;
    bool        haveEps    = false;
    double      eps        = 0.0;
    int         n2Mids     = 0;
    int         n2PerMid   = 50;
    bool        n2IfStuck  = false;
    int         climb      = 0;
    uint32_t    rngSeed    = 1;
    bool        dryRun     = false;
};

struct Parent {
    Ind                 ind;
    std::string         label;
    int                 idx = 0;
    int                 evalSeed = 0;             // semilla base de este individuo
    std::vector<double> storedReward;             // del snapshot (puede estar vacío)
};

double meanOf(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
double maxOf(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

int countHidden(const Ind& ind) {
    int n = 0;
    for (const auto& nd : ind.nodes) n += (nd.type == 3);
    return n;
}

// Recompensa [individuo][peso] de cada genoma con la misma semilla `seed`.
// Mismo esquema de dos fases que evalPop() en main_car.cpp.
std::vector<std::vector<double>> evalInds(const std::vector<const Ind*>& inds,
                                          const SnnCarTask& task, int seed)
{
    const int n  = static_cast<int>(inds.size());
    const int nW = task.numWeightVals();

    std::vector<Network> templates(n);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
        templates[i] = task.buildNetwork(*inds[i]);

    std::vector<std::vector<double>> reward(n, std::vector<double>(nW, 0.0));
    const int total = n * nW;
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < total; ++idx) {
        const int i  = idx / nW;
        const int wi = idx % nW;
        reward[i][wi] = task.evaluateWeight(templates[i], wi, seed);
    }
    return reward;
}

// Submuestrea `cap` vecinos por operador (0 = sin tope). weight[i] = n_op/cap
// para los que quedan, 1 si no se recortó ese operador.
void capPerOp(std::vector<Neighbor>& nbs, int cap, std::vector<double>& weight) {
    weight.assign(nbs.size(), 1.0);
    if (cap <= 0) return;

    std::vector<std::vector<int>> byOp(N_MUT_OPS);
    for (int i = 0; i < static_cast<int>(nbs.size()); ++i)
        byOp[static_cast<int>(nbs[i].op)].push_back(i);

    std::vector<std::pair<int,double>> chosen;  // (índice original, peso)
    for (auto& idxs : byOp) {
        const int n = static_cast<int>(idxs.size());
        if (n <= cap) {
            for (int i : idxs) chosen.push_back({i, 1.0});
        } else {
            wann::shuffle(idxs);
            for (int k = 0; k < cap; ++k)
                chosen.push_back({idxs[k], static_cast<double>(n) / cap});
        }
    }
    std::sort(chosen.begin(), chosen.end());

    std::vector<Neighbor> kept;
    kept.reserve(chosen.size());
    std::vector<double> w;
    for (auto& [i, wt] : chosen) { kept.push_back(std::move(nbs[i])); w.push_back(wt); }
    nbs    = std::move(kept);
    weight = std::move(w);
}

// Índice del nodo/gen en formato legible para el CSV.
std::string mutDesc(const Neighbor& nb) {
    return std::string(mutOpName(nb.op));
}

struct OpAgg {
    int    n = 0, better = 0, neutral = 0, worse = 0, invalid = 0, same = 0;
    double pMass = 0.0, pBetter = 0.0;
    double bestD = -std::numeric_limits<double>::infinity();
};

class Analyzer {
public:
    Analyzer(const Options& o, const Hyperparams& h, const SnnCarTask& t,
             std::ofstream& nbCsv, std::ofstream& parCsv, std::ofstream& opsCsv)
        : opt(o), hyp(h), task(t), nbOut(nbCsv), parOut(parCsv), opsOut(opsCsv),
          nW(t.numWeightVals()) {}

    void run(const Parent& p);

private:
    const Options&     opt;
    const Hyperparams& hyp;
    const SnnCarTask&  task;
    std::ofstream&     nbOut;
    std::ofstream&     parOut;
    std::ofstream&     opsOut;
    int                nW;

    void writeRow(const std::string& label, int idx, int step, int order,
                  int mid, const std::string& midOp, double midD,
                  const std::string& op, int a, int b, int c,
                  double prob, double weight, bool exprOk, const Ind& ind,
                  const std::vector<double>& r, double parentMean, double parentMax);

    struct StepOut {
        int    bestIdx = -1;             // mejor vecino que supera eps (-1 = ninguno)
        Ind    bestInd;
        std::string bestOp;
        double bestD = 0.0;
        double parentMean = 0.0;
        double pBetter = 0.0;
        bool   stuck = true;
    };

    StepOut step(const Parent& p, const Ind& cur, int stepNo, int stepSeed,
                 double& eps, bool firstStep);
};

void Analyzer::writeRow(const std::string& label, int idx, int stepNo, int order,
                        int mid, const std::string& midOp, double midD,
                        const std::string& op, int a, int b, int c,
                        double prob, double weight, bool exprOk, const Ind& ind,
                        const std::vector<double>& r, double parentMean, double parentMax)
{
    const double m  = meanOf(r);
    const double mx = maxOf(r);
    nbOut << label << ',' << idx << ',' << stepNo << ',' << order << ','
          << mid << ',' << midOp << ',';
    if (order == 2) nbOut << midD; else nbOut << "";
    nbOut << ',' << op << ',' << a << ',' << b << ',' << c << ','
          << prob << ',' << weight << ',' << (exprOk ? 1 : 0) << ','
          << ind.nConns() << ',' << countHidden(ind) << ','
          << m << ',' << mx << ',' << (m - parentMean) << ',' << (mx - parentMax);
    for (double v : r) nbOut << ',' << v;
    nbOut << '\n';
}

Analyzer::StepOut Analyzer::step(const Parent& p, const Ind& cur, int stepNo, int stepSeed,
                                 double& eps, bool firstStep)
{
    StepOut out;

    // --- padre: recompensa de referencia con la semilla de este paso ---
    std::vector<double> parentR = opt.dryRun ? std::vector<double>(nW, 0.0)
                                             : evalInds({&cur}, task, stepSeed)[0];
    const double pMean = meanOf(parentR), pMax = maxOf(parentR);
    out.parentMean = pMean;

    double stored = std::numeric_limits<double>::quiet_NaN();
    if (firstStep && !opt.dryRun && !p.storedReward.empty() && p.storedReward.size() == parentR.size()) {
        stored = 0.0;
        for (size_t k = 0; k < parentR.size(); ++k)
            stored = std::max(stored, std::abs(parentR[k] - p.storedReward[k]));
    }

    // --- ruido del simulador y umbral de neutralidad (sólo en el primer paso) ---
    double noiseSd = std::numeric_limits<double>::quiet_NaN();
    if (firstStep) {
        if (opt.noiseSeeds > 0 && !opt.dryRun) {
            std::vector<double> means{pMean};
            for (int r = 1; r <= opt.noiseSeeds; ++r)
                means.push_back(meanOf(evalInds({&cur}, task, stepSeed + r * NOISE_SEED_STRIDE)[0]));
            const double mu = meanOf(means);
            double ss = 0.0;
            for (double v : means) ss += (v - mu) * (v - mu);
            noiseSd = std::sqrt(ss / static_cast<double>(means.size() - 1));
        }
        eps = opt.haveEps ? opt.eps : (std::isnan(noiseSd) ? 0.0 : 2.0 * noiseSd);
    }

    // --- vecindad N1 ---
    auto nbs = enumerateNeighbors(cur, hyp);
    std::vector<double> weight;
    capPerOp(nbs, opt.maxPerOp, weight);

    std::vector<const Ind*> ptrs;
    ptrs.reserve(nbs.size());
    for (const auto& nb : nbs) ptrs.push_back(&nb.ind);

    std::cout << "\n== " << p.label << "  paso " << stepNo << "  (semilla base " << stepSeed << ") ==\n"
              << std::fixed << std::setprecision(3)
              << "padre: ";
    if (!opt.dryRun) std::cout << "fit_mean=" << pMean << "  fit_max=" << pMax << "  ";
    std::cout << "conexiones=" << cur.nConns() << "  ocultos=" << countHidden(cur) << '\n';
    if (firstStep) {
        if (!std::isnan(stored))
            std::cout << "  paridad con recompensa guardada: max|dif|=" << std::setprecision(6)
                      << stored << std::setprecision(3)
                      << (stored < 1e-9 ? "  (reproduce el entrenamiento)" : "  (NO coincide: revisa -d/-p)") << '\n';
        if (!std::isnan(noiseSd))
            std::cout << "  ruido: sd(fit_mean) entre " << opt.noiseSeeds + 1 << " semillas=" << noiseSd << '\n';
        std::cout << "  eps (neutralidad) = " << eps << (opt.haveEps ? " (--eps)" : "") << '\n';
    }
    std::cout << "  vecinos N1: " << nbs.size() << "  (~"
              << nbs.size() * static_cast<size_t>(nW) * static_cast<size_t>(hyp.alg_nReps)
              << " episodios)\n";

    std::vector<std::vector<double>> rewards;
    if (!opt.dryRun) rewards = evalInds(ptrs, task, stepSeed);

    // --- clasificación y salida ---
    writeRow(p.label, p.idx, stepNo, 0, -1, "", 0.0, "parent", -1, -1, -1, 0.0, 1.0, true,
             cur, parentR, pMean, pMax);

    OpAgg agg[N_MUT_OPS];
    int invalid = 0;
    for (size_t i = 0; i < nbs.size(); ++i) {
        const auto& nb = nbs[i];
        auto& a = agg[static_cast<int>(nb.op)];
        ++a.n;
        a.pMass += nb.prob * weight[i];
        if (!nb.exprOk) { ++a.invalid; ++invalid; }
        if (opt.dryRun) continue;

        const double d = meanOf(rewards[i]) - pMean;
        a.bestD = std::max(a.bestD, d);
        // Rewards identical to the parent's at every shared weight: with a
        // paired seed the simulation is deterministic, so this means the
        // mutation did not change the behaviour at all (e.g. it touched a
        // silent or disconnected part of the network).
        if (rewards[i] == parentR) ++a.same;
        if (d > eps) {
            ++a.better;
            a.pBetter += nb.prob * weight[i];
            out.pBetter += nb.prob * weight[i];
            out.stuck = false;
            if (out.bestIdx < 0 || d > out.bestD) {
                out.bestIdx = static_cast<int>(i);
                out.bestD   = d;
                out.bestInd = nb.ind;
                out.bestOp  = mutDesc(nb);
            }
        } else if (d < -eps) ++a.worse;
        else                 ++a.neutral;

        writeRow(p.label, p.idx, stepNo, 1, static_cast<int>(i), "", 0.0, mutDesc(nb),
                 nb.a, nb.b, nb.c, nb.prob, weight[i], nb.exprOk, nb.ind, rewards[i], pMean, pMax);
    }

    std::cout << "  operador           n   mejores  neutros (idénticos)  peores   mejor_d   P(op)   P(mejora)\n";
    int tn = 0, tb = 0, tne = 0, tw = 0, ts = 0;
    double tp = 0.0;
    for (int k = 0; k < N_MUT_OPS; ++k) {
        const auto& a = agg[k];
        tn += a.n; tb += a.better; tne += a.neutral; tw += a.worse; ts += a.same; tp += a.pMass;
        std::cout << "  " << std::left << std::setw(16) << mutOpName(static_cast<MutOp>(k))
                  << std::right << std::setw(6) << a.n;
        if (opt.dryRun) std::cout << "      -        -                 -       -         -";
        else std::cout << std::setw(9) << a.better << std::setw(9) << a.neutral
                       << std::setw(6) << ("(" + std::to_string(a.same) + ")")
                       << std::setw(13) << a.worse << std::setw(10) << (a.n ? a.bestD : 0.0);
        std::cout << std::setprecision(4) << std::setw(9) << a.pMass;
        if (!opt.dryRun) std::cout << std::setw(11) << a.pBetter;
        std::cout << std::setprecision(3) << '\n';

        opsOut << p.label << ',' << p.idx << ',' << stepNo << ',' << mutOpName(static_cast<MutOp>(k)) << ','
               << a.n << ',' << a.better << ',' << a.neutral << ',' << a.same << ',' << a.worse << ',' << a.invalid << ','
               << (a.n && !opt.dryRun ? a.bestD : 0.0) << ',' << a.pMass << ',' << a.pBetter << '\n';
    }
    std::cout << "  total: " << tn << " vecinos, P(mutación produce un vecino válido)"
              << (opt.maxPerOp > 0 ? " ~ " : " = ") << std::setprecision(4) << tp << std::setprecision(3);
    if (!opt.dryRun) std::cout << ";  " << ts << " no cambian el comportamiento (recompensa idéntica al padre)";
    if (invalid) std::cout << ", " << invalid << " con ciclo (express falla)";
    std::cout << '\n';

    if (!opt.dryRun) {
        if (out.stuck)
            std::cout << "  => ningún vecino de N1 mejora (eps=" << eps << "): óptimo local estricto en N1\n";
        else
            std::cout << "  => " << tb << " vecinos mejoran; mejor: " << out.bestOp << " (d_mean=+" << out.bestD
                      << ");  P(mejora por mutación)=" << std::setprecision(5) << out.pBetter
                      << "  ~ 1 de cada " << std::setprecision(0) << 1.0 / out.pBetter << " mutaciones\n"
                      << std::setprecision(3);
    }

    parOut << p.label << ',' << p.idx << ',' << stepNo << ',' << stepSeed << ',' << pMean << ',' << pMax << ','
           << cur.nConns() << ',' << countHidden(cur) << ',' << noiseSd << ',' << eps << ','
           << (std::isnan(stored) ? -1.0 : stored) << ',' << tn << ',' << tb << ',' << tne << ',' << ts << ',' << tw << ','
           << invalid << ',' << (out.stuck ? 1 : 0) << ',' << out.pBetter << ','
           << (out.pBetter > 0 ? std::to_string(1.0 / out.pBetter) : std::string("inf")) << '\n';

    // --- vecindad N2 muestreada ---
    const bool wantN2 = opt.n2Mids > 0 && firstStep && !opt.dryRun && (!opt.n2IfStuck || out.stuck);
    if (wantN2 && !nbs.empty()) {
        // Intermedios: Efraimidis-Spirakis, proporcional a la probabilidad de aparecer.
        std::vector<std::pair<double,int>> keys;
        for (size_t i = 0; i < nbs.size(); ++i)
            if (nbs[i].prob > 0.0)
                keys.push_back({std::log(std::max(randDouble(), 1e-300)) / (nbs[i].prob * weight[i]),
                                static_cast<int>(i)});
        std::sort(keys.begin(), keys.end(), [](auto& x, auto& y) { return x.first > y.first; });
        const int nMids = std::min<int>(opt.n2Mids, static_cast<int>(keys.size()));

        std::vector<Neighbor> gc;
        std::vector<int>      gcMid;
        std::vector<double>   gcW, gcMidD;
        for (int m = 0; m < nMids; ++m) {
            const int mi = keys[m].second;
            const double midD = meanOf(rewards[mi]) - pMean;
            auto sub = enumerateNeighbors(nbs[mi].ind, hyp);
            std::vector<double> w;
            capPerOp(sub, 0, w);
            if (opt.n2PerMid > 0 && static_cast<int>(sub.size()) > opt.n2PerMid) {
                wann::shuffle(sub);
                const double scale = static_cast<double>(sub.size()) / opt.n2PerMid;
                sub.resize(opt.n2PerMid);
                w.assign(sub.size(), scale);
            }
            for (size_t k = 0; k < sub.size(); ++k) {
                sub[k].prob *= nbs[mi].prob;   // P(primer paso) * P(segundo paso)
                gc.push_back(std::move(sub[k]));
                gcMid.push_back(mi);
                gcW.push_back(w[k]);
                gcMidD.push_back(midD);
            }
        }

        std::vector<const Ind*> gptrs;
        for (const auto& g : gc) gptrs.push_back(&g.ind);
        std::cout << "  N2: " << nMids << " intermedios, " << gc.size() << " nietos (~"
                  << gc.size() * static_cast<size_t>(nW) * static_cast<size_t>(hyp.alg_nReps) << " episodios)\n";
        auto gr = evalInds(gptrs, task, stepSeed);

        int nBetter = 0, viaNeutral = 0, viaValley = 0;
        for (size_t k = 0; k < gc.size(); ++k) {
            const int mi = gcMid[k];
            const double d = meanOf(gr[k]) - pMean;
            if (d > eps) {
                ++nBetter;
                if (gcMidD[k] >= -eps) ++viaNeutral; else ++viaValley;
            }
            writeRow(p.label, p.idx, stepNo, 2, mi, mutOpName(nbs[mi].op), gcMidD[k], mutDesc(gc[k]),
                     gc[k].a, gc[k].b, gc[k].c, gc[k].prob, gcW[k], gc[k].exprOk, gc[k].ind,
                     gr[k], pMean, pMax);
        }
        std::cout << "  N2: " << nBetter << " de " << gc.size() << " nietos mejoran al padre (> eps): "
                  << viaNeutral << " vía intermedio neutro o mejor, " << viaValley
                  << " vía intermedio peor (valle)\n";
    }
    return out;
}

void Analyzer::run(const Parent& p) {
    double eps = 0.0;
    Ind cur = p.ind;
    for (int s = 0; s <= opt.climb; ++s) {
        const int stepSeed = p.evalSeed + s * CLIMB_SEED_STRIDE;
        auto so = step(p, cur, s, stepSeed, eps, s == 0);
        if (opt.dryRun) break;
        if (s == opt.climb) break;
        if (so.bestIdx < 0) {
            std::cout << "  ascenso: se detiene en el paso " << s << " (sin vecino que supere eps)\n";
            break;
        }
        std::cout << "  ascenso: paso " << s << " -> " << so.bestOp << " (d_mean=+" << so.bestD << ")\n";
        cur = std::move(so.bestInd);
    }
}

std::vector<int> parseIntList(const std::string& csv) {
    std::vector<int> v;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

void usage() {
    std::cerr <<
        "Usage: wann_car_neighborhood -i <snapshot.json | red_best.out> [-d default.json]\n"
        "       [-p overrides.json] [-o prefix] [--who elite|top:K|idx:N,N|all]\n"
        "       [--seed S] [--max-per-op N] [--noise-seeds R] [--eps E]\n"
        "       [--n2-mids M] [--n2-per-mid S] [--n2-if-stuck]\n"
        "       [--climb K] [--rng-seed S] [--dry-run]\n";
}

} // namespace

int main(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(1); }
            return argv[++i];
        };
        try {
            if      (arg == "-i")             opt.input       = next();
            else if (arg == "-d")             opt.defaultHyp  = next();
            else if (arg == "-p")             opt.overrideHyp = next();
            else if (arg == "-o")             opt.outPrefix   = next();
            else if (arg == "--who")          opt.who         = next();
            else if (arg == "--seed")         { opt.seed = std::stoi(next()); opt.haveSeed = true; }
            else if (arg == "--max-per-op")   opt.maxPerOp    = std::stoi(next());
            else if (arg == "--noise-seeds")  opt.noiseSeeds  = std::stoi(next());
            else if (arg == "--eps")          { opt.eps = std::stod(next()); opt.haveEps = true; }
            else if (arg == "--n2-mids")      opt.n2Mids      = std::stoi(next());
            else if (arg == "--n2-per-mid")   opt.n2PerMid    = std::stoi(next());
            else if (arg == "--n2-if-stuck")  opt.n2IfStuck   = true;
            else if (arg == "--climb")        opt.climb       = std::stoi(next());
            else if (arg == "--rng-seed")     opt.rngSeed     = static_cast<uint32_t>(std::stoul(next()));
            else if (arg == "--dry-run")      opt.dryRun      = true;
            else { usage(); return 1; }
        } catch (const std::exception& e) {
            std::cerr << "Argumento inválido para " << arg << ": " << e.what() << '\n';
            return 1;
        }
    }
    if (opt.input.empty()) { usage(); return 1; }

    Hyperparams hyp;
    try {
        hyp = loadHyp(opt.defaultHyp);
        if (!opt.overrideHyp.empty()) updateHyp(hyp, opt.overrideHyp);
    } catch (const std::exception& e) {
        std::cerr << "Error cargando hiperparámetros: " << e.what() << '\n';
        return 1;
    }
    wann::seedRng(opt.rngSeed);

    // --- cargar los padres ---
    std::vector<Parent> parents;
    const std::string base = fs::path(opt.input).filename().string();
    try {
        if (fs::path(opt.input).extension() == ".json") {
            PopSnapshot snap = loadPopSnapshot(opt.input);
            const int n = static_cast<int>(snap.pop.size());
            std::vector<int> pick;
            if (opt.who == "elite" || opt.who.rfind("top:", 0) == 0) {
                const int k = (opt.who == "elite") ? 1 : std::stoi(opt.who.substr(4));
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                    [&](int a, int b) { return snap.pop[a].fitness > snap.pop[b].fitness; });
                pick.assign(order.begin(), order.begin() + std::min(k, n));
            } else if (opt.who.rfind("idx:", 0) == 0) {
                pick = parseIntList(opt.who.substr(4));
            } else if (opt.who == "all") {
                pick.resize(n);
                std::iota(pick.begin(), pick.end(), 0);
            } else {
                std::cerr << "--who no reconocido: " << opt.who << '\n';
                return 1;
            }
            for (int i : pick) {
                if (i < 0 || i >= n) { std::cerr << "idx fuera de rango: " << i << '\n'; return 1; }
                Parent p;
                p.ind          = snap.pop[i];
                p.idx          = i;
                p.label        = base + "#" + std::to_string(i);
                p.evalSeed     = (opt.haveSeed ? opt.seed : snap.evalSeed) * 10000 + i;
                p.storedReward = (opt.haveSeed ? std::vector<double>{} : snap.reward[i]);
                parents.push_back(std::move(p));
            }
            std::cout << "Snapshot gen " << snap.gen << ": " << n << " individuos, analizando "
                      << parents.size() << " (--who " << opt.who << ")\n";
        } else {
            Parent p;
            p.ind      = genomeFromNetFile(opt.input, hyp.ann_nInput, hyp.ann_nOutput);
            p.idx      = 0;
            p.label    = base;
            p.evalSeed = opt.seed * 10000;
            parents.push_back(std::move(p));
            std::cout << "Red " << opt.input << ": genoma reconstruido de la matriz guardada "
                         "(aproximado; ver GenomeIO.h). Semillas de evaluación propias, "
                         "no las del entrenamiento.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error cargando " << opt.input << ": " << e.what() << '\n';
        return 1;
    }

    if (opt.outPrefix.empty()) opt.outPrefix = "log/neighbors_" + fs::path(opt.input).stem().string();
    if (fs::path(opt.outPrefix).has_parent_path())
        fs::create_directories(fs::path(opt.outPrefix).parent_path());

    SnnCarTask task(hyp);
    const int nW = task.numWeightVals();

    std::ofstream nbCsv(opt.outPrefix + "_neighbors.csv");
    std::ofstream parCsv(opt.outPrefix + "_parents.csv");
    std::ofstream opsCsv(opt.outPrefix + "_ops.csv");
    if (!nbCsv || !parCsv || !opsCsv) {
        std::cerr << "No se pudo escribir " << opt.outPrefix << "_*.csv\n";
        return 1;
    }
    nbCsv << std::setprecision(10);
    parCsv << std::setprecision(10);
    opsCsv << std::setprecision(10);
    nbCsv << "label,idx,step,order,mid,mid_op,mid_d_mean,op,a,b,c,prob,weight,expr_ok,"
             "n_conn,n_hidden,fit_mean,fit_max,d_mean,d_max";
    for (int w = 0; w < nW; ++w) nbCsv << ",r" << w;
    nbCsv << '\n';
    parCsv << "label,idx,step,seed,fit_mean,fit_max,n_conn,n_hidden,noise_sd,eps,stored_maxdiff,"
              "n_neighbors,n_better,n_neutral,n_identical,n_worse,n_invalid,stuck,p_better,exp_mutations_to_improve\n";
    opsCsv << "label,idx,step,op,n,n_better,n_neutral,n_identical,n_worse,n_invalid,best_d_mean,p_op,p_better\n";

    Analyzer analyzer(opt, hyp, task, nbCsv, parCsv, opsCsv);
    for (const auto& p : parents) analyzer.run(p);

    std::cout << "\nEscrito: " << opt.outPrefix << "_{neighbors,parents,ops}.csv\n";
    return 0;
}

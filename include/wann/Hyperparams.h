#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace wann {

struct Hyperparams {
    // --- algorithm ---
    std::string task          = "swingup";
    std::string alg_wDist     = "standard";
    int    alg_nVals          = 6;
    int    alg_nReps          = 4;
    double alg_probMoo        = 0.80;
    int    maxGen             = 2048;
    int    popSize            = 128;

    // --- mutation probabilities ---
    double prob_crossover     = 0.0;
    double prob_mutAct        = 0.50;
    double prob_addNode       = 0.25;
    double prob_addConn       = 0.20;
    double prob_enable        = 0.05;
    double prob_initEnable    = 0.5;

    // --- selection ---
    double select_cullRatio   = 0.2;
    double select_eliteRatio  = 0.2;
    int    select_tournSize   = 8;

    // --- I/O ---
    int    save_mod           = 8;
    int    bestReps           = 20;

    // --- task-specific (set in JSON or programmatically) ---
    int    ann_nInput         = 5;
    int    ann_nOutput        = 1;
    int    ann_initAct        = 1;
    std::vector<int> ann_actRange = {1,2,3,4,5,6,7,8,9,10};
    double ann_absWCap        = 2.0;
};

// Load base hyperparameters from a JSON file.
inline Hyperparams loadHyp(const std::string& fname) {
    std::ifstream f(fname);
    if (!f) throw std::runtime_error("Cannot open hyperparameter file: " + fname);
    nlohmann::json j;
    f >> j;

    Hyperparams p;
    auto get = [&](auto& field, const char* key) {
        if (j.contains(key))
            field = j.at(key).get<std::remove_reference_t<decltype(field)>>();
    };
    get(p.task,              "task");
    get(p.alg_wDist,         "alg_wDist");
    get(p.alg_nVals,         "alg_nVals");
    get(p.alg_nReps,         "alg_nReps");
    get(p.alg_probMoo,       "alg_probMoo");
    get(p.maxGen,            "maxGen");
    get(p.popSize,           "popSize");
    get(p.prob_crossover,    "prob_crossover");
    get(p.prob_mutAct,       "prob_mutAct");
    get(p.prob_addNode,      "prob_addNode");
    get(p.prob_addConn,      "prob_addConn");
    get(p.prob_enable,       "prob_enable");
    get(p.prob_initEnable,   "prob_initEnable");
    get(p.select_cullRatio,  "select_cullRatio");
    get(p.select_eliteRatio, "select_eliteRatio");
    get(p.select_tournSize,  "select_tournSize");
    get(p.save_mod,          "save_mod");
    get(p.bestReps,          "bestReps");
    get(p.ann_nInput,        "ann_nInput");
    get(p.ann_nOutput,       "ann_nOutput");
    get(p.ann_initAct,       "ann_initAct");
    get(p.ann_actRange,      "ann_actRange");
    get(p.ann_absWCap,       "ann_absWCap");
    return p;
}

// Merge overrides from a second JSON file into an existing Hyperparams.
inline void updateHyp(Hyperparams& p, const std::string& fname) {
    std::ifstream f(fname);
    if (!f) throw std::runtime_error("Cannot open hyperparameter file: " + fname);
    nlohmann::json j;
    f >> j;

    auto get = [&](auto& field, const char* key) {
        if (j.contains(key))
            field = j.at(key).get<std::remove_reference_t<decltype(field)>>();
    };
    get(p.task,              "task");
    get(p.alg_wDist,         "alg_wDist");
    get(p.alg_nVals,         "alg_nVals");
    get(p.alg_nReps,         "alg_nReps");
    get(p.alg_probMoo,       "alg_probMoo");
    get(p.maxGen,            "maxGen");
    get(p.popSize,           "popSize");
    get(p.prob_crossover,    "prob_crossover");
    get(p.prob_mutAct,       "prob_mutAct");
    get(p.prob_addNode,      "prob_addNode");
    get(p.prob_addConn,      "prob_addConn");
    get(p.prob_enable,       "prob_enable");
    get(p.prob_initEnable,   "prob_initEnable");
    get(p.select_cullRatio,  "select_cullRatio");
    get(p.select_eliteRatio, "select_eliteRatio");
    get(p.select_tournSize,  "select_tournSize");
    get(p.save_mod,          "save_mod");
    get(p.bestReps,          "bestReps");
    get(p.ann_nInput,        "ann_nInput");
    get(p.ann_nOutput,       "ann_nOutput");
    get(p.ann_initAct,       "ann_initAct");
    get(p.ann_actRange,      "ann_actRange");
    get(p.ann_absWCap,       "ann_absWCap");
}

} // namespace wann

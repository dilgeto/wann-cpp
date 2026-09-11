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
    double prob_crossover          = 0.0;
    double prob_mutAct             = 0.50;
    double prob_addNode            = 0.25;
    double prob_addConn            = 0.20;
    double prob_enable             = 0.05;
    double prob_initEnable         = 0.5;
    double prob_toggleExcitatory   = 0.10;

    // --- selection ---
    double select_cullRatio   = 0.2;
    double select_eliteRatio  = 0.2;
    int    select_tournSize   = 8;

    // --- I/O ---
    int    save_mod           = 8;
    int    bestReps           = 20;

    // --- early stopping ---
    // Generations without a new fitTop record (running-best elite fitness)
    // before stopping early. 0 = disabled (run the full maxGen).
    int    early_stop_patience = 0;

    // --- task-specific (set in JSON or programmatically) ---
    int    ann_nInput         = 5;
    int    ann_nOutput        = 1;
    int    ann_initAct        = 1;
    std::vector<int> ann_actRange = {1,2,3,4,5,6,7,8,9,10};
    double ann_absWCap        = 2.0;

    // --- SNN interface ---
    // Encoder: "current" | "poisson" | "rate" | "ttfs" | "ttfs_log" | "small" | "large"
    // Decoder: "spike_count" | "rate" | "first_spike" | "voting" | "rate_argmax"
    std::string snn_encoder   = "poisson";
    std::string snn_decoder   = "rate";
    int         snn_neurons_per_var = 5;

    // --- reward shaping ---
    // Potential-based shaping: F(s,s') = scale * (phi(s') - phi(s))
    // where phi(s) = sin(3 * position).  Set to 0 to disable.
    double reward_shaping_scale = 0.0;

    // When true (default): reset SNN membrane state before each env step.
    // When false: state persists across steps (implicit recurrence).
    bool snn_reset_between_steps = true;

    // --- SNN simulator microparameters (currently wired for SnnCarTask only) ---
    // ms per env step (decision window given to the SNN before decoding an
    // action). Was a compile-time constant (WANN_CAR_SIM_WINDOW_MS); exposing
    // it here lets it be searched at runtime. Do NOT also search a separate
    // "dt" alongside this — the decoder's quantization step is 2*dt/window,
    // so dt and window are collinear for that effect; keep dt fixed and vary
    // only the window.
    double snn_window_ms       = 40.0;
    // Izhikevich AMPA/GABA conductance decay time constants (ms). Govern how
    // long an input spike's effect survives before decaying — directly
    // interacts with snn_window_ms for late-arriving TTFS spikes (see
    // ttfsEncoder: low observation values spike near the end of the window).
    double snn_tau_exc         = 5.0;
    double snn_tau_inh         = 10.0;
    // TTFS encoder: observation values below this produce no input spike at
    // all. Independent of snn_window_ms (not part of the t_max formula), but
    // interacts with it in effect (both can push toward more "no spike"
    // fallback actions). Do NOT also search snn_ttfs_tmax_ratio alongside
    // snn_window_ms — t_max = (window-dt)*tmax_ratio is a product, so the two
    // are collinear for that formula; tmax_ratio stays fixed at 1.0.
    double snn_ttfs_threshold  = 1e-9;
};

namespace detail {

inline void applyJson(Hyperparams& p, const nlohmann::json& j) {
    auto get = [&](auto& field, const char* key) {
        if (j.contains(key))
            field = j.at(key).get<std::remove_reference_t<decltype(field)>>();
    };
    get(p.task,                   "task");
    get(p.alg_wDist,              "alg_wDist");
    get(p.alg_nVals,              "alg_nVals");
    get(p.alg_nReps,              "alg_nReps");
    get(p.alg_probMoo,            "alg_probMoo");
    get(p.maxGen,                 "maxGen");
    get(p.popSize,                "popSize");
    get(p.prob_crossover,         "prob_crossover");
    get(p.prob_mutAct,            "prob_mutAct");
    get(p.prob_addNode,           "prob_addNode");
    get(p.prob_addConn,           "prob_addConn");
    get(p.prob_enable,            "prob_enable");
    get(p.prob_initEnable,        "prob_initEnable");
    get(p.prob_toggleExcitatory,  "prob_toggleExcitatory");
    get(p.select_cullRatio,       "select_cullRatio");
    get(p.select_eliteRatio,      "select_eliteRatio");
    get(p.select_tournSize,       "select_tournSize");
    get(p.save_mod,               "save_mod");
    get(p.bestReps,               "bestReps");
    get(p.early_stop_patience,    "early_stop_patience");
    get(p.ann_nInput,             "ann_nInput");
    get(p.ann_nOutput,            "ann_nOutput");
    get(p.ann_initAct,            "ann_initAct");
    get(p.ann_actRange,           "ann_actRange");
    get(p.ann_absWCap,            "ann_absWCap");
    get(p.snn_encoder,            "snn_encoder");
    get(p.snn_decoder,            "snn_decoder");
    get(p.snn_neurons_per_var,    "snn_neurons_per_var");
    get(p.reward_shaping_scale,      "reward_shaping_scale");
    get(p.snn_reset_between_steps,   "snn_reset_between_steps");
    get(p.snn_window_ms,             "snn_window_ms");
    get(p.snn_tau_exc,               "snn_tau_exc");
    get(p.snn_tau_inh,               "snn_tau_inh");
    get(p.snn_ttfs_threshold,        "snn_ttfs_threshold");
}

// Parse a string that is either a file path or an inline JSON object.
inline nlohmann::json parseFileOrInline(const std::string& s) {
    if (!s.empty() && s.front() == '{')
        return nlohmann::json::parse(s);
    std::ifstream f(s);
    if (!f) throw std::runtime_error("Cannot open hyperparameter file: " + s);
    nlohmann::json j;
    f >> j;
    return j;
}

} // namespace detail

// Load base hyperparameters. Accepts a file path or an inline JSON string.
inline Hyperparams loadHyp(const std::string& src) {
    Hyperparams p;
    detail::applyJson(p, detail::parseFileOrInline(src));
    return p;
}

// Merge overrides into an existing Hyperparams.
// Accepts a file path or an inline JSON string (e.g. '{"maxGen":1}').
inline void updateHyp(Hyperparams& p, const std::string& src) {
    detail::applyJson(p, detail::parseFileOrInline(src));
}

} // namespace wann

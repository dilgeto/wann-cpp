#include "../include/wann/OdinExport.h"

#include <core/neuron.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace wann {

namespace {

// Mirrors SnnCarTask::wannActToNeuronType (src/SnnCarTask.cpp) — kept as a
// separate copy here so this export tool doesn't have to link the rl-tools
// Car environment / snn-simulator engine just to reuse a 25-entry switch.
// Keep in sync with SnnCarTask.cpp if that mapping changes.
NeuronType actIdToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        case 7:  return NeuronType::TONIC_SPIKING;
        case 8:  return NeuronType::PHASIC_SPIKING;
        case 9:  return NeuronType::TONIC_BURSTING;
        case 10: return NeuronType::PHASIC_BURSTING;
        case 11: return NeuronType::MIXED_MODE;
        case 12: return NeuronType::SPIKE_FREQUENCY_ADAPTATION;
        case 13: return NeuronType::CLASS1_EXCITABLE;
        case 14: return NeuronType::CLASS2_EXCITABLE;
        case 15: return NeuronType::SPIKE_LATENCY;
        case 16: return NeuronType::SUBTHRESHOLD_OSCILLATIONS;
        case 17: return NeuronType::INTEGRATOR;
        case 18: return NeuronType::REBOUND_SPIKE;
        case 19: return NeuronType::REBOUND_BURST;
        case 20: return NeuronType::THRESHOLD_VARIABILITY;
        case 21: return NeuronType::BISTABILITY;
        case 22: return NeuronType::DEPOLARIZING_AFTERPOTENTIAL;
        case 23: return NeuronType::ACCOMMODATION;
        case 24: return NeuronType::INHIBITION_INDUCED_SPIKING;
        case 25: return NeuronType::INHIBITION_INDUCED_BURSTING;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

const char* neuronTypeName(NeuronType nt) {
    switch (nt) {
        case NeuronType::REGULAR_SPIKING:        return "REGULAR_SPIKING";
        case NeuronType::FAST_SPIKING:           return "FAST_SPIKING";
        case NeuronType::CHATTERING:              return "CHATTERING";
        case NeuronType::LOW_THRESHOLD_SPIKING:  return "LOW_THRESHOLD_SPIKING";
        case NeuronType::INTRINSICALLY_BURSTING: return "INTRINSICALLY_BURSTING";
        case NeuronType::RESONATOR:              return "RESONATOR";
        default:                                 return "UNKNOWN";
    }
}

// ODIN IZH hardware parameters for the 6 classic cortical/thalamic
// behaviours (Izhikevich 2004, Fig. 2) — the only ones car_snn.json's base
// ann_actRange=[1..6] evolves with. Deliberately doesn't cover the other 19
// Izhikevich behaviours: see IzhParams's doc comment in OdinExport.h for why
// (no published ODIN behaviour table, and this deployment's genomes only
// ever use these 6). Throws for anything else rather than guessing.
IzhParams izhParamsFor(NeuronType nt) {
    IzhParams p;
    switch (nt) {
        case NeuronType::REGULAR_SPIKING:
            // Plain leaky integrator, tonic regular firing, no special enables.
            p.thr = 4; p.rfr = 2; p.leakStr = 10; p.leakEn = 1;
            return p;
        case NeuronType::FAST_SPIKING:
            // Low threshold, no refractory period -> fast sustained firing.
            p.thr = 2; p.rfr = 0; p.leakStr = 6; p.leakEn = 1;
            return p;
        case NeuronType::CHATTERING:
            // Short, high-frequency bursts: high spk_ref, minimal isi_ref.
            p.thr = 4; p.rfr = 2; p.leakStr = 10; p.leakEn = 1;
            p.spkRef = 5; p.isiRef = 1;
            return p;
        case NeuronType::LOW_THRESHOLD_SPIKING:
            // Low threshold + rebound (LTS neurons classically rebound after
            // inhibitory release) -> needs neg_en for the hyperpolarized state.
            p.thr = 2; p.rfr = 1; p.leakStr = 8; p.leakEn = 1;
            p.reboundEn = true; p.negEn = true;
            return p;
        case NeuronType::INTRINSICALLY_BURSTING:
            // Slower/longer bursts than chattering.
            p.thr = 4; p.rfr = 2; p.leakStr = 10; p.leakEn = 1;
            p.spkRef = 3; p.isiRef = 3;
            return p;
        case NeuronType::RESONATOR:
            // Direct match: ODIN's dedicated resonant-behaviour circuit.
            p.thr = 4; p.rfr = 2; p.leakStr = 10; p.leakEn = 1;
            p.resonEn = true;
            return p;
        default:
            throw std::runtime_error(
                std::string("izhParamsFor: NeuronType ") + neuronTypeName(nt) +
                " is outside the 6 behaviours this deployment supports "
                "(REGULAR_SPIKING..RESONATOR) — either car_snn.json's "
                "ann_actRange grew beyond [1..6], or this genome wasn't "
                "evolved with the base config. Add a table entry (see "
                "IzhParams's doc comment) before deploying it.");
    }
}

} // namespace

OdinNetworkConfig buildOdinConfig(const std::vector<double>& wVec,
                                   const std::vector<int>&    aVec,
                                   int nInput, int nOutput,
                                   double sharedWeight, int weightIndex,
                                   int magnitude3bit,
                                   const std::string& task,
                                   const std::string& runKey)
{
    const int N0 = static_cast<int>(aVec.size());
    if (magnitude3bit < 0 || magnitude3bit > 7) {
        throw std::runtime_error("buildOdinConfig: magnitude3bit must be in [0,7]");
    }

    // Collect the raw edge list first (src, dst, excitatory) before deciding
    // on physical addresses — needed below to detect and fix Dale's-law
    // violations.
    struct Edge { int src, dst; bool excitatory; };
    std::vector<Edge> edges;
    for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N0; ++j) {
            if (i == j) continue;
            double w = wVec[i * N0 + j];
            if (w == 0.0) continue;
            edges.push_back({i, j, w > 0.0});
        }

    // Real ODIN hardware stores synapse sign per PRE-SYNAPTIC NEURON, not
    // per synapse (Dale's law) — the synapse SRAM only has weight + a mapped
    // bit, confirmed against ODIN's official doc (section 3.2) and its
    // set_syn_sign() API. WANN has no such constraint (mutToggleExcitatory
    // flips sign per connection), so a node with both excitatory and
    // inhibitory outgoing edges — which is the common case, not an edge
    // case — can't be given a single physical address as-is.
    //
    // Fix: give such a node two physical addresses ("twins") with identical
    // neuron parameters and identical incoming edges, so they fire in
    // lockstep; each twin only keeps the outgoing edges of one sign. This
    // exactly reproduces the original node's function — downstream neurons
    // see the same excitatory/inhibitory drive they would have from a
    // single mixed-sign node.
    std::vector<std::array<int, 2>> outSignSeen(N0, std::array<int, 2>{0, 0});
    for (const auto& e : edges) outSignSeen[e.src][e.excitatory ? 1 : 0]++;

    // addr[origId][0]=primary physical address, addr[origId][1]=twin address
    // (-1 if this node didn't need splitting) mapped by "excitatory" (index
    // 1) vs "inhibitory" (index 0) — primary always keeps whichever sign has
    // more outgoing edges (ties favour excitatory).
    std::vector<std::array<int, 2>> addrForSign(N0, std::array<int, 2>{-1, -1});
    std::vector<int> primarySign(N0, 1);
    int nextAddr = N0;
    std::vector<int> twinOf(N0, -1);  // orig id -> twin's orig-id-space slot (for role/param copy)

    for (int i = 0; i < N0; ++i) {
        bool hasExc = outSignSeen[i][1] > 0;
        bool hasInh = outSignSeen[i][0] > 0;
        if (hasExc && hasInh) {
            primarySign[i] = (outSignSeen[i][0] > outSignSeen[i][1]) ? 0 : 1;
            addrForSign[i][primarySign[i]] = i;
            addrForSign[i][1 - primarySign[i]] = nextAddr;
            twinOf[i] = nextAddr;
            ++nextAddr;
        } else {
            // Single sign (or no outgoing edges at all, sign irrelevant).
            addrForSign[i][0] = i;
            addrForSign[i][1] = i;
        }
    }

    const int N = nextAddr;
    if (N > ODIN_MAX_NEURONS) {
        throw std::runtime_error(
            "buildOdinConfig: network needs " + std::to_string(N) +
            " physical ODIN addresses (" + std::to_string(N0) + " original nodes + " +
            std::to_string(N - N0) + " excitatory/inhibitory twins for Dale's-law "
            "splitting), but ODIN's core only has " + std::to_string(ODIN_MAX_NEURONS) +
            " — this genome cannot be deployed as-is.");
    }

    OdinNetworkConfig cfg;
    cfg.task         = task;
    cfg.runKey       = runKey;
    cfg.weightIndex  = weightIndex;
    cfg.sharedWeight = sharedWeight;
    cfg.nNeurons     = N;

    // Same node ordering as SnnCarTask::buildNetwork(wVec,aVec) for the
    // original N0 addresses: 0 = bias/first input, 1..nInput = inputs, then
    // hidden, then nOutput outputs. Twins appended after N0 inherit their
    // origin's role/behaviour (they're the same logical neuron, split only
    // for hardware sign routing).
    cfg.neurons.resize(N);
    for (int i = 0; i < N0; ++i) {
        std::string role = (i <= nInput) ? "input"
                          : (i >= N0 - nOutput) ? "output"
                          : "hidden";
        NeuronType nt = actIdToNeuronType(aVec[i]);
        IzhParams  izh = izhParamsFor(nt);
        const char* name = neuronTypeName(nt);
        cfg.neurons[i] = {i, role, name, izh, twinOf[i]};
        if (twinOf[i] >= 0) {
            std::string twinRole = role + "_twin";
            cfg.neurons[twinOf[i]] = {twinOf[i], twinRole, name, izh, i};
        }
    }

    for (const auto& e : edges) {
        int resolvedSrc = addrForSign[e.src][e.excitatory ? 1 : 0];
        // Every physical address representing `dst` (1, or 2 if dst was
        // split) must receive this edge so both twins stay in lockstep.
        cfg.synapses.push_back({resolvedSrc, e.dst, e.excitatory, magnitude3bit});
        if (twinOf[e.dst] >= 0)
            cfg.synapses.push_back({resolvedSrc, twinOf[e.dst], e.excitatory, magnitude3bit});
    }

    return cfg;
}

void writeOdinConfig(const std::string& path, const OdinNetworkConfig& cfg) {
    nlohmann::json j;
    j["task"]          = cfg.task;
    j["run_key"]        = cfg.runKey;
    j["weight_index"]   = cfg.weightIndex;
    j["shared_weight"]  = cfg.sharedWeight;
    j["n_neurons"]      = cfg.nNeurons;
    j["odin_max_neurons"] = ODIN_MAX_NEURONS;

    auto& neurons = j["neurons"];
    neurons = nlohmann::json::array();
    for (const auto& n : cfg.neurons) {
        const auto& p = n.izh;
        neurons.push_back({
            {"addr", n.addr}, {"role", n.role}, {"neuron_type", n.neuronType},
            {"twin_addr", n.twinAddr},
            {"izh", {
                {"leak_str", p.leakStr}, {"leak_en", p.leakEn}, {"fi_sel", p.fiSel},
                {"thr", p.thr}, {"rfr", p.rfr},
                {"spk_ref", p.spkRef}, {"isi_ref", p.isiRef},
                {"dapdel", p.dapdel}, {"stim_thr", p.stimThr}, {"thrleak", p.thrleak},
                {"reson_sharp_amt", p.resonSharpAmt},
                {"spklat_en", p.spklatEn}, {"dap_en", p.dapEn},
                {"phasic_en", p.phasicEn}, {"mixed_en", p.mixedEn},
                {"class2_en", p.class2En}, {"neg_en", p.negEn},
                {"rebound_en", p.reboundEn}, {"inhin_en", p.inhinEn},
                {"bist_en", p.bistEn}, {"reson_en", p.resonEn},
                {"thrvar_en", p.thrvarEn}, {"thr_sel_of", p.thrSelOf},
                {"acc_en", p.accEn}, {"reson_sharp_en", p.resonSharpEn},
            }},
        });
    }

    auto& synapses = j["synapses"];
    synapses = nlohmann::json::array();
    for (const auto& s : cfg.synapses) {
        synapses.push_back({
            {"src", s.src}, {"dst", s.dst},
            {"excitatory", s.excitatory}, {"weight_3bit", s.weight3bit},
        });
    }

    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f << j.dump(2) << '\n';
}

OdinNetworkConfig readOdinConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot read: " + path);
    nlohmann::json j;
    f >> j;

    OdinNetworkConfig cfg;
    cfg.task         = j.at("task").get<std::string>();
    cfg.runKey       = j.at("run_key").get<std::string>();
    cfg.weightIndex  = j.at("weight_index").get<int>();
    cfg.sharedWeight = j.at("shared_weight").get<double>();
    cfg.nNeurons     = j.at("n_neurons").get<int>();

    for (const auto& n : j.at("neurons")) {
        IzhParams p;
        const auto& ij = n.at("izh");
        p.leakStr = ij.at("leak_str"); p.leakEn = ij.at("leak_en"); p.fiSel = ij.at("fi_sel");
        p.thr = ij.at("thr"); p.rfr = ij.at("rfr");
        p.spkRef = ij.at("spk_ref"); p.isiRef = ij.at("isi_ref");
        p.dapdel = ij.at("dapdel"); p.stimThr = ij.at("stim_thr"); p.thrleak = ij.at("thrleak");
        p.resonSharpAmt = ij.at("reson_sharp_amt");
        p.spklatEn = ij.at("spklat_en"); p.dapEn = ij.at("dap_en");
        p.phasicEn = ij.at("phasic_en"); p.mixedEn = ij.at("mixed_en");
        p.class2En = ij.at("class2_en"); p.negEn = ij.at("neg_en");
        p.reboundEn = ij.at("rebound_en"); p.inhinEn = ij.at("inhin_en");
        p.bistEn = ij.at("bist_en"); p.resonEn = ij.at("reson_en");
        p.thrvarEn = ij.at("thrvar_en"); p.thrSelOf = ij.at("thr_sel_of");
        p.accEn = ij.at("acc_en"); p.resonSharpEn = ij.at("reson_sharp_en");

        cfg.neurons.push_back({
            n.at("addr").get<int>(), n.at("role").get<std::string>(),
            n.at("neuron_type").get<std::string>(),
            p,
            n.value("twin_addr", -1),
        });
    }
    for (const auto& s : j.at("synapses")) {
        cfg.synapses.push_back({
            s.at("src").get<int>(), s.at("dst").get<int>(),
            s.at("excitatory").get<bool>(), s.at("weight_3bit").get<int>(),
        });
    }
    return cfg;
}

} // namespace wann

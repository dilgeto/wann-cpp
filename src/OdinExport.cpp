#include "../include/wann/OdinExport.h"

#include <core/neuron.hpp>

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

// Izhikevich (2004) parameters — mirrors the switch in
// snn-simulator/src/core/neuron.cpp (IzhikevichNeuron's NeuronType
// constructor). Duplicated here because those fields are private on
// IzhikevichNeuron with no accessor; keep in sync if that table changes.
struct AbcdName { double a, b, c, d; const char* name; };

AbcdName neuronParams(NeuronType nt) {
    switch (nt) {
        case NeuronType::REGULAR_SPIKING:             return {0.02, 0.2,  -65.0, 8.0,   "REGULAR_SPIKING"};
        case NeuronType::FAST_SPIKING:                return {0.1,  0.2,  -65.0, 2.0,   "FAST_SPIKING"};
        case NeuronType::INTRINSICALLY_BURSTING:      return {0.02, 0.2,  -55.0, 4.0,   "INTRINSICALLY_BURSTING"};
        case NeuronType::CHATTERING:                  return {0.02, 0.2,  -50.0, 2.0,   "CHATTERING"};
        case NeuronType::LOW_THRESHOLD_SPIKING:       return {0.02, 0.25, -65.0, 2.0,   "LOW_THRESHOLD_SPIKING"};
        case NeuronType::RESONATOR:                   return {0.1,  0.26, -65.0, 2.0,   "RESONATOR"};
        case NeuronType::TONIC_SPIKING:               return {0.02, 0.2,  -65.0, 6.0,   "TONIC_SPIKING"};
        case NeuronType::PHASIC_SPIKING:              return {0.02, 0.25, -65.0, 6.0,   "PHASIC_SPIKING"};
        case NeuronType::TONIC_BURSTING:              return {0.02, 0.2,  -50.0, 2.0,   "TONIC_BURSTING"};
        case NeuronType::PHASIC_BURSTING:             return {0.02, 0.25, -55.0, 0.05,  "PHASIC_BURSTING"};
        case NeuronType::MIXED_MODE:                  return {0.02, 0.2,  -55.0, 4.0,   "MIXED_MODE"};
        case NeuronType::SPIKE_FREQUENCY_ADAPTATION:  return {0.01, 0.2,  -65.0, 8.0,   "SPIKE_FREQUENCY_ADAPTATION"};
        case NeuronType::CLASS1_EXCITABLE:            return {0.02, -0.1, -55.0, 6.0,   "CLASS1_EXCITABLE"};
        case NeuronType::CLASS2_EXCITABLE:            return {0.2,  0.26, -65.0, 0.0,   "CLASS2_EXCITABLE"};
        case NeuronType::SPIKE_LATENCY:               return {0.02, 0.2,  -65.0, 6.0,   "SPIKE_LATENCY"};
        case NeuronType::SUBTHRESHOLD_OSCILLATIONS:   return {0.05, 0.26, -60.0, 0.0,   "SUBTHRESHOLD_OSCILLATIONS"};
        case NeuronType::INTEGRATOR:                  return {0.02, -0.1, -55.0, 6.0,   "INTEGRATOR"};
        case NeuronType::REBOUND_SPIKE:                return {0.03, 0.25, -60.0, 4.0,   "REBOUND_SPIKE"};
        case NeuronType::REBOUND_BURST:                return {0.03, 0.25, -52.0, 0.0,   "REBOUND_BURST"};
        case NeuronType::THRESHOLD_VARIABILITY:        return {0.03, 0.25, -60.0, 4.0,   "THRESHOLD_VARIABILITY"};
        case NeuronType::BISTABILITY:                  return {1.0,  1.5,  -60.0, 0.0,   "BISTABILITY"};
        case NeuronType::DEPOLARIZING_AFTERPOTENTIAL:  return {1.0,  0.2,  -60.0, -21.0, "DEPOLARIZING_AFTERPOTENTIAL"};
        case NeuronType::ACCOMMODATION:                return {0.02, 1.0,  -55.0, 4.0,   "ACCOMMODATION"};
        case NeuronType::INHIBITION_INDUCED_SPIKING:   return {-0.02, -1.0, -60.0, 8.0,  "INHIBITION_INDUCED_SPIKING"};
        case NeuronType::INHIBITION_INDUCED_BURSTING:  return {-0.026, -1.0, -45.0, -2.0,"INHIBITION_INDUCED_BURSTING"};
        default:                                       return {0.02, 0.2,  -65.0, 8.0,   "REGULAR_SPIKING"};
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
    const int N = static_cast<int>(aVec.size());
    if (N > ODIN_MAX_NEURONS) {
        throw std::runtime_error(
            "buildOdinConfig: network has " + std::to_string(N) +
            " nodes, ODIN's core only has " + std::to_string(ODIN_MAX_NEURONS) +
            " physical neurons — this genome cannot be deployed as-is.");
    }
    if (magnitude3bit < 0 || magnitude3bit > 7) {
        throw std::runtime_error("buildOdinConfig: magnitude3bit must be in [0,7]");
    }

    OdinNetworkConfig cfg;
    cfg.task         = task;
    cfg.runKey       = runKey;
    cfg.weightIndex  = weightIndex;
    cfg.sharedWeight = sharedWeight;
    cfg.nNeurons     = N;

    // Same node ordering as SnnCarTask::buildNetwork(wVec,aVec):
    // 0 = bias/first input, 1..nInput = inputs, then hidden, then nOutput outputs.
    cfg.neurons.reserve(N);
    for (int i = 0; i < N; ++i) {
        std::string role = (i <= nInput) ? "input"
                          : (i >= N - nOutput) ? "output"
                          : "hidden";
        auto p = neuronParams(actIdToNeuronType(aVec[i]));
        cfg.neurons.push_back({i, role, p.name, p.a, p.b, p.c, p.d});
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            double w = wVec[i * N + j];
            if (w == 0.0) continue;
            cfg.synapses.push_back({i, j, w > 0.0, magnitude3bit});
        }
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
        neurons.push_back({
            {"addr", n.addr}, {"role", n.role}, {"neuron_type", n.neuronType},
            {"a", n.a}, {"b", n.b}, {"c", n.c}, {"d", n.d},
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
        cfg.neurons.push_back({
            n.at("addr").get<int>(), n.at("role").get<std::string>(),
            n.at("neuron_type").get<std::string>(),
            n.at("a").get<double>(), n.at("b").get<double>(),
            n.at("c").get<double>(), n.at("d").get<double>(),
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

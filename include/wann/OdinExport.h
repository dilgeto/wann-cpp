#pragma once
#include <string>
#include <vector>

namespace wann {

// One ODIN neuron address (0..255). "role" mirrors the WANN node type this
// address was expressed from (input/hidden/output) purely for readability
// in the exported file — ODIN itself makes no such distinction.
struct OdinNeuronConfig {
    int         addr;
    std::string role;         // "input" | "hidden" | "output"
    std::string neuronType;   // NeuronType enum name (snn-simulator/include/core/neuron.hpp)
    double      a, b, c, d;   // Izhikevich parameters for that behaviour
};

// One ODIN synapse (crossbar entry). weight3bit is a 0..7 magnitude bucket;
// sign is carried separately via `excitatory` because ODIN's actual sign
// wiring (per-synapse vs. per-source-neuron) isn't pinned down yet — see
// OdinDriver.
struct OdinSynapseConfig {
    int  src;
    int  dst;
    bool excitatory;
    int  weight3bit;
};

struct OdinNetworkConfig {
    std::string task;
    std::string runKey;
    int         weightIndex = -1;
    double      sharedWeight = 0.0;
    int         nNeurons = 0;
    std::vector<OdinNeuronConfig>  neurons;
    std::vector<OdinSynapseConfig> synapses;
};

// ODIN's neurosynaptic core has a single 256-neuron / 256x256-synapse crossbar
// (Frenkel et al.) — every WANN node (input, hidden, output alike) must map
// to one physical address in that same space.
constexpr int ODIN_MAX_NEURONS = 256;

// Builds an ODIN-ready config from an expressed WANN+SNN network (the same
// wVec/aVec produced by wann::importNet / Ind::express, and the shared
// scalar weight the evolution picked as the winner for this genome).
// `nInput` follows the same node ordering SnnCarTask::buildNetwork(wVec,aVec)
// uses: node 0 is bias, 1..nInput are inputs, then hidden, then nOutput
// outputs at the end. Throws std::runtime_error if N > ODIN_MAX_NEURONS.
// `magnitude3bit` (0..7) is the caller's quantization of `sharedWeight` into
// ODIN's 3-bit synapse weight, since only the caller (the task-specific eval
// tool) knows the shared-weight domain (e.g. SnnCarTask::WEIGHT_VALS) needed
// to normalize it sensibly. `sharedWeight`/`weightIndex` are kept only for
// bookkeeping in the exported file.
OdinNetworkConfig buildOdinConfig(const std::vector<double>& wVec,
                                   const std::vector<int>&    aVec,
                                   int nInput, int nOutput,
                                   double sharedWeight, int weightIndex,
                                   int magnitude3bit,
                                   const std::string& task,
                                   const std::string& runKey);

// Serializes to / reads back from JSON (see src/main_odin_export.cpp for
// the on-disk shape).
void writeOdinConfig(const std::string& path, const OdinNetworkConfig& cfg);
OdinNetworkConfig readOdinConfig(const std::string& path);

} // namespace wann

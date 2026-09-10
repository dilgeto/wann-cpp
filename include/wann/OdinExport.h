#pragma once
#include <string>
#include <vector>

namespace wann {

// ODIN's phenomenological Izhikevich (IZH) neuron parameters, bit-for-bit
// matching the fields neuron_core.v feeds into izh_neuron.v (see
// OdinDriver::neuronIzh). Scoped ONLY to the 6 classic cortical/thalamic
// behaviours snn-simulator's NeuronType enum lists first (REGULAR_SPIKING..
// RESONATOR) — the ones car_snn.json's ann_actRange actually evolves with —
// not all 20: see buildOdinConfig's izhParamsFor(). SDSP/learning fields
// (ca_en, thetamem, ca_theta1-3, caleak, burst_incr) aren't here because
// they're always 0 (no on-chip learning — weights are evolved offline).
//
// CAVEAT: this table is this project's best-effort reading of the enable
// bits' names/control logic in ODIN's public RTL (src/izh_neuron*.v,
// src/neuron_core.v) — ODIN's own documentation never published a
// behaviour-to-parameter table, so treat the numeric fields (thr, rfr,
// spkRef, isiRef, ...) as untuned starting points to validate/adjust
// empirically on hardware, not as verified-correct values.
struct IzhParams {
    int leakStr = 10, leakEn = 1;      // 7-bit, 1-bit
    int fiSel = 3;                     // 3-bit accumulator depth
    int thr = 4, rfr = 2;              // 3-bit, 3-bit
    int spkRef = 0, isiRef = 0;        // 3-bit, 3-bit — burst spike count / inter-spike interval
    int dapdel = 0, stimThr = 0;       // 3-bit, 3-bit
    int thrleak = 0;                   // 4-bit
    int resonSharpAmt = 0;             // 3-bit
    bool spklatEn = false, dapEn = false, phasicEn = false, mixedEn = false;
    bool class2En = false, negEn = false, reboundEn = false, inhinEn = false;
    bool bistEn = false, resonEn = false, thrvarEn = false, thrSelOf = false;
    bool accEn = false, resonSharpEn = false;
};

// One ODIN neuron address (0..255). "role" mirrors the WANN node type this
// address was expressed from (input/hidden/output) purely for readability
// in the exported file — ODIN itself makes no such distinction.
struct OdinNeuronConfig {
    int         addr;
    std::string role;         // "input" | "hidden" | "output" (+ "_twin" — see buildOdinConfig)
    std::string neuronType;   // NeuronType enum name (snn-simulator/include/core/neuron.hpp)
    IzhParams   izh;          // hardware IZH configuration for this neuron
    // Address of this neuron's Dale's-law excitatory/inhibitory twin (see
    // buildOdinConfig), or -1 if this node didn't need splitting. Symmetric:
    // if A's twinAddr is B, B's twinAddr is A. Both addresses must receive
    // identical stimulation/incoming synapses to stay in lockstep — this is
    // how a caller (e.g. OdinNetwork) finds the other half of a split input
    // or output node.
    int         twinAddr = -1;
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

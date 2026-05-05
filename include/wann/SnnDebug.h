#pragma once

// Debug utility: compare a WANN genome with the SNN Network built from it.
//
// Include this header from any translation unit that already links against the
// SNN simulator (e.g. main_acrobot.cpp, main_snn.cpp).
//
// Usage:
//   wann::debugSnn(ind);            // prints to stdout
//   wann::debugSnn(ind, std::cerr); // prints to another stream

#include "Ind.h"

#include <core/network.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <unordered_map>

namespace wann {

// Returns a short human-readable label for a NodeGene type.
inline const char* nodeTypeLabel(int t) {
    switch (t) {
        case 1: return "input ";
        case 2: return "output";
        case 3: return "hidden";
        case 4: return "bias  ";
        default: return "???   ";
    }
}

// Maps a WANN activation ID to the NeuronType name used in the SNN simulator.
inline const char* actToNeuronTypeName(int actId) {
    switch (actId) {
        case 1:  return "REGULAR_SPIKING";
        case 2:  return "FAST_SPIKING";
        case 3:  return "CHATTERING";
        case 4:  return "LOW_THRESHOLD_SPIKING";
        case 5:  return "INTRINSICALLY_BURSTING";
        case 6:  return "RESONATOR";
        case 7:  return "FAST_SPIKING";
        case 8:  return "REGULAR_SPIKING";
        case 9:  return "REGULAR_SPIKING";
        case 10: return "CHATTERING";
        default: return "REGULAR_SPIKING";
    }
}

inline NeuronType actToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        case 7:  return NeuronType::FAST_SPIKING;
        case 8:  return NeuronType::REGULAR_SPIKING;
        case 9:  return NeuronType::REGULAR_SPIKING;
        case 10: return NeuronType::CHATTERING;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

// Print a full side-by-side comparison of a WANN genome and the SNN Network
// constructed from it using the same logic as SnnAcrobotTask::buildNetwork.
//
// The mapping WANN_node_id → SNN_neuron_id is printed for every node, and
// every connection gene is shown with its SNN outcome (added / skipped).
//
// Then the SNN Network topology is printed independently so you can verify
// node counts and synapse excitatory/inhibitory flags match the genome.
inline void debugSnn(const Ind& ind, std::ostream& out = std::cout)
{
    out << "\n╔══════════════════════════════════════════════════════╗\n"
        << "║              WANN ↔ SNN DEBUG COMPARISON             ║\n"
        << "╚══════════════════════════════════════════════════════╝\n";

    // ----------------------------------------------------------------
    // 1. Build the SNN Network, capturing the mapping at the same time
    // ----------------------------------------------------------------
    Network net(1.0, true);
    std::unordered_map<int, int> snn_id;  // WANN node id → SNN neuron id
    snn_id.reserve(ind.nodes.size());

    out << "\n── Nodes ─────────────────────────────────────────────\n";
    out << std::left
        << std::setw(10) << "WANN_id"
        << std::setw(9)  << "type"
        << std::setw(25) << "neuron_type"
        << std::setw(10) << "SNN_id"
        << "SNN_layer\n";
    out << std::string(65, '-') << '\n';

    for (const auto& ng : ind.nodes) {
        NeuronType nt = actToNeuronType(ng.activation);
        int sid = -1;
        const char* layer = "???";
        switch (ng.type) {
            case 4: sid = net.addInputNeuron(nt);  layer = "input (bias)"; break;
            case 1: sid = net.addInputNeuron(nt);  layer = "input";        break;
            case 3: sid = net.addHiddenNeuron(nt); layer = "hidden";       break;
            case 2: sid = net.addOutputNeuron(nt); layer = "output";       break;
            default:
                out << std::setw(10) << ng.id
                    << std::setw(9)  << nodeTypeLabel(ng.type)
                    << "(skipped – unknown type)\n";
                continue;
        }
        snn_id[ng.id] = sid;
        out << std::setw(10) << ng.id
            << std::setw(9)  << nodeTypeLabel(ng.type)
            << std::setw(25) << actToNeuronTypeName(ng.activation)
            << std::setw(10) << sid
            << layer << '\n';
    }

    // ----------------------------------------------------------------
    // 2. Add synapses, printing every gene with its outcome
    // ----------------------------------------------------------------
    out << "\n── Connections ───────────────────────────────────────\n";
    out << std::setw(8)  << "innov"
        << std::setw(14) << "WANN src→dst"
        << std::setw(10) << "enabled"
        << std::setw(12) << "polarity"
        << std::setw(16) << "SNN src→dst"
        << "outcome\n";
    out << std::string(65, '-') << '\n';

    int added = 0, skipped_disabled = 0, skipped_missing = 0;
    for (const auto& cg : ind.conns) {
        out << std::setw(8)  << cg.innov
            << "  #" << std::setw(4) << cg.src << " → #" << std::setw(4) << cg.dst
            << std::setw(10) << (cg.enabled ? "YES" : "no")
            << std::setw(12) << (cg.excitatory ? "EXC" : "INH");

        if (!cg.enabled) {
            out << "  —                [skipped: disabled]\n";
            ++skipped_disabled;
            continue;
        }

        auto it_src = snn_id.find(cg.src);
        auto it_dst = snn_id.find(cg.dst);
        if (it_src == snn_id.end() || it_dst == snn_id.end()) {
            out << "  —                [skipped: node not in SNN]\n";
            ++skipped_missing;
            continue;
        }

        int ssrc = it_src->second, sdst = it_dst->second;
        int syn_id = net.addSynapse(ssrc, sdst, cg.excitatory);
        if (syn_id < 0) {
            out << "  —                [REJECTED by addSynapse – check stderr]\n";
            ++skipped_missing;
        } else {
            out << "  SNN#" << std::setw(3) << ssrc << " → SNN#" << std::setw(3) << sdst
                << "  syn_id=" << syn_id << "  [added]\n";
            ++added;
        }
    }

    out << "\nSummary: "
        << added << " synapses added, "
        << skipped_disabled << " disabled genes skipped, "
        << skipped_missing  << " skipped (node missing).\n";

    // ----------------------------------------------------------------
    // 3. Print the SNN Network topology independently for verification
    // ----------------------------------------------------------------
    out << "\n── SNN Network (as seen by the simulator) ────────────\n";
    out << "Input  neurons (" << net.getInputCount()  << "): ";
    for (int id : net.getInputNeuronIds())  out << id << ' ';
    out << '\n';
    out << "Hidden neurons (" << (net.getNeuronCount()
                                  - net.getInputCount()
                                  - net.getOutputCount()) << "): ";
    // hidden = all neurons not in input or output lists
    {
        std::vector<int> all_in  = net.getInputNeuronIds();
        std::vector<int> all_out = net.getOutputNeuronIds();
        for (int id = 0; id < static_cast<int>(net.getNeuronCount()); ++id) {
            bool is_in  = std::find(all_in.begin(),  all_in.end(),  id) != all_in.end();
            bool is_out = std::find(all_out.begin(), all_out.end(), id) != all_out.end();
            if (!is_in && !is_out) out << id << ' ';
        }
    }
    out << '\n';
    out << "Output neurons (" << net.getOutputCount() << "): ";
    for (int id : net.getOutputNeuronIds()) out << id << ' ';
    out << '\n';

    out << "\nSynapses (" << net.getSynapseCount() << "):\n";
    for (size_t i = 0; i < net.getSynapseCount(); ++i) {
        const Synapse& s = net.getSynapse(static_cast<int>(i));
        out << "  [" << i << "] SNN#" << s.getPreNeuronId()
            << " →(" << (s.isExcitatory() ? "EXC" : "INH") << ") SNN#"
            << s.getPostNeuronId()
            << (s.isEnabled() ? "" : "  [DISABLED]")
            << '\n';
    }

    out << "╚══════════════════════════════════════════════════════╝\n\n";
}

} // namespace wann

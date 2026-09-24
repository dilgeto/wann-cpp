#pragma once
#include "Hyperparams.h"
#include "Ind.h"

#include <vector>

namespace wann {

// Mutation operators, in the order Wann::topoMutate rolls them (also the
// MutStats / ChildInfo::op index).
enum class MutOp { AddConn = 0, AddNode = 1, Enable = 2, MutAct = 3, ToggleExcitatory = 4 };
constexpr int N_MUT_OPS = 5;

const char* mutOpName(MutOp op);

// One genome reachable from a parent by exactly one application of one
// operator.
struct Neighbor {
    MutOp  op;
    Ind    ind;          // genes + express() already run
    // What was done (node/innovation ids of the parent's genome):
    //   AddConn          a = src node,   b = dst node
    //   AddNode          a = split src,  b = split dst,  c = new node's activation
    //   Enable           a = src node,   b = dst node
    //   MutAct           a = node id,    b = new activation, c = old activation
    //   ToggleExcitatory a = src node,   b = dst node,  c = 1 if it became excitatory
    int    a = -1, b = -1, c = -1;
    // Probability that ONE call of Wann::topoMutate on the parent yields
    // exactly this neighbour: P(operator rolled) * P(this choice | operator).
    // The mass that does not add up to 1 over the whole neighbourhood is the
    // chance the roll is a no-op (operator has no valid move).
    double prob = 0.0;
    // False when express() failed (the structural graph, disabled edges
    // included, has a cycle). Training still simulates such genomes
    // (SnnCarTask::buildNetwork reads the genes), but they carry nConn = 0.
    bool   exprOk = true;
};

// All single-mutation neighbours of `parent`, mirroring Wann::mutAddConn /
// mutAddNode / the enable, mutAct and toggle branches of Wann::topoMutate
// (same validity rules, same filters, same quirks) but enumerating every
// outcome instead of sampling one. Wann's mutation operators are private and
// draw from the RNG in a fixed order, so they are re-implemented here rather
// than refactored; if either side changes the other must follow.
std::vector<Neighbor> enumerateNeighbors(const Ind& parent, const Hyperparams& hyp);

} // namespace wann

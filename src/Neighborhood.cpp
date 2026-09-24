#include "../include/wann/Neighborhood.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Every block below mirrors the operator of the same name in src/Wann.cpp.
// Where that code has a quirk it is reproduced on purpose: the point of this
// module is to know exactly which genomes evolution can reach in one step,
// so "fixing" a quirk here would make the analysis describe a different
// algorithm.

namespace {

using namespace wann;

std::unordered_set<int> outputIds(const std::vector<NodeGene>& nodes) {
    std::unordered_set<int> ids;
    for (const auto& n : nodes)
        if (n.type == 2) ids.insert(n.id);
    return ids;
}

// Enabled excitatory / inhibitory incoming counts per output node id.
std::unordered_map<int, std::pair<int,int>> outputExcInh(
    const std::vector<ConnGene>& conns, const std::unordered_set<int>& outs)
{
    std::unordered_map<int, std::pair<int,int>> counts;
    for (const auto& c : conns) {
        if (!c.enabled || !outs.count(c.dst)) continue;
        auto& [exc, inh] = counts[c.dst];
        (c.excitatory ? exc : inh)++;
    }
    return counts;
}

int maxInnov(const std::vector<ConnGene>& conns) {
    int m = -1;
    for (const auto& c : conns) m = std::max(m, c.innov);
    return m;
}

int maxNodeId(const std::vector<NodeGene>& nodes) {
    int m = -1;
    for (const auto& n : nodes) m = std::max(m, n.id);
    return m;
}

Neighbor makeNeighbor(MutOp op, std::vector<ConnGene> conns, std::vector<NodeGene> nodes,
                      int a, int b, int c, double prob)
{
    Neighbor nb;
    nb.op   = op;
    nb.ind  = Ind(conns, nodes);
    nb.a = a; nb.b = b; nb.c = c;
    nb.prob   = prob;
    nb.exprOk = nb.ind.express();
    return nb;
}

// --- Wann::mutAddConn ------------------------------------------------------
// The source is drawn uniformly among nodes that still have a free
// destination, then the destination uniformly among that source's free
// ones, so P(pair) = 1 / (#sources with a free destination) / #free(src).
void addConnNeighbors(const Ind& parent, double pOp, std::vector<Neighbor>& out) {
    const auto& nodes = parent.nodes;
    const auto& conns = parent.conns;
    const int nNodes = static_cast<int>(nodes.size());
    int nIns = 0, nOuts = 0;
    for (const auto& n : nodes) {
        if (n.type == 1 || n.type == 4) ++nIns;
        if (n.type == 2)                ++nOuts;
    }

    auto [order, wMat] = getNodeOrder(nodes, conns);
    if (order.empty()) return;  // cycle: mutAddConn gives up too

    const int nHidden = nNodes - nIns - nOuts;
    std::vector<double> hMat(nHidden * nHidden, 0.0);
    for (int i = 0; i < nHidden; ++i)
        for (int j = 0; j < nHidden; ++j)
            hMat[i * nHidden + j] = wMat[(nIns + i) * nNodes + (nIns + j)];
    auto hLay = getLayer(hMat, nHidden);

    // Inputs/bias = layer 0, hidden = hLay+1, outputs = lastLayer. Note
    // lastLayer equals the deepest hidden layer, not one past it, so a
    // hidden node in the deepest layer has no output as a valid AddConn
    // destination (only AddNode can wire it to an output).
    double lastLayer = 1.0;
    for (double l : hLay) lastLayer = std::max(lastLayer, l + 1.0);

    struct NodeKey { int id; double layer; };
    std::vector<NodeKey> nodeKey(nNodes);
    for (int i = 0; i < nIns;    ++i) nodeKey[i] = {nodes[order[i]].id, 0.0};
    for (int i = 0; i < nHidden; ++i) nodeKey[nIns + i] = {nodes[order[nIns + i]].id, hLay[i] + 1.0};
    for (int i = 0; i < nOuts;   ++i) nodeKey[nIns + nHidden + i] = {nodes[order[nIns + nHidden + i]].id, lastLayer};

    // Free (src, dst) pairs. "Existing" counts every gene, disabled or not.
    std::unordered_set<long long> exists;
    for (const auto& c : conns)
        exists.insert(static_cast<long long>(c.src) * 1000003LL + c.dst);

    std::vector<std::pair<int, std::vector<int>>> avail;  // (srcId, dstIds)
    for (int s = 0; s < nNodes; ++s) {
        std::vector<int> dsts;
        for (int k = 0; k < nNodes; ++k) {
            if (nodeKey[k].layer <= nodeKey[s].layer) continue;
            long long key = static_cast<long long>(nodeKey[s].id) * 1000003LL + nodeKey[k].id;
            if (exists.count(key)) continue;
            dsts.push_back(nodeKey[k].id);
        }
        if (!dsts.empty()) avail.push_back({nodeKey[s].id, std::move(dsts)});
    }
    if (avail.empty()) return;

    const int newInnov = maxInnov(conns) + 1;
    for (const auto& [srcId, dsts] : avail)
        for (int dstId : dsts) {
            auto c2 = conns;
            c2.push_back({newInnov, srcId, dstId, 1.0, true});
            double p = pOp / static_cast<double>(avail.size()) / static_cast<double>(dsts.size());
            out.push_back(makeNeighbor(MutOp::AddConn, std::move(c2), nodes, srcId, dstId, -1, p));
        }
}

// --- Wann::mutAddNode ------------------------------------------------------
void addNodeNeighbors(const Ind& parent, const Hyperparams& hyp, double pOp,
                      std::vector<Neighbor>& out)
{
    const auto& conns = parent.conns;
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) active.push_back(i);
    if (active.empty() || hyp.ann_actRange.empty()) return;

    const int newNodeId = maxNodeId(parent.nodes) + 1;
    const int nextInnov = maxInnov(conns) + 1;
    const double p = pOp / static_cast<double>(active.size())
                         / static_cast<double>(hyp.ann_actRange.size());

    for (int ci : active)
        for (int act : hyp.ann_actRange) {
            auto c2 = conns;
            auto n2 = parent.nodes;

            ConnGene connTo   = c2[ci];
            connTo.innov      = nextInnov;
            connTo.dst        = newNodeId;
            connTo.weight     = 1.0;
            connTo.enabled    = true;

            ConnGene connFrom = c2[ci];
            connFrom.innov    = nextInnov + 1;
            connFrom.src      = newNodeId;
            connFrom.weight   = c2[ci].weight;
            connFrom.enabled  = true;

            c2[ci].enabled = false;
            n2.push_back({newNodeId, 3, act});
            c2.push_back(connTo);
            c2.push_back(connFrom);
            out.push_back(makeNeighbor(MutOp::AddNode, std::move(c2), std::move(n2),
                                       conns[ci].src, conns[ci].dst, act, p));
        }
}

// --- topoMutate case 3: enable a disabled connection -----------------------
void enableNeighbors(const Ind& parent, const Hyperparams& hyp, double pOp,
                     std::vector<Neighbor>& out)
{
    const auto& conns = parent.conns;
    std::vector<int> disabled;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (!conns[i].enabled) disabled.push_back(i);

    std::vector<int> safe;
    if (hyp.require_output_excitatory_majority) {
        auto outs   = outputIds(parent.nodes);
        auto counts = outputExcInh(conns, outs);
        for (int i : disabled) {
            if (!conns[i].excitatory && outs.count(conns[i].dst)) {
                const auto& [exc, inh] = counts[conns[i].dst];
                if (exc <= inh + 1) continue;  // would tie/flip the majority
            }
            safe.push_back(i);
        }
    } else {
        safe = disabled;
    }
    if (safe.empty()) return;

    const double p = pOp / static_cast<double>(safe.size());
    for (int i : safe) {
        auto c2 = conns;
        c2[i].enabled = true;
        out.push_back(makeNeighbor(MutOp::Enable, std::move(c2), parent.nodes,
                                   conns[i].src, conns[i].dst, -1, p));
    }
}

// --- topoMutate case 4: change one node's activation -----------------------
void mutActNeighbors(const Ind& parent, const Hyperparams& hyp, double pOp,
                     std::vector<Neighbor>& out)
{
    const auto& nodes = parent.nodes;
    if (nodes.empty()) return;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        std::vector<int> pool;
        for (int a : hyp.ann_actRange)
            if (a != nodes[i].activation) pool.push_back(a);
        if (pool.empty()) continue;
        const double p = pOp / static_cast<double>(nodes.size()) / static_cast<double>(pool.size());
        for (int a : pool) {
            auto n2 = nodes;
            n2[i].activation = a;
            out.push_back(makeNeighbor(MutOp::MutAct, parent.conns, std::move(n2),
                                       nodes[i].id, a, nodes[i].activation, p));
        }
    }
}

// --- Wann::mutToggleExcitatory ---------------------------------------------
void toggleNeighbors(const Ind& parent, const Hyperparams& hyp, double pOp,
                     std::vector<Neighbor>& out)
{
    const auto& conns = parent.conns;
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) active.push_back(i);

    std::vector<int> safe;
    if (hyp.require_output_excitatory_majority) {
        auto outs   = outputIds(parent.nodes);
        auto counts = outputExcInh(conns, outs);
        for (int i : active) {
            if (conns[i].excitatory && outs.count(conns[i].dst)) {
                const auto& [exc, inh] = counts[conns[i].dst];
                if (exc - 1 <= inh) continue;  // would break the excitatory majority
            }
            safe.push_back(i);
        }
    } else {
        safe = active;
    }
    if (safe.empty()) return;

    const double p = pOp / static_cast<double>(safe.size());
    for (int i : safe) {
        auto c2 = conns;
        c2[i].excitatory = !c2[i].excitatory;
        const int nowExcitatory = c2[i].excitatory ? 1 : 0;
        out.push_back(makeNeighbor(MutOp::ToggleExcitatory, std::move(c2), parent.nodes,
                                   conns[i].src, conns[i].dst, nowExcitatory, p));
    }
}

} // namespace

namespace wann {

const char* mutOpName(MutOp op) {
    switch (op) {
        case MutOp::AddConn:          return "addConn";
        case MutOp::AddNode:          return "addNode";
        case MutOp::Enable:           return "enable";
        case MutOp::MutAct:           return "mutAct";
        case MutOp::ToggleExcitatory: return "toggleExcitatory";
    }
    return "?";
}

std::vector<Neighbor> enumerateNeighbors(const Ind& parent, const Hyperparams& hyp) {
    const double total = hyp.prob_addConn + hyp.prob_addNode + hyp.prob_enable
                       + hyp.prob_mutAct + hyp.prob_toggleExcitatory;
    std::vector<Neighbor> out;
    if (total <= 0.0) return out;

    addConnNeighbors(parent, hyp.prob_addConn / total, out);
    addNodeNeighbors(parent, hyp, hyp.prob_addNode / total, out);
    enableNeighbors (parent, hyp, hyp.prob_enable / total, out);
    mutActNeighbors (parent, hyp, hyp.prob_mutAct / total, out);
    toggleNeighbors (parent, hyp, hyp.prob_toggleExcitatory / total, out);
    return out;
}

} // namespace wann

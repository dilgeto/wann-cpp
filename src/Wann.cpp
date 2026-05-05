#include "../include/wann/Wann.h"
#include "../include/wann/NsgaSort.h"
#include "../include/wann/Random.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace wann {

// =========================================================================
// Construction
// =========================================================================

Wann::Wann(const Hyperparams& hyp) : p(hyp) {}

// =========================================================================
// ask / tell
// =========================================================================

std::vector<Ind>& Wann::ask() {
    if (pop.empty()) {
        initPop();
    } else {
        probMoo();
        speciate();
        evolvePop();
    }
    ++gen;
    return pop;
}

void Wann::tell(const std::vector<std::vector<double>>& reward) {
    for (int i = 0; i < static_cast<int>(reward.size()); ++i) {
        double sum = 0.0, best = -std::numeric_limits<double>::infinity();
        for (double r : reward[i]) {
            sum += r;
            best = std::max(best, r);
        }
        pop[i].fitness = sum / static_cast<double>(reward[i].size());
        pop[i].fitMax  = best;
        // nConn is set by express() and not overwritten here.
    }
}

// =========================================================================
// initPop
// Mirrors Python Wann::initPop exactly.
// =========================================================================
void Wann::initPop() {
    // ----- Build base nodes -----
    //   ID=0          : bias (type 4)
    //   ID=1..nIn     : inputs (type 1)
    //   ID=nIn+1..nIn+nOut : outputs (type 2)
    const int nIn  = p.ann_nInput;
    const int nOut = p.ann_nOutput;
    const int nNodes = 1 + nIn + nOut;

    std::vector<NodeGene> baseNodes;
    baseNodes.reserve(nNodes);
    baseNodes.push_back({0, 4, p.ann_initAct});              // bias
    for (int i = 1; i <= nIn;            ++i)
        baseNodes.push_back({i, 1, p.ann_initAct});          // inputs
    for (int i = nIn+1; i <= nIn+nOut;   ++i)
        baseNodes.push_back({i, 2, p.ann_initAct});          // outputs

    // ----- Build base connections: all (bias+inputs) → outputs -----
    // Innovation IDs start at 0.
    std::vector<ConnGene> baseConns;
    int innovIdx = 0;
    for (int src = 0; src <= nIn; ++src)           // 0..nIn (bias + inputs)
        for (int dst = nIn+1; dst <= nIn+nOut; ++dst)
            baseConns.push_back({innovIdx++, src, dst, 1.0, true});

    // ----- Create population -----
    pop.clear();
    pop.reserve(p.popSize);
    for (int i = 0; i < p.popSize; ++i) {
        Ind ind(baseConns, baseNodes);
        // Randomise enable flags; weight value irrelevant (WANN uses shared w).
        for (auto& c : ind.conns) {
            c.weight  = 1.0;
            c.enabled = randDouble() < p.prob_initEnable;
        }
        ind.express();
        ind.birth = 0;
        pop.push_back(std::move(ind));
    }

    // ----- Build initial innovation record -----
    innov.clear();
    innov.reserve(static_cast<int>(baseConns.size()));
    for (const auto& c : baseConns)
        innov.push_back({c.innov, c.src, c.dst, -1, 0});
}

// =========================================================================
// probMoo – rank population via Pareto dominance (NSGA-II).
// With probability alg_probMoo uses objectives [meanFit, 1/nConn];
// otherwise uses [meanFit, maxFit].
// =========================================================================
void Wann::probMoo() {
    const int n = static_cast<int>(pop.size());

    std::vector<double> meanFit(n), maxFit(n), invConn(n);
    for (int i = 0; i < n; ++i) {
        meanFit[i] = pop[i].fitness;
        maxFit[i]  = pop[i].fitMax;
        int nc     = std::max(pop[i].nConn, 1);  // avoid division by 0
        invConn[i] = 1.0 / static_cast<double>(nc);
    }

    // Build 2-objective matrix to pass to nsga_sort.
    std::vector<std::vector<double>> objVals(n, std::vector<double>(2));
    bool useMooConn = (p.alg_probMoo > randDouble());
    for (int i = 0; i < n; ++i) {
        objVals[i][0] = meanFit[i];
        objVals[i][1] = useMooConn ? invConn[i] : maxFit[i];
    }

    auto rank = nsga_sort(objVals);
    for (int i = 0; i < n; ++i)
        pop[i].rank = rank[i];
}

// =========================================================================
// speciate – WANN uses no speciation; entire population = one species.
// =========================================================================
void Wann::speciate() {
    species.resize(1);
    species[0].seedIdx  = 0;
    species[0].nOffspring = p.popSize;
    species[0].memberIdx.resize(pop.size());
    std::iota(species[0].memberIdx.begin(), species[0].memberIdx.end(), 0);
    for (auto& ind : pop) ind.species = 0;
}

// =========================================================================
// evolvePop – create next generation from all species.
// =========================================================================
void Wann::evolvePop() {
    std::vector<Ind> newPop;
    newPop.reserve(p.popSize);
    for (const auto& sp : species) {
        auto children = recombine(sp);
        for (auto& c : children) newPop.push_back(std::move(c));
    }
    pop = std::move(newPop);
}

// =========================================================================
// recombine – produce offspring from one species.
// Mirrors Python _variation.py::recombine.
// =========================================================================
std::vector<Ind> Wann::recombine(const Species& sp) {
    // Collect pointers in rank order (members already have .rank set).
    std::vector<Ind*> members;
    members.reserve(sp.memberIdx.size());
    for (int idx : sp.memberIdx) members.push_back(&pop[idx]);
    std::sort(members.begin(), members.end(),
              [](const Ind* a, const Ind* b){ return a->rank < b->rank; });

    int nOffspring = sp.nOffspring;

    // Cull: remove bottom fraction from breeding pool.
    int toCull = static_cast<int>(std::floor(p.select_cullRatio
                                             * static_cast<double>(members.size())));
    if (toCull > 0) members.resize(members.size() - toCull);

    const int poolSize = static_cast<int>(members.size());
    std::vector<Ind> children;
    children.reserve(nOffspring);

    // Elitism: keep top fraction unchanged.
    int nElites = static_cast<int>(std::floor(static_cast<double>(poolSize)
                                              * p.select_eliteRatio));
    for (int i = 0; i < nElites && nOffspring > 0; ++i, --nOffspring)
        children.push_back(*members[i]);

    // Tournament selection: pick best index from a random set.
    // Since members are sorted by rank (ascending = better), lower index = fitter.
    auto tournament = [&]() -> int {
        int best = randInt(0, poolSize - 1);
        for (int t = 1; t < p.select_tournSize; ++t)
            best = std::min(best, randInt(0, poolSize - 1));
        return best;
    };

    for (int i = 0; i < nOffspring; ++i) {
        int pa = tournament();
        int pb = tournament();
        if (pa > pb) std::swap(pa, pb);  // pa = more fit (lower index)

        Ind child;
        if (randDouble() > p.prob_crossover) {
            // Mutation only: copy from fitter parent.
            child = Ind(members[pa]->conns, members[pa]->nodes);
        } else {
            child = crossover(*members[pa], *members[pb]);
        }

        topoMutate(child);
        child.express();
        children.push_back(std::move(child));
    }
    return children;
}

// =========================================================================
// crossover – combine genes of two individuals.
// Mirrors Python _variation.py::crossover.
// =========================================================================
Ind Wann::crossover(const Ind& parentA, const Ind& parentB) {
    // Inherit all structure from the fitter parent (A).
    Ind child(parentA.conns, parentA.nodes);

    // Find matching innovation numbers between parents.
    // Replace child weights with parentB's weights with prob 0.5.
    for (int ia = 0; ia < static_cast<int>(child.conns.size()); ++ia) {
        int innovA = child.conns[ia].innov;
        // Search parentB for matching innovation.
        for (const auto& cb : parentB.conns) {
            if (cb.innov == innovA && randDouble() < 0.5) {
                child.conns[ia].weight     = cb.weight;
                child.conns[ia].excitatory = cb.excitatory;
                break;
            }
        }
    }
    return child;
}

// =========================================================================
// mutAddConn – add one new feed-forward connection.
// Mirrors Python _variation.py::mutAddConn.
// =========================================================================
void Wann::mutAddConn(std::vector<ConnGene>& conns,
                      const std::vector<NodeGene>& nodes)
{
    const int nNodes = static_cast<int>(nodes.size());
    int nIns = 0, nOuts = 0;
    for (const auto& n : nodes) {
        if (n.type == 1 || n.type == 4) ++nIns;
        if (n.type == 2)                ++nOuts;
    }

    // Topological sort to get the ordered weight matrix.
    auto [order, wMat] = getNodeOrder(nodes, conns);
    if (order.empty()) return;  // cycle, skip

    // Extract hidden-only submatrix to compute layers.
    const int nHidden = nNodes - nIns - nOuts;
    std::vector<double> hMat(nHidden * nHidden, 0.0);
    for (int i = 0; i < nHidden; ++i)
        for (int j = 0; j < nHidden; ++j)
            hMat[i * nHidden + j] = wMat[(nIns+i) * nNodes + (nIns+j)];
        // Note: hidden starts at nIns in ordered matrix (outputs come last).

    auto hLay = getLayer(hMat, nHidden);

    // Assign layers to all nodes (ordered by topological sort).
    // Inputs/bias = layer 0, hidden = hLay+1, outputs = lastLayer.
    double lastLayer = 1.0;
    for (double l : hLay) lastLayer = std::max(lastLayer, l + 1.0);

    struct NodeKey { int id; double layer; };
    std::vector<NodeKey> nodeKey(nNodes);
    for (int i = 0; i < nIns;            ++i)
        nodeKey[i] = {nodes[order[i]].id, 0.0};
    for (int i = 0; i < nHidden;         ++i)
        nodeKey[nIns + i] = {nodes[order[nIns + i]].id, hLay[i] + 1.0};
    for (int i = 0; i < nOuts;           ++i)
        nodeKey[nIns+nHidden+i] = {nodes[order[nIns+nHidden+i]].id, lastLayer};

    // Build quick-lookup: node ID → index in nodeKey.
    std::unordered_map<int,int> idToNK;
    for (int i = 0; i < nNodes; ++i) idToNK[nodeKey[i].id] = i;

    // Try each source in random order until we find a valid new connection.
    std::vector<int> sources(nNodes);
    std::iota(sources.begin(), sources.end(), 0);
    wann::shuffle(sources);

    for (int srcIdx : sources) {
        double srcLayer = nodeKey[srcIdx].layer;

        // Candidate destinations: strictly higher layer.
        std::vector<int> dest;
        for (int k = 0; k < nNodes; ++k)
            if (nodeKey[k].layer > srcLayer) dest.push_back(k);

        // Remove already-existing connections from this source.
        int srcId = nodeKey[srcIdx].id;
        std::unordered_set<int> existDst;
        for (const auto& c : conns)
            if (c.src == srcId) {
                auto it = idToNK.find(c.dst);
                if (it != idToNK.end()) existDst.insert(it->second);
            }
        dest.erase(std::remove_if(dest.begin(), dest.end(),
                                  [&](int k){ return existDst.count(k) > 0; }),
                   dest.end());

        if (dest.empty()) continue;

        wann::shuffle(dest);
        int dstId = nodeKey[dest[0]].id;

        int newInnov = innov.back().innov + 1;
        conns.push_back({newInnov, srcId, dstId, 1.0, true});
        innov.push_back({newInnov, srcId, dstId, -1, gen});
        break;
    }
}

// =========================================================================
// mutAddNode – split an existing connection with a new hidden node.
// Mirrors Python _variation.py::mutAddNode.
// =========================================================================
void Wann::mutAddNode(std::vector<ConnGene>& conns,
                      std::vector<NodeGene>& nodes)
{
    // Find active connections.
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) active.push_back(i);
    if (active.empty()) return;

    int connSplit = active[randInt(0, static_cast<int>(active.size()) - 1)];

    // Choose random activation from allowed range.
    const auto& actRange = p.ann_actRange;
    int newActivation = actRange[randInt(0, static_cast<int>(actRange.size()) - 1)];

    // New node ID = max destination ID seen in innovation record + 1.
    int newNodeId = 0;
    for (const auto& ir : innov) newNodeId = std::max(newNodeId, ir.dst);
    ++newNodeId;

    int nextInnov = innov.back().innov + 1;

    // connTo  : original source → new node (weight = 1)
    ConnGene connTo = conns[connSplit];
    connTo.innov  = nextInnov;
    connTo.dst    = newNodeId;
    connTo.weight = 1.0;
    connTo.enabled= true;

    // connFrom: new node → original destination (weight = original weight)
    ConnGene connFrom = conns[connSplit];
    connFrom.innov  = nextInnov + 1;
    connFrom.src    = newNodeId;
    connFrom.weight = conns[connSplit].weight;
    connFrom.enabled= true;

    // Disable original connection.
    conns[connSplit].enabled = false;

    // Record both innovations.
    innov.push_back({nextInnov,     connTo.src,   connTo.dst,   newNodeId, gen});
    innov.push_back({nextInnov + 1, connFrom.src, connFrom.dst, -1,        gen});

    // Add new structures.
    nodes.push_back({newNodeId, 3, newActivation});
    conns.push_back(connTo);
    conns.push_back(connFrom);
}

// =========================================================================
// mutToggleExcitatory – flip the excitatory/inhibitory polarity of one
// randomly chosen enabled connection.
// =========================================================================
void Wann::mutToggleExcitatory(std::vector<ConnGene>& conns) {
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) active.push_back(i);
    if (active.empty()) return;
    int idx = active[randInt(0, static_cast<int>(active.size()) - 1)];
    conns[idx].excitatory = !conns[idx].excitatory;
}

// =========================================================================
// topoMutate – choose exactly one topological mutation via roulette wheel.
// Options: [addConn, addNode, enable, mutAct, toggleExcitatory]
// =========================================================================
void Wann::topoMutate(Ind& child) {
    auto& conns = child.conns;
    auto& nodes = child.nodes;

    // Roulette weights: [addConn, addNode, enable, mutAct, toggleExcitatory]
    double weights[5] = {
        p.prob_addConn,
        p.prob_addNode,
        p.prob_enable,
        p.prob_mutAct,
        p.prob_toggleExcitatory
    };
    double total = weights[0]+weights[1]+weights[2]+weights[3]+weights[4];
    double spin  = randDouble(0.0, total);

    // Default = last option (toggleExcitatory); loop checks first 4 slots.
    int choice = 5;
    double slot = weights[0];
    for (int i = 1; i < 5; ++i) {
        if (spin < slot) { choice = i; break; }
        slot += weights[i];
    }

    switch (choice) {
        case 1:  // Add connection
            mutAddConn(conns, nodes);
            break;

        case 2:  // Add node
            mutAddNode(conns, nodes);
            break;

        case 3: { // Enable a disabled connection
            std::vector<int> disabled;
            for (int i = 0; i < static_cast<int>(conns.size()); ++i)
                if (!conns[i].enabled) disabled.push_back(i);
            if (!disabled.empty())
                conns[disabled[randInt(0, static_cast<int>(disabled.size()) - 1)]].enabled = true;
            break;
        }

        case 4: { // Mutate activation (NeuronType) of any node
            // In the SNN context every neuron type is meaningful (Izhikevich
            // parameters differ), so we target all nodes, not just hidden ones.
            if (!nodes.empty()) {
                int mutIdx = randInt(0, static_cast<int>(nodes.size()) - 1);
                int curAct = nodes[mutIdx].activation;
                const auto& actRange = p.ann_actRange;
                std::vector<int> pool;
                for (int a : actRange) if (a != curAct) pool.push_back(a);
                if (!pool.empty())
                    nodes[mutIdx].activation = pool[randInt(0, static_cast<int>(pool.size()) - 1)];
            }
            break;
        }

        case 5:  // Toggle excitatory/inhibitory polarity of one connection
            mutToggleExcitatory(conns);
            break;

        default: break;
    }

    child.birth = gen;
}

} // namespace wann

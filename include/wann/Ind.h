#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace wann {

// -------------------------------------------------------------------------
// Gene types
// -------------------------------------------------------------------------

struct NodeGene {
    int id;
    int type;        // 1=input  2=output  3=hidden  4=bias
    int activation;  // 1..11 (see applyAct)
};

struct ConnGene {
    int    innov;
    int    src;     // source node ID
    int    dst;     // destination node ID
    double weight;  // may be NaN when disabled (to preserve structure)
    bool   enabled;
};

// Tracks structural innovations across the population (NE-style).
struct InnovRecord {
    int innov;
    int src;
    int dst;
    int newNode;  // -1 = no new node created; else = new node ID
    int gen;
};

// -------------------------------------------------------------------------
// Individual
// -------------------------------------------------------------------------

class Ind {
public:
    Ind() = default;
    Ind(const std::vector<ConnGene>& c, const std::vector<NodeGene>& n);

    // --- Genome ---
    std::vector<NodeGene> nodes;
    std::vector<ConnGene> conns;
    int nInput  = 0;
    int nOutput = 0;

    // --- Expressed network (set by express()) ---
    std::vector<double> wMat;  // N×N weight matrix, row-major (NaN = no conn)
    std::vector<double> wVec;  // wMat flattened, NaN replaced by 0
    std::vector<int>    aVec;  // activation function per node (topo order)
    int nConn  = 0;            // number of active connections
    int nNodes = 0;            // total nodes in expressed network

    // --- Fitness (assigned externally by tell()) ---
    double fitness = 0.0;
    double fitMax  = 0.0;
    int    rank    = 0;
    int    birth   = 0;
    int    species = 0;

    // Count enabled connection genes (gene-level, before expression).
    int nConns() const;

    // Build wMat / wVec / aVec from genes. Returns false if cycle detected.
    bool express();
};

// -------------------------------------------------------------------------
// ANN topology helpers
// -------------------------------------------------------------------------

// Topological sort of nodes; builds the ordered N×N weight matrix.
// Returns ({}, {}) on cycle detection.
std::pair<std::vector<int>, std::vector<double>>
getNodeOrder(const std::vector<NodeGene>& nodes,
             const std::vector<ConnGene>& conns);

// Layer assignment for hidden-only adjacency (used by mutAddConn).
// Input: binary/weighted NxN hidden-to-hidden matrix (flat, row-major).
// Returns layer index for each hidden node (0-based).
std::vector<double> getLayer(const std::vector<double>& wMat, int n);

// -------------------------------------------------------------------------
// ANN forward pass
// -------------------------------------------------------------------------

// Activation function lookup (matches Python applyAct).
double applyAct(int actId, double x);

// Feed-forward pass for a single input pattern.
// weights: N*N flattened weight matrix (NaN treated as 0).
// aVec   : activation function index per node.
// Returns nOutput values.
std::vector<double> act(const std::vector<double>& weights,
                        const std::vector<int>&    aVec,
                        int nInput, int nOutput,
                        const std::vector<double>& inPattern);

// Set a single shared weight value on the connection topology.
// wVec  : flattened weight vector from express() (non-zero = connected).
// wVal  : shared weight value to assign to every connection.
// Returns N×N weight matrix.
std::vector<double> setWeights(const std::vector<double>& wVec, double wVal);

// -------------------------------------------------------------------------
// File I/O  (matches Python exportNet / importNet)
// -------------------------------------------------------------------------

// Save as [N × (N+1)] CSV: weight columns + activation column.
void exportNet(const std::string& filename,
               const std::vector<double>& wMat, int N,
               const std::vector<int>& aVec);

// Load a network saved by exportNet.
// Returns {wVec, aVec, wKey} where wKey holds indices of non-zero weights.
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
importNet(const std::string& filename);

} // namespace wann

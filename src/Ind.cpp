#include "../include/wann/Ind.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace wann {

// -------------------------------------------------------------------------
// Ind
// -------------------------------------------------------------------------

Ind::Ind(const std::vector<ConnGene>& c, const std::vector<NodeGene>& n)
    : nodes(n), conns(c)
{
    for (const auto& nd : nodes) {
        if (nd.type == 1) ++nInput;
        if (nd.type == 2) ++nOutput;
    }
}

int Ind::nConns() const {
    int cnt = 0;
    for (const auto& c : conns) cnt += c.enabled ? 1 : 0;
    return cnt;
}

bool Ind::express() {
    auto [Q, wm] = getNodeOrder(nodes, conns);
    if (Q.empty()) return false;

    nNodes = static_cast<int>(nodes.size());
    wMat   = std::move(wm);

    // Build activation vector in topological order.
    aVec.resize(nNodes);
    for (int i = 0; i < nNodes; ++i)
        aVec[i] = nodes[Q[i]].activation;

    // Build wVec: NaN → 0, count non-zero connections.
    wVec.resize(nNodes * nNodes);
    nConn = 0;
    for (int i = 0; i < nNodes * nNodes; ++i) {
        double w = wMat[i];
        if (std::isnan(w)) {
            wVec[i] = 0.0;
        } else {
            wVec[i] = w;
            if (w != 0.0) ++nConn;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// getNodeOrder
//
// Replicates Python ind.py::getNodeOrder exactly:
//   - Disabled connections get NaN weight (treated as structural edges).
//   - Hidden nodes are topologically sorted via Kahn's algorithm.
//   - The returned wMat is reordered as [inputs/bias | hidden_sorted | outputs].
//   - Hidden-to-hidden sub-block is binarised (0/1) as a side-effect, which
//     is harmless because WANN replaces all weights with a shared scalar.
// -------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<double>>
getNodeOrder(const std::vector<NodeGene>& nodes,
             const std::vector<ConnGene>& conns)
{
    const int nNodes = static_cast<int>(nodes.size());

    int nIns  = 0, nOuts = 0;
    for (const auto& n : nodes) {
        if (n.type == 1 || n.type == 4) ++nIns;
        if (n.type == 2)                ++nOuts;
    }
    const int nHidden = nNodes - nIns - nOuts;

    // Map node ID → array index.
    std::unordered_map<int,int> idToIdx;
    idToIdx.reserve(nNodes);
    for (int i = 0; i < nNodes; ++i)
        idToIdx[nodes[i].id] = i;

    // Build N×N weight matrix.
    // Disabled connections carry NaN (preserves structural edge for topo sort).
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> wMat(nNodes * nNodes, 0.0);
    for (const auto& c : conns) {
        int si = idToIdx.at(c.src);
        int di = idToIdx.at(c.dst);
        wMat[si * nNodes + di] = c.enabled ? c.weight : NaN;
    }

    // Trivial case: no hidden nodes, no sort needed.
    if (nHidden == 0) {
        std::vector<int> Q(nNodes);
        std::iota(Q.begin(), Q.end(), 0);
        return {Q, wMat};
    }

    // Build binary hidden-to-hidden adjacency matrix.
    // NaN != 0 is true in C++ (IEEE 754), so disabled conns also create edges.
    std::vector<double> connMat(nHidden * nHidden, 0.0);
    for (int i = 0; i < nHidden; ++i)
        for (int j = 0; j < nHidden; ++j) {
            double w = wMat[(nIns+nOuts+i) * nNodes + (nIns+nOuts+j)];
            if (w != 0.0)   // NaN != 0 → true, handles both enabled & disabled
                connMat[i * nHidden + j] = 1.0;
        }

    // Kahn's algorithm on hidden nodes.
    std::vector<double> edgeIn(nHidden, 0.0);
    for (int j = 0; j < nHidden; ++j)
        for (int i = 0; i < nHidden; ++i)
            edgeIn[j] += connMat[i * nHidden + j];

    std::vector<int> Q;
    Q.reserve(nHidden);
    std::set<int> inQ;
    for (int i = 0; i < nHidden; ++i)
        if (edgeIn[i] == 0.0) { Q.push_back(i); inQ.insert(i); }

    for (int i = 0; i < nHidden; ++i) {
        if (i >= static_cast<int>(Q.size()))
            return {{}, {}};  // cycle detected

        // Remove outgoing edges of Q[i].
        for (int j = 0; j < nHidden; ++j)
            edgeIn[j] -= connMat[Q[i] * nHidden + j];

        // Enqueue newly zero-in-degree nodes.
        for (int j = 0; j < nHidden; ++j)
            if (edgeIn[j] == 0.0 && inQ.find(j) == inQ.end()) {
                Q.push_back(j); inQ.insert(j);
            }

        double sum = 0.0;
        for (double e : edgeIn) sum += e;
        if (sum == 0.0) break;
    }

    // Compose full ordering: [inputs/bias | sorted_hidden | outputs].
    std::vector<int> fullQ;
    fullQ.reserve(nNodes);
    for (int i = 0; i < nIns;           ++i) fullQ.push_back(i);
    for (int hi : Q)                         fullQ.push_back(hi + nIns + nOuts);
    for (int i = nIns; i < nIns+nOuts;  ++i) fullQ.push_back(i);

    // Reorder weight matrix according to fullQ.
    std::vector<double> wMatOrdered(nNodes * nNodes);
    for (int r = 0; r < nNodes; ++r)
        for (int c = 0; c < nNodes; ++c)
            wMatOrdered[r * nNodes + c] = wMat[fullQ[r] * nNodes + fullQ[c]];

    // Binarise hidden-to-hidden block (matches Python side-effect).
    for (int i = nIns; i < nIns+nHidden; ++i)
        for (int j = nIns; j < nIns+nHidden; ++j) {
            double& w = wMatOrdered[i * nNodes + j];
            if (w != 0.0) w = 1.0;
        }

    return {fullQ, wMatOrdered};
}

// -------------------------------------------------------------------------
// getLayer
// Iterative fixed-point layer assignment (mirrors Python getLayer).
// Input wMat is nxn binary matrix of hidden-to-hidden connections.
// -------------------------------------------------------------------------
std::vector<double> getLayer(const std::vector<double>& wMat, int n) {
    std::vector<double> layer(n, 0.0);
    while (true) {
        std::vector<double> prev = layer;
        for (int curr = 0; curr < n; ++curr) {
            double maxSrc = 0.0;
            for (int src = 0; src < n; ++src) {
                double w = wMat[src * n + curr];
                if (!std::isnan(w) && w != 0.0)
                    maxSrc = std::max(maxSrc, layer[src]);
            }
            layer[curr] = maxSrc + 1.0;
        }
        if (layer == prev) break;
    }
    for (double& l : layer) l -= 1.0;
    return layer;
}

// -------------------------------------------------------------------------
// applyAct – matches Python applyAct cases 1-11
// -------------------------------------------------------------------------
double applyAct(int actId, double x) {
    constexpr double Pi = 3.141592653589793;
    switch (actId) {
        case 1:  return x;                                          // Linear
        case 2:  return x > 0.0 ? 1.0 : 0.0;                      // Step
        case 3:  return std::sin(Pi * x);                          // Sin
        case 4:  return std::exp(-x * x / 2.0);                    // Gaussian
        case 5:  return std::tanh(x);                              // Tanh
        case 6:  return (std::tanh(x / 2.0) + 1.0) / 2.0;         // Sigmoid
        case 7:  return -x;                                        // Inverse
        case 8:  return std::abs(x);                               // Abs
        case 9:  return x > 0.0 ? x : 0.0;                        // ReLU
        case 10: return std::cos(Pi * x);                          // Cosine
        case 11: return x * x;                                     // Squared
        default: return x;
    }
}

// -------------------------------------------------------------------------
// act – single-sample feed-forward pass
// -------------------------------------------------------------------------
std::vector<double> act(const std::vector<double>& weights,
                        const std::vector<int>&    aVec,
                        int nInput, int nOutput,
                        const std::vector<double>& inPattern)
{
    const int nNodes = static_cast<int>(std::sqrt(static_cast<double>(weights.size())));

    std::vector<double> nodeAct(nNodes, 0.0);
    nodeAct[0] = 1.0;  // bias node
    for (int i = 0; i < nInput; ++i)
        nodeAct[1 + i] = inPattern[i];

    for (int iNode = nInput + 1; iNode < nNodes; ++iNode) {
        double rawAct = 0.0;
        for (int j = 0; j < nNodes; ++j) {
            double w = weights[j * nNodes + iNode];
            if (!std::isnan(w)) rawAct += nodeAct[j] * w;
        }
        nodeAct[iNode] = applyAct(aVec[iNode], rawAct);
    }

    std::vector<double> output(nOutput);
    for (int i = 0; i < nOutput; ++i)
        output[i] = nodeAct[nNodes - nOutput + i];
    return output;
}

// -------------------------------------------------------------------------
// setWeights – replace every connection with a single shared scalar
// -------------------------------------------------------------------------
std::vector<double> setWeights(const std::vector<double>& wVec, double wVal) {
    // wVec already has NaN→0 (from express()), so 0 = no connection.
    const int dim = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    std::vector<double> wMat(dim * dim, 0.0);
    for (int i = 0; i < dim * dim; ++i)
        if (wVec[i] != 0.0) wMat[i] = wVal;
    return wMat;
}

// -------------------------------------------------------------------------
// exportNet / importNet
// -------------------------------------------------------------------------
void exportNet(const std::string& filename,
               const std::vector<double>& wMat, int N,
               const std::vector<int>& aVec)
{
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot write: " + filename);
    f << std::scientific;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            double w = wMat[r * N + c];
            if (std::isnan(w)) f << "nan";
            else                f << w;
            f << ',';
        }
        f << static_cast<double>(aVec[r]) << '\n';
    }
}

std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
importNet(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot read: " + filename);

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<double> row;
        while (std::getline(ss, token, ',')) {
            if (token == "nan") row.push_back(std::numeric_limits<double>::quiet_NaN());
            else                row.push_back(std::stod(token));
        }
        rows.push_back(std::move(row));
    }

    const int N = static_cast<int>(rows.size());
    std::vector<double> wVec;
    std::vector<int>    aVec(N), wKey;
    wVec.reserve(N * N);

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            double w = std::isnan(rows[r][c]) ? 0.0 : rows[r][c];
            wVec.push_back(w);
            if (w != 0.0) wKey.push_back(r * N + c);
        }
        aVec[r] = static_cast<int>(rows[r][N]);
    }
    return {wVec, aVec, wKey};
}

} // namespace wann

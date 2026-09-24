#include "../include/wann/GenomeIO.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace wann {

namespace {

nlohmann::json genomeToJson(const Ind& ind) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& n : ind.nodes)
        nodes.push_back({n.id, n.type, n.activation});

    // Weights are not stored: WANN replaces every weight by the shared
    // scalar, and all genes carry 1.0 (the sign lives in `excitatory`).
    nlohmann::json conns = nlohmann::json::array();
    for (const auto& c : ind.conns)
        conns.push_back({c.innov, c.src, c.dst, c.enabled ? 1 : 0, c.excitatory ? 1 : 0});

    nlohmann::json j;
    j["nodes"] = std::move(nodes);
    j["conns"] = std::move(conns);
    return j;
}

Ind genomeFromJson(const nlohmann::json& j) {
    std::vector<NodeGene> nodes;
    for (const auto& n : j.at("nodes"))
        nodes.push_back({n.at(0).get<int>(), n.at(1).get<int>(), n.at(2).get<int>()});

    std::vector<ConnGene> conns;
    for (const auto& c : j.at("conns")) {
        ConnGene g;
        g.innov      = c.at(0).get<int>();
        g.src        = c.at(1).get<int>();
        g.dst        = c.at(2).get<int>();
        g.weight     = 1.0;
        g.enabled    = c.at(3).get<int>() != 0;
        g.excitatory = c.at(4).get<int>() != 0;
        conns.push_back(g);
    }
    Ind ind(conns, nodes);
    ind.express();
    return ind;
}

} // namespace

void savePopSnapshot(const std::string& path, const PopSnapshot& snap) {
    nlohmann::json inds = nlohmann::json::array();
    for (size_t i = 0; i < snap.pop.size(); ++i) {
        const Ind& ind = snap.pop[i];
        nlohmann::json j = genomeToJson(ind);
        j["idx"]     = static_cast<int>(i);
        j["fitness"] = ind.fitness;
        j["fitMax"]  = ind.fitMax;
        j["nConn"]   = ind.nConn;
        if (i < snap.reward.size())  j["reward"] = snap.reward[i];
        if (i < snap.lineage.size()) {
            const ChildInfo& ci = snap.lineage[i];
            j["parent"]  = ci.parent;
            j["parentB"] = ci.parentB;
            j["op"]      = ci.op;
            j["applied"] = ci.applied ? 1 : 0;
        }
        inds.push_back(std::move(j));
    }

    nlohmann::json root;
    root["gen"]      = snap.gen;
    root["evalSeed"] = snap.evalSeed;
    root["pop"]      = std::move(inds);

    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f << root.dump() << '\n';
}

PopSnapshot loadPopSnapshot(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot read: " + path);
    nlohmann::json root;
    f >> root;

    PopSnapshot snap;
    snap.gen      = root.at("gen").get<int>();
    snap.evalSeed = root.at("evalSeed").get<int>();
    for (const auto& j : root.at("pop")) {
        Ind ind = genomeFromJson(j);
        ind.fitness = j.value("fitness", 0.0);
        ind.fitMax  = j.value("fitMax", 0.0);
        snap.pop.push_back(std::move(ind));
        snap.reward.push_back(j.contains("reward")
                                  ? j.at("reward").get<std::vector<double>>()
                                  : std::vector<double>{});
        ChildInfo ci;
        if (j.contains("parent")) {
            ci.parent  = j.at("parent").get<int>();
            ci.parentB = j.at("parentB").get<int>();
            ci.op      = j.at("op").get<int>();
            ci.applied = j.at("applied").get<int>() != 0;
        }
        snap.lineage.push_back(ci);
    }
    return snap;
}

Ind genomeFromNetFile(const std::string& outFile, int nInput, int nOutput) {
    std::ifstream f(outFile);
    if (!f) throw std::runtime_error("Cannot read: " + outFile);

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
    const int nHidden = N - 1 - nInput - nOutput;
    if (nHidden < 0)
        throw std::runtime_error(outFile + ": matrix is " + std::to_string(N) +
                                 " nodes, fewer than bias+" + std::to_string(nInput) +
                                 " inputs+" + std::to_string(nOutput) + " outputs");
    for (const auto& r : rows)
        if (static_cast<int>(r.size()) != N + 1)
            throw std::runtime_error(outFile + ": not an N x (N+1) network matrix");

    // Matrix position -> node id. Ids follow the layout initPop() uses
    // (bias 0, inputs 1..nIn, outputs nIn+1..nIn+nOut, hidden after that).
    std::vector<int> idOfPos(N);
    for (int p = 0; p < N; ++p) {
        if (p <= nInput)              idOfPos[p] = p;
        else if (p < N - nOutput)     idOfPos[p] = nInput + nOutput + (p - nInput);
        else                          idOfPos[p] = nInput + 1 + (p - (N - nOutput));
    }

    // nodes[] order must be [bias, inputs, outputs, hidden] (getNodeOrder
    // relies on it), which is not the matrix order.
    std::vector<NodeGene> nodes;
    nodes.push_back({idOfPos[0], 4, static_cast<int>(rows[0][N])});
    for (int p = 1; p <= nInput; ++p)
        nodes.push_back({idOfPos[p], 1, static_cast<int>(rows[p][N])});
    for (int p = N - nOutput; p < N; ++p)
        nodes.push_back({idOfPos[p], 2, static_cast<int>(rows[p][N])});
    for (int p = nInput + 1; p < N - nOutput; ++p)
        nodes.push_back({idOfPos[p], 3, static_cast<int>(rows[p][N])});

    std::vector<ConnGene> conns;
    int innov = 0;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            double w = rows[r][c];
            if (!std::isnan(w) && w == 0.0) continue;
            ConnGene g;
            g.innov      = innov++;
            g.src        = idOfPos[r];
            g.dst        = idOfPos[c];
            g.weight     = 1.0;
            g.enabled    = !std::isnan(w);
            g.excitatory = std::isnan(w) || w > 0.0;
            conns.push_back(g);
        }

    Ind ind(conns, nodes);
    ind.express();
    return ind;
}

} // namespace wann

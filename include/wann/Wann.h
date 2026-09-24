#pragma once
#include "Hyperparams.h"
#include "Ind.h"
#include <vector>

namespace wann {

// Per-generation counters of the variation operators, filled by ask().
// Index order for the arrays: [addConn, addNode, enable, mutAct, toggleExcitatory].
// `chosen` = times the roulette selected the operator; `applied` = times it
// actually changed the genome (a chosen mutation can be a no-op, e.g. addConn
// with no free (src,dst) pair, enable with no disabled connection).
// `mooConn` is 1 if the generation's selection used [meanFit, 1/nConn]
// (prob alg_probMoo), 0 if it used [meanFit, maxFit], -1 at gen 0 (no selection).
struct MutStats {
    int chosen[5]  = {0, 0, 0, 0, 0};
    int applied[5] = {0, 0, 0, 0, 0};
    int mooConn    = -1;
};

// Where one member of the population returned by ask() came from.
// `parent` indexes the population evaluated in the previous generation
// (the one whose reward matrix was given to tell()); -1 at generation 0.
// `parentB` is the less-fit parent when crossover was used, else -1.
// `op` is the operator picked by topoMutate, using the MutStats order
// [addConn, addNode, enable, mutAct, toggleExcitatory]; -1 for an elite
// copied unchanged (and at generation 0). `applied` = the operator really
// changed the genome.
struct ChildInfo {
    int  parent  = -1;
    int  parentB = -1;
    int  op      = -1;
    bool applied = false;
};

// -------------------------------------------------------------------------
// Wann – main evolutionary algorithm (ask / tell interface).
//
// Usage:
//   Wann wann(hyp);
//   for (int gen = 0; gen < hyp.maxGen; ++gen) {
//       auto& pop = wann.ask();        // get current population
//       // evaluate each ind: reward[i][j] = fitness of pop[i] with weight j
//       wann.tell(reward);             // assign fitness
//   }
// -------------------------------------------------------------------------
class Wann {
public:
    explicit Wann(const Hyperparams& hyp);

    // Returns reference to the current population (after ask()).
    std::vector<Ind>& ask();

    // Assigns fitness from a 2-D reward matrix.
    // reward[i][j] = fitness of pop[i] evaluated with the j-th weight value.
    void tell(const std::vector<std::vector<double>>& reward);

    // Direct read access to population and generation counter.
    const std::vector<Ind>& population() const { return pop; }
    int generation() const { return gen; }

    // Operator counters for the population returned by the last ask().
    const MutStats& lastMutStats() const { return mutStats_; }

    // Provenance of each member of the population returned by the last ask()
    // (same indexing as pop). See ChildInfo.
    const std::vector<ChildInfo>& lastLineage() const { return lineage_; }

    // Current population (public for DataGatherer access).
    std::vector<Ind> pop;

private:
    Hyperparams p;
    std::vector<InnovRecord> innov;
    int gen = 0;
    MutStats mutStats_;
    std::vector<ChildInfo> lineage_;

    // Minimal species container (WANN uses a single species).
    struct Species {
        int         seedIdx    = 0;    // index into pop
        std::vector<int> memberIdx;    // indices into pop
        int         nOffspring = 0;
    };
    std::vector<Species> species;

    void initPop();
    void probMoo();
    void speciate();
    void evolvePop();

    // Returns children produced from one species; updates innov.
    std::vector<Ind> recombine(const Species& sp);

    Ind    crossover   (const Ind& parentA, const Ind& parentB);
    // Returns {operator index 0..4 (MutStats order), whether it changed the genome}.
    std::pair<int,bool> topoMutate(Ind& child);
    // The mutation operators return true iff they changed the genome.
    bool   mutAddConn          (std::vector<ConnGene>& conns,
                                const std::vector<NodeGene>& nodes);
    bool   mutAddNode          (std::vector<ConnGene>& conns,
                                std::vector<NodeGene>& nodes);
    bool   mutToggleExcitatory (std::vector<ConnGene>& conns,
                                const std::vector<NodeGene>& nodes);
};

} // namespace wann

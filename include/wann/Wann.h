#pragma once
#include "Hyperparams.h"
#include "Ind.h"
#include <vector>

namespace wann {

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

    // Current population (public for DataGatherer access).
    std::vector<Ind> pop;

private:
    Hyperparams p;
    std::vector<InnovRecord> innov;
    int gen = 0;

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
    void   topoMutate  (Ind& child);
    void   mutAddConn          (std::vector<ConnGene>& conns,
                                const std::vector<NodeGene>& nodes);
    void   mutAddNode          (std::vector<ConnGene>& conns,
                                std::vector<NodeGene>& nodes);
    void   mutToggleExcitatory (std::vector<ConnGene>& conns);
};

} // namespace wann

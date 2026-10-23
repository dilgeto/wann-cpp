#pragma once
#include "Ind.h"
#include "Wann.h"

#include <string>
#include <vector>

namespace wann {

// Whole-population snapshot written by main_car.cpp (hyp.snapshot_interval > 0)
// and read by wann_car_neighborhood. Unlike the *_best.out files, which hold
// only the *expressed* network of the running best, this keeps full genomes:
// node ids, innovation numbers, disabled connections and their polarity,
// which the neighbourhood enumeration needs.
struct PopSnapshot {
    int gen      = 0;
    // The `seed` argument evalPop() received for this generation
    // ((int)seed + gen). Individual i was evaluated with
    // evalSeed * 10000 + i, so keeping it allows a paired re-evaluation.
    int evalSeed = 0;
    std::vector<Ind>                 pop;      // genes + fitness/fitMax/nConn
    std::vector<std::vector<double>> reward;   // [i][weight index]
    std::vector<ChildInfo>           lineage;  // same indexing as pop (may be empty)
};

void        savePopSnapshot(const std::string& path, const PopSnapshot& snap);
PopSnapshot loadPopSnapshot(const std::string& path);

// Rebuild a genome from a saved expressed network (the *_best.out file
// written by exportNet: N x N matrix in [bias+inputs | hidden | outputs]
// order plus an activation column). Approximate, because the file does not
// hold the genome:
//   - "nan" entries become disabled connections; their polarity is lost, so
//     they are assumed excitatory.
//   - innovation numbers are re-assigned, node ids follow matrix position.
//   - Files written before Ind::express restored NaN for disabled
//     hidden->hidden edges (2026-10-23) hold them as +1 (getNodeOrder
//     binarises that block with copysign, and copysign(1, NaN) = +1), so
//     those come back as enabled excitatory connections; such a file is NOT
//     the network training evaluated and cannot be repaired. Files written
//     since then keep them as NaN and round-trip exactly.
Ind genomeFromNetFile(const std::string& outFile, int nInput, int nOutput);

} // namespace wann

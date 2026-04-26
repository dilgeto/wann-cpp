#pragma once
#include "Hyperparams.h"
#include "Ind.h"
#include <string>
#include <vector>

namespace wann {

// -------------------------------------------------------------------------
// DataGatherer – collects per-generation statistics and writes them to disk.
// Mirrors Python wann_src/dataGatherer.py::DataGatherer.
// -------------------------------------------------------------------------
class DataGatherer {
public:
    DataGatherer(const std::string& filePrefix, const Hyperparams& hyp);

    // Append one generation of statistics.
    void gatherData(const std::vector<Ind>& pop);

    // Format a one-line summary for console output.
    std::string display() const;

    // Write accumulated data to disk.
    //   gen = current generation index (used for per-generation best file).
    void save(int gen = -1);

    // ---- read-only accessors (for checkBest logic in main) ----
    bool newBest() const { return newBest_; }
    const Ind& bestInd() const { return best_.back(); }
    void overrideBestFitness(double fit) {
        best_.back().fitness = fit;
        fitTop_.back() = fit;
    }
    void revertBest(int saveMod);

private:
    std::string prefix_;        // "log/<output_prefix>"
    const Hyperparams& p_;

    // Per-generation statistics (one entry per call to gatherData).
    std::vector<double> xScale_;    // cumulative evaluations
    std::vector<double> fitMed_;    // median fitness in population
    std::vector<double> fitMax_;    // best fitness this generation
    std::vector<double> fitTop_;    // best fitness seen so far
    std::vector<double> fitPeak_;   // best fitMax seen so far
    std::vector<double> nodeMed_;   // median node count
    std::vector<double> connMed_;   // median connection count

    // Best individual tracking.
    std::vector<Ind> elite_;    // best per generation
    std::vector<Ind> best_;     // running best (may plateau)
    bool             newBest_ = false;

    // Helper to save a CSV file.
    static void lsave(const std::string& fname,
                      const std::vector<std::vector<double>>& data);
};

} // namespace wann

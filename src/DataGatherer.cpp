#include "wann/DataGatherer.h"
#include "wann/Ind.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace wann {

namespace fs = std::filesystem;

// -------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------
static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const int n = static_cast<int>(v.size());
    std::nth_element(v.begin(), v.begin() + n/2, v.end());
    if (n % 2 == 1) return v[n/2];
    double hi = v[n/2];
    std::nth_element(v.begin(), v.begin() + n/2 - 1, v.end());
    return (v[n/2-1] + hi) / 2.0;
}

// -------------------------------------------------------------------------
// DataGatherer
// -------------------------------------------------------------------------
DataGatherer::DataGatherer(const std::string& filePrefix, const Hyperparams& hyp)
    : prefix_("log/" + filePrefix), p_(hyp)
{
    fs::create_directories("log");
}

void DataGatherer::gatherData(const std::vector<Ind>& pop) {
    // Per-individual scalars.
    std::vector<double> fitness, nodes, conns, peaks;
    fitness.reserve(pop.size());
    nodes.reserve(pop.size());
    conns.reserve(pop.size());
    peaks.reserve(pop.size());

    for (const auto& ind : pop) {
        fitness.push_back(ind.fitness);
        peaks.push_back(ind.fitMax);
        nodes.push_back(static_cast<double>(ind.nNodes));
        conns.push_back(static_cast<double>(ind.nConn));
    }

    // x-scale: cumulative number of evaluations.
    if (xScale_.empty()) xScale_.push_back(static_cast<double>(pop.size()));
    else                  xScale_.push_back(xScale_.back() + static_cast<double>(pop.size()));

    // Elite: best individual this generation.
    int eliteIdx = static_cast<int>(
        std::max_element(fitness.begin(), fitness.end()) - fitness.begin());
    elite_.push_back(pop[eliteIdx]);

    // Running best.
    if (best_.empty()) {
        best_.push_back(elite_.back());
        newBest_ = true;
    } else if (elite_.back().fitness > best_.back().fitness) {
        best_.push_back(elite_.back());
        newBest_ = true;
    } else {
        best_.push_back(best_.back());
        newBest_ = false;
    }

    fitMed_.push_back(median(fitness));
    fitMax_.push_back(elite_.back().fitness);
    fitTop_.push_back(best_.back().fitness);
    fitPeak_.push_back(best_.back().fitMax);
    nodeMed_.push_back(median(nodes));
    connMed_.push_back(median(conns));
}

std::string DataGatherer::display() const {
    if (fitMax_.empty()) return "";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "|---| Elite Fit: " << fitMax_.back()
       << " \t|---| Best Fit:  " << fitTop_.back()
       << " \t|---| Peak Fit:  " << fitPeak_.back();
    return ss.str();
}

void DataGatherer::save(int gen) {
    // --- Generation statistics ---
    {
        const int nRows = static_cast<int>(xScale_.size());
        std::vector<std::vector<double>> stats(nRows, std::vector<double>(7));
        for (int i = 0; i < nRows; ++i) {
            stats[i] = {xScale_[i], fitMed_[i], fitMax_[i],
                        fitTop_[i], fitPeak_[i], nodeMed_[i], connMed_[i]};
        }
        lsave(prefix_ + "_stats.out", stats);
    }

    // --- Best individual network ---
    if (!best_.empty()) {
        const Ind& b = (gen >= 0 && gen < static_cast<int>(best_.size()))
                       ? best_[gen] : best_.back();
        if (!b.wMat.empty())
            exportNet(prefix_ + "_best.out", b.wMat, b.nNodes, b.aVec);

        // Per-generation snapshot.
        if (gen > 1) {
            std::string folder = prefix_ + "_best/";
            fs::create_directories(folder);
            std::ostringstream fname;
            fname << folder << std::setw(4) << std::setfill('0') << gen << ".out";
            if (!b.wMat.empty())
                exportNet(fname.str(), b.wMat, b.nNodes, b.aVec);
        }
    }
}

void DataGatherer::revertBest(int saveMod) {
    // When the new best turns out to be unlucky, roll back the last saveMod
    // entries of best_ and fit_top_ to the previous best value.
    if (best_.size() < 2) return;
    const Ind&   prev    = best_[best_.size() - saveMod - 1];
    const double prevFit = fitTop_[fitTop_.size() - saveMod - 1];
    for (int i = 0; i < saveMod && !best_.empty(); ++i) {
        best_.back()   = prev;
        fitTop_.back() = prevFit;
        // Don't actually erase — just overwrite in-place.
        if (static_cast<int>(best_.size()) > saveMod)
            break;
    }
    newBest_ = false;
}

void DataGatherer::lsave(const std::string& fname,
                          const std::vector<std::vector<double>>& data)
{
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot write: " + fname);
    f << std::scientific << std::setprecision(2);
    for (const auto& row : data) {
        for (int i = 0; i < static_cast<int>(row.size()); ++i) {
            if (i > 0) f << ',';
            f << row[i];
        }
        f << '\n';
    }
}

} // namespace wann

#include "../include/wann/SnnBipedalTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>

#include <rl_tools/operations/cpu.h>
// Box2D-free BipedalWalker port, header-only, from snn-simulator/include/.
#include <rl/environments/bipedal_walker/operations_generic.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace rlt  = rl_tools;
namespace phys = ::bipedal_walker::physics;

// The physics port was validated against Box2D in float32 (snn-simulator's
// bipedal tests all use float), so the environment runs in float; the SNN
// side stays in double.
using EnvT   = float;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG    = typename rlt::devices::random::CPU::ENGINE<>;

using EnvSpec     = rlt::rl::environments::bipedal_walker::Specification<EnvT, TI>;
using NormalEnv   = rlt::rl::environments::BipedalWalker<EnvSpec>;
using HardcoreEnv = rlt::rl::environments::BipedalWalkerHardcore<EnvSpec>;

using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<EnvT, TI, 1, 24, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<EnvT, TI, 1, 4, false>>;

static_assert(NormalEnv::Observation::DIM == wann::SnnBipedalTask::N_OBS, "BipedalWalker observation must be 24-dim");
static_assert(NormalEnv::ACTION_DIM == wann::SnnBipedalTask::N_ACTIONS, "BipedalWalker action must be 4-dim");

namespace {
std::unique_ptr<Encoder> makeEncoder(wann::SnnEncoder type, uint32_t seed, double ttfsThreshold) {
    constexpr double MAX_RATE   = 100.0;
    constexpr double REF_PERIOD = 2.0;
    switch (type) {
        case wann::SnnEncoder::POISSON: {
            auto e = std::make_unique<PoissonEncoder>(MAX_RATE, true, REF_PERIOD);
            e->reseed(seed);
            return e;
        }
        case wann::SnnEncoder::RATE: {
            auto e = std::make_unique<RateEncoder>(MAX_RATE);
            e->reseed(seed);
            return e;
        }
        case wann::SnnEncoder::TTFS:
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LINEAR, ttfsThreshold);
        case wann::SnnEncoder::TTFS_LOG:
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LOGARITHMIC, ttfsThreshold);
        default:
            return nullptr;  // CURRENT / SMALL / LARGE inject currents, not spike trains
    }
}

// Normalisation range per observation channel. The env scales most values
// but gives no hard bounds (Gymnasium's observation space is ±inf):
//   hull angle / angular velocity / vx / vy: ±1 covers normal walking; a hull
//     tilted past ~1 rad is about to touch the ground (episode over).
//   joint angles: the joint limits (hip [-0.8, 1.1], knee+1 [-0.6, 0.9])
//     plus a small margin for soft-constraint overshoot.
//   joint speeds (already divided by the motor's nominal speed): ±1.5, so
//     impact spikes past the nominal speed don't saturate immediately.
//   contacts {0,1}, lidar [0,1]: already normalised.
// Values outside the range are clamped after normalising.
constexpr std::array<std::pair<double, double>, 24> OBS_LIMITS = {{
    {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0},       // hull
    {-0.9, 1.2}, {-1.5, 1.5}, {-0.7, 1.0}, {-1.5, 1.5}, {0.0, 1.0},  // leg 1
    {-0.9, 1.2}, {-1.5, 1.5}, {-0.7, 1.0}, {-1.5, 1.5}, {0.0, 1.0},  // leg 2
    {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},      // lidar
    {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},
}};

double normalise(int i, double v) {
    const auto [lo, hi] = OBS_LIMITS[i];
    return std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

// Classical population vector (same as SnnCarTask): neuron k of a
// `count`-sized population has a fixed preferred value uniformly spaced in
// [-1,1]; the action is the spike-count-weighted average. No spikes → 0.0.
double populationVectorDecode(const std::vector<std::vector<double>>& outSpikes, int start, int count)
{
    double weightedSum = 0.0, totalCount = 0.0;
    for (int k = 0; k < count; ++k) {
        double pref = (count > 1) ? (-1.0 + 2.0 * k / (count - 1)) : 0.0;
        double cnt  = static_cast<double>(outSpikes[start + k].size());
        weightedSum += cnt * pref;
        totalCount  += cnt;
    }
    return (totalCount > 0.0) ? (weightedSum / totalCount) : 0.0;
}

// Decode output spike trains into the 4 torques ∈ [-1, 1].
// FIRST_SPIKE: latency → [0,1] → [-1,1]; a silent output gives 0.0 (no
//   torque), not the -1 that the latency formula would give, since "never
//   spiked" has no latency and full reverse torque is not a neutral action
//   (same reasoning as SnnL2FTask).
// SPIKE_COUNT / default (rate-style): n / max_spikes → [-1,1]; 0 spikes is
//   the bottom of a continuous scale here, so it stays -1.
// POPULATION_VECTOR: outSpikes[a*P .. a*P+P-1] = population of action a.
std::array<double, 4> decodeActions(const std::vector<std::vector<double>>& outSpikes,
                                    wann::SnnDecoder decoder,
                                    const RLDecoder& rl_decoder,
                                    double max_spikes,
                                    int neuronsPerVar)
{
    std::array<double, 4> a{};
    for (int k = 0; k < 4; ++k) {
        double v;
        switch (decoder) {
            case wann::SnnDecoder::FIRST_SPIKE:
                v = outSpikes[k].empty()
                    ? 0.0
                    : rl_decoder.decodeContinuousAction(outSpikes[k]) * 2.0 - 1.0;
                break;
            case wann::SnnDecoder::POPULATION_VECTOR:
                v = populationVectorDecode(outSpikes, k * neuronsPerVar, neuronsPerVar);
                break;
            default:
                v = std::min(1.0, static_cast<double>(outSpikes[k].size()) / max_spikes) * 2.0 - 1.0;
                break;
        }
        a[k] = std::clamp(v, -1.0, 1.0);
    }
    return a;
}
} // anonymous namespace

namespace wann {

const double SnnBipedalTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 2.0, 3.0, 5.0, 8.0
};

SnnBipedalTask::SnnBipedalTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
    // Rounded for the same reason as SnnCarTask: a fractional window makes
    // TTFS spike times fall off the integer dt=1ms grid and get dropped.
    , windowMs_(std::round(hyp.snn_window_ms))
    , tauExc_(hyp.snn_tau_exc)
    , tauInh_(hyp.snn_tau_inh)
    , ttfsThreshold_(hyp.snn_ttfs_threshold)
    , hardcore_(hyp.bipedal_hardcore)
    , maxSteps_(hyp.bipedal_max_steps)
{
    // decodeActions indexes outSpikes by action unconditionally; catch a
    // mismatched geometry here instead of an out-of-bounds access later.
    const int expectedOut = (decoder_ == SnnDecoder::POPULATION_VECTOR)
                            ? N_ACTIONS * neuronsPerVar_ : N_ACTIONS;
    if (nOutput_ != expectedOut)
        throw std::runtime_error(
            "SnnBipedalTask: snn_decoder=" + hyp.snn_decoder + " requires ann_nOutput=" +
            std::to_string(expectedOut) + " (4 torques" +
            (decoder_ == SnnDecoder::POPULATION_VECTOR ? " x snn_neurons_per_var" : "") +
            "), got " + std::to_string(nOutput_));

    const int expectedIn = (encoder_ == SnnEncoder::SMALL) ? N_OBS * 2
                         : (encoder_ == SnnEncoder::LARGE) ? N_OBS * neuronsPerVar_
                         : N_OBS;
    if (nInput_ != expectedIn)
        throw std::runtime_error(
            "SnnBipedalTask: snn_encoder=" + hyp.snn_encoder + " requires ann_nInput=" +
            std::to_string(expectedIn) + ", got " + std::to_string(nInput_));

    if (windowMs_ < 1.0)
        throw std::runtime_error("SnnBipedalTask: snn_window_ms must be >= 1");
}

NeuronType SnnBipedalTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;

        // Izhikevich (2004) Fig. 1 — same mapping as SnnCarTask (15/17/20
        // duplicate 7/13/18).
        case 7:  return NeuronType::TONIC_SPIKING;
        case 8:  return NeuronType::PHASIC_SPIKING;
        case 9:  return NeuronType::TONIC_BURSTING;
        case 10: return NeuronType::PHASIC_BURSTING;
        case 11: return NeuronType::MIXED_MODE;
        case 12: return NeuronType::SPIKE_FREQUENCY_ADAPTATION;
        case 13: return NeuronType::CLASS1_EXCITABLE;
        case 14: return NeuronType::CLASS2_EXCITABLE;
        case 15: return NeuronType::SPIKE_LATENCY;
        case 16: return NeuronType::SUBTHRESHOLD_OSCILLATIONS;
        case 17: return NeuronType::INTEGRATOR;
        case 18: return NeuronType::REBOUND_SPIKE;
        case 19: return NeuronType::REBOUND_BURST;
        case 20: return NeuronType::THRESHOLD_VARIABILITY;
        case 21: return NeuronType::BISTABILITY;
        case 22: return NeuronType::DEPOLARIZING_AFTERPOTENTIAL;
        case 23: return NeuronType::ACCOMMODATION;
        case 24: return NeuronType::INHIBITION_INDUCED_SPIKING;
        case 25: return NeuronType::INHIBITION_INDUCED_BURSTING;

        default: return NeuronType::REGULAR_SPIKING;
    }
}

Network SnnBipedalTask::buildNetwork(const Ind& ind) const
{
    Network net(1.0, true);
    std::unordered_map<int, int> snn_id;
    snn_id.reserve(ind.nodes.size());

    for (const auto& ng : ind.nodes) {
        NeuronType nt = wannActToNeuronType(ng.activation);
        int sid;
        switch (ng.type) {
            case 4: sid = net.addInputNeuron(nt);  break;
            case 1: sid = net.addInputNeuron(nt);  break;
            case 3: sid = net.addHiddenNeuron(nt); break;
            case 2: sid = net.addOutputNeuron(nt); break;
            default: continue;
        }
        snn_id[ng.id] = sid;
    }

    for (const auto& cg : ind.conns) {
        if (!cg.enabled) continue;
        auto it_src = snn_id.find(cg.src);
        auto it_dst = snn_id.find(cg.dst);
        if (it_src == snn_id.end() || it_dst == snn_id.end()) continue;
        net.addSynapse(it_src->second, it_dst->second, cg.excitatory);
    }

    net.setTimeConstants(tauExc_, tauInh_);
    return net;
}

Network SnnBipedalTask::buildNetwork(const std::vector<double>& wVec,
                                     const std::vector<int>&    aVec) const
{
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    Network net(1.0, true);
    std::vector<int> snn_id(N, -1);

    snn_id[0] = net.addInputNeuron(wannActToNeuronType(aVec[0]));
    for (int i = 1; i <= nInput_; ++i)
        snn_id[i] = net.addInputNeuron(wannActToNeuronType(aVec[i]));
    for (int i = nInput_ + 1; i < N - nOutput_; ++i)
        snn_id[i] = net.addHiddenNeuron(wannActToNeuronType(aVec[i]));
    for (int i = N - nOutput_; i < N; ++i)
        snn_id[i] = net.addOutputNeuron(wannActToNeuronType(aVec[i]));

    for (int i = 0; i < N; ++i) {
        if (snn_id[i] < 0) continue;
        for (int j = 0; j < N; ++j) {
            if (snn_id[j] < 0 || i == j) continue;
            if (wVec[i * N + j] != 0.0)
                net.addSynapse(snn_id[i], snn_id[j], wVec[i * N + j] > 0.0);
        }
    }

    net.setTimeConstants(tauExc_, tauInh_);
    return net;
}

std::pair<double,double> SnnBipedalTask::runEpisode(Network& net, double sharedWeight,
                                                    long long episodeSeed,
                                                    std::ostream* csv) const
{
    return hardcore_
        ? runEpisodeImpl<HardcoreEnv>(net, sharedWeight, episodeSeed, csv)
        : runEpisodeImpl<NormalEnv>(net, sharedWeight, episodeSeed, csv);
}

template <typename ENV>
std::pair<double,double> SnnBipedalTask::runEpisodeImpl(Network& net, double sharedWeight,
                                                        long long episodeSeed,
                                                        std::ostream* csv) const
{
    net.fastReset();
    DEVICE device;
    ENV env;
    typename ENV::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));

    // Terrain (normal or hardcore, picked by overload on ENV) and the
    // initial push on the hull are both drawn from the episode seed.
    rlt::sample_initial_parameters(device, env, params, rng);

    typename ENV::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    typename ENV::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int stepLimit = (maxSteps_ > 0)
        ? std::min<int>(maxSteps_, static_cast<int>(ENV::EPISODE_STEP_LIMIT))
        : static_cast<int>(ENV::EPISODE_STEP_LIMIT);

    const int window_steps = static_cast<int>(windowMs_);
    const int n_channels   = nInput_ + 1;  // bias + observations

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu, ttfsThreshold_);

    const RLDecoder::DecodingType dec_type =
        (decoder_ == SnnDecoder::FIRST_SPIKE) ? RLDecoder::DecodingType::FIRST_SPIKE :
        (decoder_ == SnnDecoder::SPIKE_COUNT) ? RLDecoder::DecodingType::SPIKE_COUNT :
                                                RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, windowMs_);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    std::unique_ptr<RLEncoder> rl_enc;
    if (encoder_ == SnnEncoder::SMALL || encoder_ == SnnEncoder::LARGE)
        rl_enc = std::make_unique<RLEncoder>(encoder_ == SnnEncoder::SMALL
                                             ? RLEncoder::EncodingType::SMALL
                                             : RLEncoder::EncodingType::LARGE,
                                             100.0, 5, static_cast<size_t>(neuronsPerVar_));
    const std::vector<std::pair<double,double>> limits(OBS_LIMITS.begin(), OBS_LIMITS.end());

    std::array<double, N_OBS> obs{};
    std::vector<double> norm(n_channels, 0.0);
    std::vector<double> currents(n_channels, 0.0);
    std::vector<std::vector<double>> spike_trains(n_channels);
    std::vector<std::vector<double>> outSpikes(nOutput_);

    double total_reward = 0.0;

    for (int step = 0; step < stepLimit; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        for (int i = 0; i < N_OBS; ++i) obs[i] = static_cast<double>(rlt::get(obs_mat, 0, i));

        if (resetBetweenSteps_) net.fastReset();
        for (auto& s : outSpikes) s.clear();

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            norm[0] = 1.0;
            for (int i = 0; i < N_OBS; ++i) norm[i + 1] = normalise(i, obs[i]);
            for (int ch = 0; ch < n_channels; ++ch)
                spike_trains[ch] = enc->encode(norm[ch], windowMs_, DT);

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                for (int oi = 0; oi < nOutput_ && oi < static_cast<int>(out.size()); ++oi)
                    if (out[oi]) outSpikes[oi].push_back(static_cast<double>(t));
            }
        } else {
            currents[0] = BIAS_CURRENT;
            if (rl_enc) {
                const std::vector<double> obs_vals(obs.begin(), obs.end());
                const std::vector<double> enc_currents = (encoder_ == SnnEncoder::SMALL)
                    ? rl_enc->encodeObservationSmall(obs_vals)
                    : rl_enc->encodeObservationLarge(obs_vals, limits);
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: normalised injection in [0, 20] mA
                for (int i = 0; i < N_OBS; ++i) currents[i + 1] = normalise(i, obs[i]) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                for (int oi = 0; oi < nOutput_ && oi < static_cast<int>(out.size()); ++oi)
                    if (out[oi]) outSpikes[oi].push_back(static_cast<double>(t));
            }
        }

        const auto action = decodeActions(outSpikes, decoder_, rl_decoder, max_spikes, neuronsPerVar_);
        for (int k = 0; k < N_ACTIONS; ++k)
            rlt::set(action_mat, 0, k, static_cast<EnvT>(action[k]));

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double reward = static_cast<double>(
            rlt::reward(device, env, params, state, action_mat, next_state, rng));
        total_reward += reward;

        if (csv) {
            *csv << step << ','
                 << phys::body_origin(params.geometry, state.bodies, 0).x;
            for (int i = 0; i < N_OBS; ++i) *csv << ',' << obs[i];
            for (int k = 0; k < N_ACTIONS; ++k) *csv << ',' << action[k];
            *csv << ',' << reward << '\n';
        }

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return {total_reward, total_reward};
}

double SnnBipedalTask::evaluateWeight(const Network& templateNet, int wi, int seed) const
{
    Network net = templateNet;
    double total = 0.0;
    for (int rep = 0; rep < nReps_; ++rep) {
        long long episodeSeed = static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
        total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed).first;
    }
    return total / static_cast<double>(nReps_);
}

std::vector<double> SnnBipedalTask::evaluate(const Ind& ind, int seed)
{
    Network templateNet = buildNetwork(ind);
    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi)
        rewards[wi] = evaluateWeight(templateNet, wi, seed);
    return rewards;
}

std::vector<double> SnnBipedalTask::getDistFitness(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        int seed)
{
    Network templateNet = buildNetwork(wVec, aVec);
    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi)
        rewards[wi] = evaluateWeight(templateNet, wi, seed);
    return rewards;
}

std::pair<std::vector<double>,std::vector<double>> SnnBipedalTask::evalEpisodes(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        double weight, int nEpisodes, int baseSeed) const
{
    Network net = buildNetwork(wVec, aVec);
    std::vector<double> shaped(nEpisodes), original(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i) {
        auto [s, o] = runEpisode(net, weight, baseSeed + i);
        shaped[i]   = s;
        original[i] = o;
    }
    return {shaped, original};
}

void SnnBipedalTask::exportTrajectory(const std::vector<double>& wVec,
                                      const std::vector<int>&    aVec,
                                      int bestWi, int evalSeed,
                                      const std::string& outFile,
                                      bool directSeed) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,hull_x";
    for (int i = 0; i < N_OBS; ++i) csv << ",obs" << i;
    for (int k = 0; k < N_ACTIONS; ++k) csv << ",a" << k;
    csv << ",reward\n";

    Network net = buildNetwork(wVec, aVec);
    const long long episodeSeed = directSeed
        ? static_cast<long long>(evalSeed)
        : static_cast<long long>(evalSeed) * 10000 + bestWi * 100 + 0;

    runEpisode(net, WEIGHT_VALS[bestWi], episodeSeed, &csv);
    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

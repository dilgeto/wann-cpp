#include "../include/wann/SnnBottleneckTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace rlt = rl_tools;

using T          = double;
using TI         = size_t;
using DEVICE     = rlt::devices::DefaultCPU;
using RNG        = typename rlt::devices::random::CPU::ENGINE<>;
using TypePolicy = rlt::numeric_types::Policy<T>;

// Same Parameters as rl-tools' own benchmark for this environment
// (rl::zoo::bottleneck_v0::ENVIRONMENT_FACTORY::ENVIRONMENT_PARAMETERS in
// src/rl/zoo/bottleneck-v0/environment.h) — 5 agents, a wider FOV/gap and a
// faster/more agile agent than bottleneck::DefaultParameters. Fields not
// listed here (LIDAR_RESOLUTION, arena size, EPISODE_STEP_LIMIT, ...)
// inherit DefaultParameters unchanged.
template <typename TT, typename TTI>
struct BenchmarkParameters : rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<TT, TTI> {
    static constexpr TTI N_AGENTS                     = 5;
    static constexpr TT   LIDAR_FOV                   = rlt::math::PI<TT> * 90 / 180;
    static constexpr TT   BOTTLENECK_WIDTH             = 5;
    static constexpr TT   AGENT_MAX_SPEED              = 4;
    static constexpr TT   AGENT_MAX_ANGULAR_VELOCITY   = 4;
    static constexpr TT   AGENT_MAX_ACCELERATION       = 20;
    static constexpr TT   AGENT_MAX_ANGULAR_ACCELERATION = 20;
};

using EnvSpec = rlt::rl::environments::multi_agent::bottleneck::Specification<
    TypePolicy, TI, BenchmarkParameters<T, TI>>;
using Env     = rlt::rl::environments::multi_agent::Bottleneck<EnvSpec>;

using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::Observation::DIM, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::ACTION_DIM, false>>;

static_assert(Env::PARAMETERS::N_AGENTS == wann::SnnBottleneckTask::N_AGENTS,
              "BenchmarkParameters::N_AGENTS must match SnnBottleneckTask::N_AGENTS");
static_assert(Env::Observation::DIM ==
                  wann::SnnBottleneckTask::N_AGENTS * wann::SnnBottleneckTask::PER_AGENT_OBS_DIM,
              "Bottleneck per-agent observation must be 12-dim (7 + LIDAR_RESOLUTION)");
static_assert(Env::ACTION_DIM ==
                  wann::SnnBottleneckTask::N_AGENTS * wann::SnnBottleneckTask::PER_AGENT_ACTION_DIM,
              "Bottleneck per-agent action must be 2-dim");

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

// Normalisation range per per-agent observation channel (BenchmarkParameters:
// ARENA_WIDTH=ARENA_HEIGHT=20 (unchanged from DefaultParameters),
// AGENT_MAX_SPEED=4, AGENT_MAX_ANGULAR_VELOCITY=4).
// Order matches observe() in bottleneck/operations_generic.h.
//
// Lidar: update_lidar() in operations_generic.h declares LIDAR_RANGE but
// never actually uses it to clamp the ray or the returned intersection — the
// arena's bounding rectangle always intersects, so the real upper bound on a
// reported distance is the arena's diagonal (sqrt(20^2+20^2) ≈ 28.28), not
// the nominal LIDAR_RANGE=3. Confirmed empirically (readings up to ~8.5 seen
// in a smoke test) before fixing this, since trusting the unused constant
// would have silently saturated most lidar readings to 1.0.
constexpr double LIDAR_MAX_DIST = 28.284271247461902;  // sqrt(20^2 + 20^2)
constexpr std::array<std::pair<double, double>, 12> OBS_LIMITS = {{
    {0.0, 20.0}, {0.0, 20.0},                 // position.x, position.y
    {-3.14159265358979323846, 3.14159265358979323846},  // orientation
    {-4.0, 4.0}, {-4.0, 4.0},                 // velocity.x, velocity.y
    {-4.0, 4.0},                              // angular_velocity
    {-1.0, 1.0},                              // alive flag
    {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST},
    {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST},  // lidar x5
}};

double normalise(int i, double v) {
    const auto [lo, hi] = OBS_LIMITS[i];
    return std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

// Potential-based shaping term: phi(s) = (sum of agents' x-position) /
// ARENA_WIDTH. Sum, not mean, so one agent's progress isn't diluted by
// N_AGENTS — a single agent crossing at full speed (dx ≈ AGENT_MAX_SPEED*DT)
// gives a per-step delta on the same order as the environment's own
// per-step reward (~0.1/agent while on the right side), instead of a
// negligible fraction of it. Dead agents freeze in place rather than
// teleporting (rl-tools' step() leaves a dead agent's state untouched), so
// phi is continuous across a death transition — it does not double-count
// the environment's own one-time -1 death penalty. Being a pure function of
// state (not of the action taken), this is a valid potential-based shaping
// term (Ng, Harada & Russell 1999): it cannot change which policy is
// ultimately optimal, only how easy it is for evolution to find it.
double phi(const Env::State& s) {
    double sum = 0.0;
    for (int i = 0; i < wann::SnnBottleneckTask::N_AGENTS; ++i)
        sum += s.agent_states[i].position[0];
    return sum / Env::PARAMETERS::ARENA_WIDTH;
}

// Classical population vector (same as SnnCarTask/SnnBipedalTask): neuron k
// of a `count`-sized population has a fixed preferred value uniformly spaced
// in [-1,1]; the action is the spike-count-weighted average. No spikes → 0.0.
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

// Decode one agent's output spike trains into its 2 actions ∈ [-1, 1]
// (linear acceleration, angular acceleration). Same semantics as
// SnnBipedalTask::decodeActions, sized for PER_AGENT_ACTION_DIM instead of 4.
std::array<double, wann::SnnBottleneckTask::PER_AGENT_ACTION_DIM> decodeActions(
        const std::vector<std::vector<double>>& outSpikes,
        wann::SnnDecoder decoder,
        const RLDecoder& rl_decoder,
        double max_spikes,
        int neuronsPerVar)
{
    constexpr int NA = wann::SnnBottleneckTask::PER_AGENT_ACTION_DIM;
    std::array<double, NA> a{};
    for (int k = 0; k < NA; ++k) {
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

// NOT the {0.5..8} range the other tasks (incl. SnnBipedalTask, where this
// was copied from) use — recalibrated against this task's actual fan-in.
// Measured directly against the SNN simulator (tau_exc=5, tau_inh=10,
// window=20ms, matching bottleneck_snn.json): 5 converging excitatory
// synapses firing within the window barely cross threshold even at
// weight=8 (3-4/20 steps in the best case); weights in the 60-130 range are
// needed for an output to fire reliably. This span covers "mostly silent"
// (5) through "reliably firing" (160) so evolution can actually find a
// weight where moving is possible, instead of every value sitting in the
// silent regime where "do nothing" wins by default (see class doc comment).
const double SnnBottleneckTask::WEIGHT_VALS[N_WEIGHTS] = {
    5.0, 15.0, 30.0, 60.0, 100.0, 160.0
};

SnnBottleneckTask::SnnBottleneckTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
    // Rounded for the same reason as SnnCarTask/SnnBipedalTask: a fractional
    // window makes TTFS spike times fall off the integer dt=1ms grid and get
    // dropped.
    , windowMs_(std::round(hyp.snn_window_ms))
    , tauExc_(hyp.snn_tau_exc)
    , tauInh_(hyp.snn_tau_inh)
    , ttfsThreshold_(hyp.snn_ttfs_threshold)
    , shapingScale_(hyp.reward_shaping_scale)
{
    const int expectedOut = (decoder_ == SnnDecoder::POPULATION_VECTOR)
                            ? PER_AGENT_ACTION_DIM * neuronsPerVar_ : PER_AGENT_ACTION_DIM;
    if (nOutput_ != expectedOut)
        throw std::runtime_error(
            "SnnBottleneckTask: snn_decoder=" + hyp.snn_decoder + " requires ann_nOutput=" +
            std::to_string(expectedOut) + " (" + std::to_string(PER_AGENT_ACTION_DIM) +
            " actions per agent" +
            (decoder_ == SnnDecoder::POPULATION_VECTOR ? " x snn_neurons_per_var" : "") +
            "), got " + std::to_string(nOutput_));

    const int expectedIn = (encoder_ == SnnEncoder::SMALL) ? PER_AGENT_OBS_DIM * 2
                         : (encoder_ == SnnEncoder::LARGE) ? PER_AGENT_OBS_DIM * neuronsPerVar_
                         : PER_AGENT_OBS_DIM;
    if (nInput_ != expectedIn)
        throw std::runtime_error(
            "SnnBottleneckTask: snn_encoder=" + hyp.snn_encoder + " requires ann_nInput=" +
            std::to_string(expectedIn) + ", got " + std::to_string(nInput_));

    if (windowMs_ < 1.0)
        throw std::runtime_error("SnnBottleneckTask: snn_window_ms must be >= 1");
}

NeuronType SnnBottleneckTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;

        // Izhikevich (2004) Fig. 1 — same mapping as SnnCarTask/SnnBipedalTask
        // (15/17/20 duplicate 7/13/18).
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

Network SnnBottleneckTask::buildNetwork(const Ind& ind) const
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

Network SnnBottleneckTask::buildNetwork(const std::vector<double>& wVec,
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

std::pair<double,double> SnnBottleneckTask::runEpisode(const Network& templateNet, double sharedWeight,
                                                        long long episodeSeed,
                                                        std::ostream* csv) const
{
    // One independent copy of the shared topology/weights per agent
    // (parameter sharing, decentralized execution) so each agent's membrane
    // state evolves on its own without agents contaminating each other
    // within the same env step.
    std::vector<Network> nets(N_AGENTS, templateNet);
    for (auto& n : nets) n.fastReset();

    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    typename Env::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int stepLimit   = static_cast<int>(Env::EPISODE_STEP_LIMIT);
    const int window_steps = static_cast<int>(windowMs_);
    const int n_channels   = nInput_ + 1;  // bias + per-agent observations

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

    std::array<double, PER_AGENT_OBS_DIM> obsLocal{};
    std::vector<double> norm(n_channels, 0.0);
    std::vector<double> currents(n_channels, 0.0);
    std::vector<std::vector<double>> spike_trains(n_channels);
    std::vector<std::vector<double>> outSpikes(nOutput_);

    double total_shaped   = 0.0;
    double total_original = 0.0;

    for (int step = 0; step < stepLimit; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (csv) *csv << step;

        for (int agent_i = 0; agent_i < N_AGENTS; ++agent_i) {
            const int base = agent_i * PER_AGENT_OBS_DIM;
            for (int i = 0; i < PER_AGENT_OBS_DIM; ++i)
                obsLocal[i] = static_cast<double>(rlt::get(obs_mat, 0, base + i));

            if (resetBetweenSteps_) nets[agent_i].fastReset();
            for (auto& s : outSpikes) s.clear();

            if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
                norm[0] = 1.0;
                for (int i = 0; i < PER_AGENT_OBS_DIM; ++i) norm[i + 1] = normalise(i, obsLocal[i]);
                for (int ch = 0; ch < n_channels; ++ch)
                    spike_trains[ch] = enc->encode(norm[ch], windowMs_, DT);

                for (int t = 0; t < window_steps; ++t) {
                    nets[agent_i].applyInputSpikes(spike_trains, nets[agent_i].getCurrentTime());
                    nets[agent_i].step(sharedWeight);
                    const auto& out = nets[agent_i].getOutputSpikes();
                    for (int oi = 0; oi < nOutput_ && oi < static_cast<int>(out.size()); ++oi)
                        if (out[oi]) outSpikes[oi].push_back(static_cast<double>(t));
                }
            } else {
                currents[0] = BIAS_CURRENT;
                if (rl_enc) {
                    const std::vector<double> obs_vals(obsLocal.begin(), obsLocal.end());
                    const std::vector<double> enc_currents = (encoder_ == SnnEncoder::SMALL)
                        ? rl_enc->encodeObservationSmall(obs_vals)
                        : rl_enc->encodeObservationLarge(obs_vals, limits);
                    for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                        currents[i + 1] = enc_currents[i];
                } else {
                    // CURRENT: normalised injection in [0, 20] mA
                    for (int i = 0; i < PER_AGENT_OBS_DIM; ++i) currents[i + 1] = normalise(i, obsLocal[i]) * 20.0;
                }

                for (int t = 0; t < window_steps; ++t) {
                    nets[agent_i].setInputCurrents(currents);
                    nets[agent_i].step(sharedWeight);
                    const auto& out = nets[agent_i].getOutputSpikes();
                    for (int oi = 0; oi < nOutput_ && oi < static_cast<int>(out.size()); ++oi)
                        if (out[oi]) outSpikes[oi].push_back(static_cast<double>(t));
                }
            }

            const auto action = decodeActions(outSpikes, decoder_, rl_decoder, max_spikes, neuronsPerVar_);
            for (int k = 0; k < PER_AGENT_ACTION_DIM; ++k)
                rlt::set(action_mat, 0, agent_i * PER_AGENT_ACTION_DIM + k, static_cast<T>(action[k]));

            if (csv) {
                for (int i = 0; i < PER_AGENT_OBS_DIM; ++i) *csv << ',' << obsLocal[i];
                for (int k = 0; k < PER_AGENT_ACTION_DIM; ++k) *csv << ',' << action[k];
            }
        }

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double reward = static_cast<double>(
            rlt::reward(device, env, params, state, action_mat, next_state, rng));
        total_original += reward;
        total_shaped   += reward;
        if (shapingScale_ != 0.0)
            total_shaped += shapingScale_ * (phi(next_state) - phi(state));

        if (csv) *csv << ',' << reward << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return {total_shaped, total_original};
}

double SnnBottleneckTask::evaluateWeight(const Network& templateNet, int wi, int seed) const
{
    double total = 0.0;
    for (int rep = 0; rep < nReps_; ++rep) {
        long long episodeSeed = static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
        total += runEpisode(templateNet, WEIGHT_VALS[wi], episodeSeed).first;
    }
    return total / static_cast<double>(nReps_);
}

std::vector<double> SnnBottleneckTask::evaluate(const Ind& ind, int seed)
{
    Network templateNet = buildNetwork(ind);
    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi)
        rewards[wi] = evaluateWeight(templateNet, wi, seed);
    return rewards;
}

std::vector<double> SnnBottleneckTask::getDistFitness(
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

std::pair<std::vector<double>,std::vector<double>> SnnBottleneckTask::evalEpisodes(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        double weight, int nEpisodes, int baseSeed) const
{
    Network templateNet = buildNetwork(wVec, aVec);
    std::vector<double> shaped(nEpisodes), original(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i) {
        auto [s, o] = runEpisode(templateNet, weight, baseSeed + i);
        shaped[i]   = s;
        original[i] = o;
    }
    return {shaped, original};
}

void SnnBottleneckTask::exportTrajectory(const std::vector<double>& wVec,
                                         const std::vector<int>&    aVec,
                                         int bestWi, int evalSeed,
                                         const std::string& outFile,
                                         bool directSeed) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step";
    for (int agent_i = 0; agent_i < N_AGENTS; ++agent_i) {
        for (int i = 0; i < PER_AGENT_OBS_DIM; ++i) csv << ",a" << agent_i << "_obs" << i;
        for (int k = 0; k < PER_AGENT_ACTION_DIM; ++k) csv << ",a" << agent_i << "_act" << k;
    }
    csv << ",reward\n";

    Network templateNet = buildNetwork(wVec, aVec);
    const long long episodeSeed = directSeed
        ? static_cast<long long>(evalSeed)
        : static_cast<long long>(evalSeed) * 10000 + bestWi * 100 + 0;

    runEpisode(templateNet, WEIGHT_VALS[bestWi], episodeSeed, &csv);
    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

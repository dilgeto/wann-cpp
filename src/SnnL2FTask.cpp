#include "../include/wann/SnnL2FTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>
#include <encoding/rateEncoder.hpp>
#include <encoding/poissonEncoder.hpp>
#include <encoding/ttfsEncoder.hpp>
#include <encoding/rlEncoder.hpp>
#include <memory>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace rlt = rl_tools;
namespace l2f = rl_tools::rl::environments::l2f;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;

// Crazyflie dynamics + Lissajous trajectory tracking, 5s @ 100Hz = 500 steps,
// domain randomization disabled (all rl-tools defaults).
using PARAMETER_FACTORY = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;

struct L2FStaticParams {
    static constexpr TI N_SUBSTEPS        = 1;
    static constexpr TI EPISODE_STEP_LIMIT = PARAMETER_FACTORY::EPISODE_STEP_LIMIT_OUTER;  // 500

    // Minimal state: rigid body + rotor RPM (Crazyflie motor lag) + trajectory phase.
    using STATE_TYPE = l2f::StateTrajectory<l2f::StateSpecification<T, TI,
                        l2f::StateRotors<l2f::StateRotorsSpecification<T, TI, /*CLOSED_FORM=*/false,
                        l2f::StateBase<l2f::StateSpecification<T, TI>>>>>>;

    // 13-float observation: position error(3) + quaternion(4) + linear
    // velocity error(3) + angular velocity(3), relative to the Lissajous target.
    using OBSERVATION_TYPE =
        l2f::observation::TrajectoryTrackingPosition<l2f::observation::TrajectoryTrackingPositionSpecification<T, TI,
        l2f::observation::OrientationQuaternion<l2f::observation::OrientationQuaternionSpecification<T, TI,
        l2f::observation::TrajectoryTrackingLinearVelocity<l2f::observation::TrajectoryTrackingLinearVelocitySpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>
        >>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;

    using PARAMETERS = typename PARAMETER_FACTORY::PARAMETERS_TYPE;
    static constexpr PARAMETERS PARAMETER_VALUES = PARAMETER_FACTORY::nominal_parameters;

    static constexpr T STATE_LIMIT_POSITION        = 100000;
    static constexpr T STATE_LIMIT_VELOCITY        = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

using EnvSpec = l2f::Specification<T, TI, L2FStaticParams>;
using Env     = rlt::rl::environments::Multirotor<EnvSpec>;
using RNG     = typename rlt::devices::random::CPU::ENGINE<>;

static_assert(Env::OBSERVATION_DIM == 13, "L2F observation must be 13-dim");
static_assert(Env::ACTION_DIM      == 4,  "L2F action must be 4-dim (4 rotors)");

using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::OBSERVATION_DIM, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::ACTION_DIM, false>>;

namespace {
std::unique_ptr<Encoder> makeEncoder(wann::SnnEncoder type, uint32_t seed) {
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
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LINEAR, 1e-9);
        case wann::SnnEncoder::TTFS_LOG:
            return std::make_unique<TTFSEncoder>(TTFSEncoder::Mapping::LOGARITHMIC, 1e-9);
        case wann::SnnEncoder::SMALL:
        case wann::SnnEncoder::LARGE:
            return nullptr;  // current injection, not spike trains
        default:
            return nullptr;  // CURRENT handled separately
    }
}

// Decode N independent continuous-action spike trains (one per rotor) into
// normalized commands in [-1, 1]. Same convention as SnnCarTask's
// decodeActions(), generalized from 2 outputs to nOutput_ outputs.
std::vector<double> decodeContinuousActions(
    const std::vector<std::vector<double>>& multi_spikes,
    wann::SnnDecoder decoder,
    const RLDecoder&  rl_decoder,
    double            max_spikes)
{
    std::vector<double> action(multi_spikes.size(), 0.0);
    for (size_t o = 0; o < multi_spikes.size(); ++o) {
        double a;
        switch (decoder) {
            case wann::SnnDecoder::FIRST_SPIKE:
                a = rl_decoder.decodeContinuousAction(multi_spikes[o]) * 2.0 - 1.0;
                break;
            case wann::SnnDecoder::SPIKE_COUNT:
                a = rl_decoder.decodeContinuousAction(multi_spikes[o]);
                break;
            default:
                a = static_cast<double>(multi_spikes[o].size()) / max_spikes * 2.0 - 1.0;
                break;
        }
        action[o] = std::clamp(a, -1.0, 1.0);
    }
    return action;
}
} // anonymous namespace

namespace wann {

const double SnnL2FTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 1.5, 2.0, 5.0, 8.0
};

SnnL2FTask::SnnL2FTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , neuronsPerVar_(hyp.snn_neurons_per_var)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnL2FTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        case 7:  return NeuronType::FAST_SPIKING;
        case 8:  return NeuronType::REGULAR_SPIKING;
        case 9:  return NeuronType::REGULAR_SPIKING;
        case 10: return NeuronType::CHATTERING;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

Network SnnL2FTask::buildNetwork(const Ind& ind) const
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

    return net;
}

Network SnnL2FTask::buildNetwork(const std::vector<double>& wVec,
                                 const std::vector<int>&    aVec) const
{
    const int N = static_cast<int>(std::sqrt(static_cast<double>(wVec.size())));
    Network net(1.0, true);
    std::vector<int> snn_id(N, -1);

    snn_id[0] = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
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

    return net;
}

namespace {
// Normalization ranges for the 13 observation channels, chosen with margin
// over the MDP termination thresholds (position=1m, linear_velocity=2m/s,
// angular_velocity=35rad/s) so episodes normally terminate before saturating.
constexpr double POS_ERR_RANGE    = 1.5;   // ±1.5 m
constexpr double LINVEL_ERR_RANGE = 2.5;   // ±2.5 m/s
constexpr double ANGVEL_RANGE     = 35.0;  // ±35 rad/s

// obs = [pos_err_x,y,z, qw,qx,qy,qz, vel_err_x,y,z, wx,wy,wz]
void fillNormalizedChannels(const std::array<double, 13>& obs, std::vector<double>& norm, int nInput) {
    norm[0] = 1.0;  // bias
    if (nInput >= 1)  norm[1]  = (obs[0]  / POS_ERR_RANGE    + 1.0) * 0.5;
    if (nInput >= 2)  norm[2]  = (obs[1]  / POS_ERR_RANGE    + 1.0) * 0.5;
    if (nInput >= 3)  norm[3]  = (obs[2]  / POS_ERR_RANGE    + 1.0) * 0.5;
    if (nInput >= 4)  norm[4]  = (obs[3]  + 1.0) * 0.5;  // qw ∈ [-1,1]
    if (nInput >= 5)  norm[5]  = (obs[4]  + 1.0) * 0.5;  // qx
    if (nInput >= 6)  norm[6]  = (obs[5]  + 1.0) * 0.5;  // qy
    if (nInput >= 7)  norm[7]  = (obs[6]  + 1.0) * 0.5;  // qz
    if (nInput >= 8)  norm[8]  = (obs[7]  / LINVEL_ERR_RANGE + 1.0) * 0.5;
    if (nInput >= 9)  norm[9]  = (obs[8]  / LINVEL_ERR_RANGE + 1.0) * 0.5;
    if (nInput >= 10) norm[10] = (obs[9]  / LINVEL_ERR_RANGE + 1.0) * 0.5;
    if (nInput >= 11) norm[11] = (obs[10] / ANGVEL_RANGE + 1.0) * 0.5;
    if (nInput >= 12) norm[12] = (obs[11] / ANGVEL_RANGE + 1.0) * 0.5;
    if (nInput >= 13) norm[13] = (obs[12] / ANGVEL_RANGE + 1.0) * 0.5;
}

constexpr std::array<std::pair<double, double>, 13> OBS_LIMITS = {{
    {-POS_ERR_RANGE,    POS_ERR_RANGE},
    {-POS_ERR_RANGE,    POS_ERR_RANGE},
    {-POS_ERR_RANGE,    POS_ERR_RANGE},
    {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0},
    {-LINVEL_ERR_RANGE, LINVEL_ERR_RANGE},
    {-LINVEL_ERR_RANGE, LINVEL_ERR_RANGE},
    {-LINVEL_ERR_RANGE, LINVEL_ERR_RANGE},
    {-ANGVEL_RANGE,     ANGVEL_RANGE},
    {-ANGVEL_RANGE,     ANGVEL_RANGE},
    {-ANGVEL_RANGE,     ANGVEL_RANGE},
}};
} // anonymous namespace

std::pair<double,double> SnnL2FTask::runEpisode(Network& net, double sharedWeight, long long episodeSeed) const
{
    net.fastReset();
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);
    rlt::sample_initial_parameters(device, env, params, rng);  // fills the Lissajous trajectory table

    typename Env::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;  // bias + 13 observations

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    const RLDecoder::DecodingType dec_type =
        (decoder_ == SnnDecoder::FIRST_SPIKE) ? RLDecoder::DecodingType::FIRST_SPIKE :
        (decoder_ == SnnDecoder::SPIKE_COUNT) ? RLDecoder::DecodingType::SPIKE_COUNT :
                                                RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    double total_reward = 0.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        std::array<double, 13> obs;
        for (int i = 0; i < 13; ++i) obs[i] = rlt::get(obs_mat, 0, i);

        if (resetBetweenSteps_) net.fastReset();
        std::vector<std::vector<double>> multi_spikes(nOutput_);

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            fillNormalizedChannels(obs, norm, nInput_);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                for (int o = 0; o < nOutput_ && o < (int)out.size(); ++o)
                    if (out[o]) multi_spikes[o].push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;

            if (encoder_ == SnnEncoder::SMALL || encoder_ == SnnEncoder::LARGE) {
                RLEncoder rl_enc(encoder_ == SnnEncoder::SMALL
                                 ? RLEncoder::EncodingType::SMALL
                                 : RLEncoder::EncodingType::LARGE,
                                 100.0, 5, static_cast<size_t>(neuronsPerVar_));
                const int n_obs = (encoder_ == SnnEncoder::SMALL)
                                  ? nInput_ / 2
                                  : nInput_ / neuronsPerVar_;
                std::vector<double> obs_vals(obs.begin(), obs.begin() + std::min(n_obs, 13));
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    const std::vector<std::pair<double,double>> limits(
                        OBS_LIMITS.begin(), OBS_LIMITS.begin() + obs_vals.size());
                    enc_currents = rl_enc.encodeObservationLarge(obs_vals, limits);
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                // CURRENT: map each observation to [0, 20] mA
                std::vector<double> norm(n_channels, 0.0);
                fillNormalizedChannels(obs, norm, nInput_);
                for (int ch = 1; ch < n_channels; ++ch)
                    currents[ch] = std::clamp(norm[ch], 0.0, 1.0) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                const auto& out = net.getOutputSpikes();
                for (int o = 0; o < nOutput_ && o < (int)out.size(); ++o)
                    if (out[o]) multi_spikes[o].push_back(static_cast<double>(t));
            }
        }

        auto action = decodeContinuousActions(multi_spikes, decoder_, rl_decoder, max_spikes);
        for (int o = 0; o < static_cast<int>(Env::ACTION_DIM); ++o)
            rlt::set(action_mat, 0, o, o < nOutput_ ? action[o] : 0.0);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return {total_reward, total_reward};
}

std::vector<double> SnnL2FTask::evaluate(const Ind& ind, int seed)
{
    Network net = buildNetwork(ind);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            long long episodeSeed = static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed).first;
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

std::vector<double> SnnL2FTask::getDistFitness(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        int seed)
{
    Network net = buildNetwork(wVec, aVec);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            long long episodeSeed = static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed).first;
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

std::pair<std::vector<double>,std::vector<double>> SnnL2FTask::evalEpisodes(
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

void SnnL2FTask::exportTrajectory(const std::vector<double>& wVec,
                                  const std::vector<int>&    aVec,
                                  int bestWi, int evalSeed,
                                  const std::string& outFile,
                                  bool directSeed) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,pos_err_x,pos_err_y,pos_err_z,qw,qx,qy,qz,"
           "vel_err_x,vel_err_y,vel_err_z,wx,wy,wz,"
           "rotor0,rotor1,rotor2,rotor3,reward\n";

    Network net = buildNetwork(wVec, aVec);

    long long episodeSeed;
    if (directSeed) {
        episodeSeed = evalSeed;
    } else {
        episodeSeed = static_cast<long long>(evalSeed) * 10000 + bestWi * 100 + 0;
    }
    const double weight = WEIGHT_VALS[bestWi];

    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);
    rlt::sample_initial_parameters(device, env, params, rng);

    typename Env::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;

    constexpr double DT = 1.0;
    auto enc = makeEncoder(encoder_, static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);

    const RLDecoder::DecodingType dec_type =
        (decoder_ == SnnDecoder::FIRST_SPIKE) ? RLDecoder::DecodingType::FIRST_SPIKE :
        (decoder_ == SnnDecoder::SPIKE_COUNT) ? RLDecoder::DecodingType::SPIKE_COUNT :
                                                RLDecoder::DecodingType::RATE;
    RLDecoder rl_decoder(dec_type, SIM_WINDOW_MS);
    const double max_spikes = static_cast<double>(window_steps) / 2.0;

    for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        std::array<double, 13> obs;
        for (int i = 0; i < 13; ++i) obs[i] = rlt::get(obs_mat, 0, i);

        if (resetBetweenSteps_) net.fastReset();
        std::vector<std::vector<double>> multi_spikes(nOutput_);

        if (encoder_ != SnnEncoder::CURRENT && encoder_ != SnnEncoder::SMALL && encoder_ != SnnEncoder::LARGE) {
            std::vector<double> norm(n_channels, 0.0);
            fillNormalizedChannels(obs, norm, nInput_);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double v = std::clamp(norm[ch], 0.0, 1.0);
                spike_trains[ch] = enc->encode(v, SIM_WINDOW_MS, DT);
            }
            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(weight);
                const auto& out = net.getOutputSpikes();
                for (int o = 0; o < nOutput_ && o < (int)out.size(); ++o)
                    if (out[o]) multi_spikes[o].push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;

            if (encoder_ == SnnEncoder::SMALL || encoder_ == SnnEncoder::LARGE) {
                RLEncoder rl_enc(encoder_ == SnnEncoder::SMALL
                                 ? RLEncoder::EncodingType::SMALL
                                 : RLEncoder::EncodingType::LARGE,
                                 100.0, 5, static_cast<size_t>(neuronsPerVar_));
                const int n_obs = (encoder_ == SnnEncoder::SMALL)
                                  ? nInput_ / 2
                                  : nInput_ / neuronsPerVar_;
                std::vector<double> obs_vals(obs.begin(), obs.begin() + std::min(n_obs, 13));
                std::vector<double> enc_currents;
                if (encoder_ == SnnEncoder::SMALL) {
                    enc_currents = rl_enc.encodeObservationSmall(obs_vals);
                } else {
                    const std::vector<std::pair<double,double>> limits(
                        OBS_LIMITS.begin(), OBS_LIMITS.begin() + obs_vals.size());
                    enc_currents = rl_enc.encodeObservationLarge(obs_vals, limits);
                }
                for (size_t i = 0; i < enc_currents.size() && i + 1 < static_cast<size_t>(n_channels); ++i)
                    currents[i + 1] = enc_currents[i];
            } else {
                std::vector<double> norm(n_channels, 0.0);
                fillNormalizedChannels(obs, norm, nInput_);
                for (int ch = 1; ch < n_channels; ++ch)
                    currents[ch] = std::clamp(norm[ch], 0.0, 1.0) * 20.0;
            }

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(weight);
                const auto& out = net.getOutputSpikes();
                for (int o = 0; o < nOutput_ && o < (int)out.size(); ++o)
                    if (out[o]) multi_spikes[o].push_back(static_cast<double>(t));
            }
        }

        auto action = decodeContinuousActions(multi_spikes, decoder_, rl_decoder, max_spikes);
        for (int o = 0; o < static_cast<int>(Env::ACTION_DIM); ++o)
            rlt::set(action_mat, 0, o, o < nOutput_ ? action[o] : 0.0);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        double reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        csv << step << ',';
        for (int i = 0; i < 13; ++i) csv << obs[i] << ',';
        for (int o = 0; o < 4; ++o) csv << (o < nOutput_ ? action[o] : 0.0) << ',';
        csv << reward << '\n';

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

#include "../include/wann/SnnDiscMCTask.h"

#include <core/simulator.hpp>
#include <decoding/rlDecoder.hpp>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/mountain_car/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG    = typename rlt::devices::random::CPU::ENGINE<>;

// Custom parameters matching Gymnasium MountainCar-v0 (discrete).
struct DiscMCParams {
    static constexpr T min_action         = -1.0;
    static constexpr T max_action         =  1.0;
    static constexpr T min_position       = -1.2;
    static constexpr T max_position       =  0.6;
    static constexpr T max_speed          =  0.07;
    static constexpr T power              =  0.001;  // Gymnasium discrete power
    static constexpr T goal_position      =  0.5;    // Gymnasium discrete goal
    static constexpr T goal_velocity      =  0.0;
    static constexpr T action_cost        =  0.0;    // no action penalty
    static constexpr T goal_reward        =  0.0;    // reward handled manually
    static constexpr T initial_position_min = -0.6;
    static constexpr T initial_position_max = -0.4;
};

using MCSpec    = rlt::rl::environments::mountain_car::Specification<T, TI, DiscMCParams>;
using Env       = rlt::rl::environments::MountainCarContinuous<MCSpec>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;

static constexpr int    EPISODE_STEPS = 200;
static constexpr double POS_MIN = -1.2, POS_MAX = 0.6;
static constexpr double VEL_MAX = 0.07;

namespace wann {

const double SnnDiscMCTask::WEIGHT_VALS[N_WEIGHTS] = {
    0.5, 1.0, 2.0, 3.0, 5.0, 8.0
};

SnnDiscMCTask::SnnDiscMCTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , encoder_(parseEncoder(hyp.snn_encoder))
    , decoder_(parseDecoder(hyp.snn_decoder))
    , resetBetweenSteps_(hyp.snn_reset_between_steps)
{}

NeuronType SnnDiscMCTask::wannActToNeuronType(int actId) {
    switch (actId) {
        case 1:  return NeuronType::REGULAR_SPIKING;
        case 2:  return NeuronType::FAST_SPIKING;
        case 3:  return NeuronType::CHATTERING;
        case 4:  return NeuronType::LOW_THRESHOLD_SPIKING;
        case 5:  return NeuronType::INTRINSICALLY_BURSTING;
        case 6:  return NeuronType::RESONATOR;
        default: return NeuronType::REGULAR_SPIKING;
    }
}

Network SnnDiscMCTask::buildNetwork(const Ind& ind) const
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

Network SnnDiscMCTask::buildNetwork(const std::vector<double>& wVec,
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
                net.addSynapse(snn_id[i], snn_id[j], true);
        }
    }

    return net;
}

double SnnDiscMCTask::runEpisode(Network& net, double sharedWeight, int episodeSeed) const
{
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mountain_car::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;

    constexpr double MAX_RATE   = 100.0;
    constexpr double REF_PERIOD = 2.0;
    constexpr double DT         = 1.0;
    std::mt19937 enc_rng(static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const RLDecoder::DecodingType disc_type = (decoder_ == SnnDecoder::VOTING)
        ? RLDecoder::DecodingType::VOTING
        : RLDecoder::DecodingType::WINNER_TAKES_ALL;
    RLDecoder disc_decoder(disc_type, SIM_WINDOW_MS, 5);

    double total_reward = 0.0;

    for (int step = 0; step < EPISODE_STEPS; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        if (resetBetweenSteps_) net.fastReset();

        std::vector<std::vector<double>> multi_spikes(N_ACTIONS);

        if (encoder_ != SnnEncoder::CURRENT) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (rlt::get(obs_mat, 0, 0) - POS_MIN) / (POS_MAX - POS_MIN);
            if (nInput_ >= 2) norm[2] = (rlt::get(obs_mat, 0, 1) + VEL_MAX) / (2.0 * VEL_MAX);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double rate   = std::clamp(norm[ch], 0.0, 1.0) * MAX_RATE;
                double sp_prob = (rate / 1000.0) * DT;
                double last_t  = -REF_PERIOD - 1.0;
                for (int t = 0; t < window_steps; ++t) {
                    double ct = static_cast<double>(t);
                    if ((ct - last_t) >= REF_PERIOD && udist(enc_rng) < sp_prob) {
                        spike_trains[ch].push_back(ct);
                        last_t = ct;
                    }
                }
            }

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(sharedWeight);
                const auto& outs = net.getOutputSpikes();
                for (int a = 0; a < N_ACTIONS; ++a)
                    if ((int)outs.size() > a && outs[a])
                        multi_spikes[a].push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;
            if (nInput_ >= 1)
                currents[1] = (rlt::get(obs_mat, 0, 0) - POS_MIN) / (POS_MAX - POS_MIN) * 20.0;
            if (nInput_ >= 2)
                currents[2] = (rlt::get(obs_mat, 0, 1) + VEL_MAX) / (2.0 * VEL_MAX) * 20.0;

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(sharedWeight);
                const auto& outs = net.getOutputSpikes();
                for (int a = 0; a < N_ACTIONS; ++a)
                    if ((int)outs.size() > a && outs[a])
                        multi_spikes[a].push_back(static_cast<double>(t));
            }
        }

        int winner;
        if (decoder_ == SnnDecoder::FIRST_SPIKE) {
            winner = N_ACTIONS / 2;
            double earliest = SIM_WINDOW_MS + 1.0;
            for (int a = 0; a < N_ACTIONS; ++a)
                if (!multi_spikes[a].empty() && multi_spikes[a][0] < earliest) {
                    earliest = multi_spikes[a][0];
                    winner = a;
                }
        } else {
            winner = disc_decoder.decodeDiscreteAction(multi_spikes);
        }

        double force = static_cast<double>(winner - 1);
        rlt::set(action_mat, 0, 0, force);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        total_reward += -1.0;  // Gymnasium MountainCar-v0: -1 per step
        state = next_state;

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    return total_reward;
}

std::vector<double> SnnDiscMCTask::evaluate(const Ind& ind, int seed)
{
    Network net = buildNetwork(ind);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            int episodeSeed = (seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed);
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

std::vector<double> SnnDiscMCTask::getDistFitness(
        const std::vector<double>& wVec,
        const std::vector<int>&    aVec,
        int seed)
{
    Network net = buildNetwork(wVec, aVec);

    std::vector<double> rewards(N_WEIGHTS, 0.0);
    for (int wi = 0; wi < N_WEIGHTS; ++wi) {
        double total = 0.0;
        for (int rep = 0; rep < nReps_; ++rep) {
            int episodeSeed = (seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
            total += runEpisode(net, WEIGHT_VALS[wi], episodeSeed);
        }
        rewards[wi] = total / static_cast<double>(nReps_);
    }
    return rewards;
}

void SnnDiscMCTask::exportTrajectory(const std::vector<double>& wVec,
                                     const std::vector<int>&    aVec,
                                     int bestWi, int evalSeed,
                                     const std::string& outFile) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,position,velocity,action,reward\n";

    Network net = buildNetwork(wVec, aVec);

    // Warm-up: reproduce SNN state from training
    for (int wi = 0; wi < bestWi; ++wi)
        for (int rep = 0; rep < nReps_; ++rep)
            runEpisode(net, WEIGHT_VALS[wi], evalSeed * 10000 + wi * 100 + rep);

    const int    episodeSeed = evalSeed * 10000 + bestWi * 100 + 0;
    const double weight      = WEIGHT_VALS[bestWi];

    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(episodeSeed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mountain_car::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;

    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int window_steps = static_cast<int>(SIM_WINDOW_MS);
    const int n_channels   = nInput_ + 1;

    constexpr double MAX_RATE   = 100.0;
    constexpr double REF_PERIOD = 2.0;
    constexpr double DT         = 1.0;
    std::mt19937 enc_rng(static_cast<uint32_t>(episodeSeed) ^ 0xDEADBEEFu);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const RLDecoder::DecodingType disc_type = (decoder_ == SnnDecoder::VOTING)
        ? RLDecoder::DecodingType::VOTING
        : RLDecoder::DecodingType::WINNER_TAKES_ALL;
    RLDecoder disc_decoder(disc_type, SIM_WINDOW_MS, 5);

    for (int step = 0; step < EPISODE_STEPS; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

        double pos = rlt::get(obs_mat, 0, 0);
        double vel = rlt::get(obs_mat, 0, 1);

        if (resetBetweenSteps_) net.fastReset();

        std::vector<std::vector<double>> multi_spikes(N_ACTIONS);

        if (encoder_ != SnnEncoder::CURRENT) {
            std::vector<double> norm(n_channels, 0.0);
            norm[0] = 1.0;
            if (nInput_ >= 1) norm[1] = (pos - POS_MIN) / (POS_MAX - POS_MIN);
            if (nInput_ >= 2) norm[2] = (vel + VEL_MAX) / (2.0 * VEL_MAX);

            std::vector<std::vector<double>> spike_trains(n_channels);
            for (int ch = 0; ch < n_channels; ++ch) {
                double rate    = std::clamp(norm[ch], 0.0, 1.0) * MAX_RATE;
                double sp_prob = (rate / 1000.0) * DT;
                double last_t  = -REF_PERIOD - 1.0;
                for (int t = 0; t < window_steps; ++t) {
                    double ct = static_cast<double>(t);
                    if ((ct - last_t) >= REF_PERIOD && udist(enc_rng) < sp_prob) {
                        spike_trains[ch].push_back(ct);
                        last_t = ct;
                    }
                }
            }

            for (int t = 0; t < window_steps; ++t) {
                net.applyInputSpikes(spike_trains, net.getCurrentTime());
                net.step(weight);
                const auto& outs = net.getOutputSpikes();
                for (int a = 0; a < N_ACTIONS; ++a)
                    if ((int)outs.size() > a && outs[a])
                        multi_spikes[a].push_back(static_cast<double>(t));
            }
        } else {
            std::vector<double> currents(n_channels, 0.0);
            currents[0] = BIAS_CURRENT;
            if (nInput_ >= 1)
                currents[1] = (pos - POS_MIN) / (POS_MAX - POS_MIN) * 20.0;
            if (nInput_ >= 2)
                currents[2] = (vel + VEL_MAX) / (2.0 * VEL_MAX) * 20.0;

            for (int t = 0; t < window_steps; ++t) {
                net.setInputCurrents(currents);
                net.step(weight);
                const auto& outs = net.getOutputSpikes();
                for (int a = 0; a < N_ACTIONS; ++a)
                    if ((int)outs.size() > a && outs[a])
                        multi_spikes[a].push_back(static_cast<double>(t));
            }
        }

        int winner;
        if (decoder_ == SnnDecoder::FIRST_SPIKE) {
            winner = N_ACTIONS / 2;
            double earliest = SIM_WINDOW_MS + 1.0;
            for (int a = 0; a < N_ACTIONS; ++a)
                if (!multi_spikes[a].empty() && multi_spikes[a][0] < earliest) {
                    earliest = multi_spikes[a][0];
                    winner = a;
                }
        } else {
            winner = disc_decoder.decodeDiscreteAction(multi_spikes);
        }

        double force = static_cast<double>(winner - 1);
        double reward = -1.0;
        rlt::set(action_mat, 0, 0, force);
        rlt::step(device, env, params, state, action_mat, next_state, rng);
        state = next_state;

        csv << step    << ','
            << pos     << ',' << vel    << ','
            << winner  << ',' << reward << '\n';

        if (rlt::terminated(device, env, params, state, rng)) break;
    }

    std::cout << "Trayectoria guardada en: " << outFile << '\n';
}

} // namespace wann

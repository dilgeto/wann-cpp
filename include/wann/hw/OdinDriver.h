#pragma once
#include "../OdinExport.h"

#include <array>
#include <cstdint>
#include <vector>

namespace wann::hw {

// Minimal AXI-Lite MMIO window (one per IP core) — mirrors what pynq.MMIO
// gives odin.py, but via direct /dev/mem mmap since there's no PYNQ/Python
// on the C++ side.
class MmioWindow {
public:
    MmioWindow(int memFd, std::uintptr_t physAddr, std::size_t span);
    ~MmioWindow();
    MmioWindow(const MmioWindow&) = delete;
    MmioWindow& operator=(const MmioWindow&) = delete;

    std::uint32_t read(std::uintptr_t offset) const;
    void          write(std::uintptr_t offset, std::uint32_t value);

private:
    void* base_ = nullptr;
    std::size_t span_ = 0;
};

// C++ port of odin.py's QuadSpiMaster: minimal AXI Quad SPI (LogiCORE
// PG153) driver, standard SPI mode, MSB first, mode 0.
class QuadSpiMaster {
public:
    explicit QuadSpiMaster(MmioWindow& mmio);
    std::vector<std::uint8_t> xfer(const std::vector<std::uint8_t>& tx);

private:
    MmioWindow& mmio_;
};

// C++ port of odin.py's Odin class — same method names/semantics, talking
// to the same AER GPIOs + AXI Quad SPI config port over /dev/mem instead of
// PYNQ. Protocol confirmed against ODIN's official doc (SPI 20-bit
// addressing, AER 4-phase handshake / 17-bit input event encoding) and the
// user's own working odin.py.
class OdinDriver {
public:
    OdinDriver();
    ~OdinDriver();
    OdinDriver(const OdinDriver&) = delete;
    OdinDriver& operator=(const OdinDriver&) = delete;

    // Full chip bring-up: stop, wipe all 65536 synapses and 256 neurons to a
    // safe state, clear global registers. Slow (~30s, dominated by the
    // synapse wipe) — matches odin.py's init(); call once per power cycle
    // /before loading a new network, not once per episode.
    void init(bool verbose = true);

    void stop();   // GATE_ACTIVITY = 1 (safe to program)
    void start();  // GATE_ACTIVITY = 0 (chip runs)

    // Programs one neuron as LIF — used internally by init()/resetAllNeurons()
    // to put every neuron in a safe disabled state (thr=255) before loading a
    // real config. Mirrors odin.py's neuron_lif().
    void neuronLif(int neuronId, int thr = 14, int leakStr = 10, int leakEn = 1,
                   int caEn = 0, int thetamem = 0, int caTheta1 = 0,
                   int caTheta2 = 0, int caTheta3 = 0, int caLeak = 0);

    // Programs one neuron as a phenomenological Izhikevich (IZH) model — bit
    // layout confirmed against neuron_core.v (LSB=0 selects IZH mode). See
    // wann::IzhParams (OdinExport.h) for the caveats on the parameter values
    // themselves (best-effort mapping, not an ODIN-published table).
    void neuronIzh(int neuronId, const wann::IzhParams& p);

    // weight in [0,7]; mapped=0 clears the synapse (matches odin.py).
    void synapse(int pre, int post, int weight, int mapped = 1);

    // Sign (excitatory/inhibitory) is a per-PRE-SYNAPTIC-NEURON property in
    // real ODIN hardware (Dale's law), not per individual synapse — the
    // synapse SRAM only stores weight + mapped bit (confirmed in ODIN's
    // official doc, section 3.2). This sets the sign broadcast to every
    // outgoing synapse of `neuronId`.
    void setSynSign(int neuronId, bool inhibitory);

    // Loads every neuron + synapse from `cfg` (see OdinExport.h) onto a
    // freshly-init()'d chip, leaving it stopped — caller must start().
    void loadConfig(const wann::OdinNetworkConfig& cfg, bool verbose = true);

    // Stimulates `neuronId` directly (bypassing the synaptic crossbar) with
    // the given weight/sign — this is how encoded input spikes are injected
    // (odin.py's send_virtual / run_sample pattern), not a real AER
    // single-synapse event.
    int sendVirtual(int neuronId, int weight = 7, int leakBit = 0, int inhibBit = 0,
                    int timeoutUs = 100'000);

    // Makes the chip treat `neuronId` as if it had just fired: its outgoing
    // synapses are stimulated (sign from SPI_SYN_SIGN[neuronId]), without
    // going through the neuron's own membrane/threshold (AER event
    // {0, pre, 0x07}). This is what an encoded input spike must do —
    // snn-simulator's Network::applyInputSpikes forces the input neuron to
    // fire (conductance 100) — unlike sendVirtual(), which only nudges the
    // neuron's own membrane by ~1 pulse and rarely reaches threshold.
    int sendNeuronSpike(int neuronId, int timeoutUs = 100'000);

    // Drains AER_OUT for up to `timeoutUs` of wall-clock time, returning the
    // address of every neuron that fired during that window (in order).
    std::vector<int> drainSpikes(int timeoutUs);

    // Clears any AER IN/OUT handshake left mid-transaction (e.g. after a
    // previous run was interrupted) — cheap, safe to call defensively.
    void aerBusReset();

    // Broadcasts one "tref" (time reference) AER event to every neuron —
    // this is the IZH model's own clock tick, not a diagnostic. Without
    // periodic trefs an IZH neuron's refractory/burst/latency counters never
    // advance: it fires once and then stays stuck in refractory forever, or
    // never reaches the delay a latency/rebound behaviour needs to fire at
    // all (confirmed against ODIN's verified param docs — see
    // OdinExport.h's IzhParams). LIF neurons don't need this to fire on
    // threshold crossing, only for their leak.
    int sendTrefAll(int timeoutUs = 100'000);

    // Global register 18 (SPI_BURST_TIMEREF). Must be != 0 (1023 is ODIN's
    // own recommended value) for any IZH neuron using spk_ref != 0
    // (CHATTERING/INTRINSICALLY_BURSTING in OdinExport's izhParamsFor) —
    // without it those neurons "quedan enganchadas" (get stuck) instead of
    // bursting. init() zeroes this along with every other global register,
    // so call this again after loadConfig().
    void setBurstTimeref(int value = 1023);

    // Global register 19 (SPI_AER_SRC_CTRL_nNEUR). Needed alongside
    // setBurstTimeref() for burst behaviours to actually work, not just for
    // visibility — enabling it also makes AER_OUT report every scheduler
    // event, not just neuron spikes, but drainSpikes()/OdinNetwork only look
    // for known output addresses, so the extra events are harmlessly
    // ignored. Also zeroed by init(); call again after loadConfig().
    void setAerSrcCtrl(bool enabled);

private:
    int memFd_ = -1;
    MmioWindow aerIn_;
    MmioWindow aerInAck_;
    MmioWindow aerOut_;
    MmioWindow aerOutAck_;
    MmioWindow spiMmio_;
    QuadSpiMaster spi_;

    std::array<std::uint32_t, 16> synSignCache_{};

    void sendNeuronWord(int neuronId, const std::array<int, 128>& bits);

    static std::uint32_t addrGlobal(int index);
    static std::uint32_t addrNeuronByte(int neuronId, int byteOffset);
    static std::uint32_t addrSynapse(int pre, int post);

    void spiSend40(std::uint32_t addr20, std::uint32_t data20);

    void resetAllNeurons();
    void resetAllSynapses();

    static std::uint32_t aer17(int bit16, int hi8, int lo8);
    int aerSendRaw(std::uint32_t addr17, int timeoutUs);
};

} // namespace wann::hw

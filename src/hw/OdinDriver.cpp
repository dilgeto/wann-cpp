#include "../../include/wann/hw/OdinDriver.h"
#include "../../include/wann/hw/OdinRegisters.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace wann::hw {

using Clock = std::chrono::steady_clock;

namespace {
long long elapsedUs(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
}

// ===========================================================================
// MmioWindow
// ===========================================================================
MmioWindow::MmioWindow(int memFd, std::uintptr_t physAddr, std::size_t span)
    : span_(span)
{
    base_ = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, memFd,
                 static_cast<off_t>(physAddr));
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        throw std::runtime_error("MmioWindow: mmap failed for phys addr 0x" +
                                  std::to_string(physAddr));
    }
}

MmioWindow::~MmioWindow() {
    if (base_) munmap(base_, span_);
}

std::uint32_t MmioWindow::read(std::uintptr_t offset) const {
    auto* p = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(base_) + offset);
    return *p;
}

void MmioWindow::write(std::uintptr_t offset, std::uint32_t value) {
    auto* p = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(base_) + offset);
    *p = value;
}

namespace {
// Masked read-modify-write, matching pynq.lib.axigpio.AxiGPIO.Channel.write(value, mask).
void writeMasked(MmioWindow& w, std::uintptr_t offset, std::uint32_t value, std::uint32_t mask) {
    std::uint32_t cur = w.read(offset);
    w.write(offset, (cur & ~mask) | (value & mask));
}
}

// ===========================================================================
// QuadSpiMaster — port of odin.py's QuadSpiMaster
// ===========================================================================
QuadSpiMaster::QuadSpiMaster(MmioWindow& mmio) : mmio_(mmio) {
    mmio_.write(QSPI_SRR, 0x0A);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    std::uint32_t cfg = SPICR_MASTER | SPICR_MANUAL_SS | SPICR_TRANS_INHIBIT |
                        SPICR_TXFIFO_RST | SPICR_RXFIFO_RST;
    mmio_.write(QSPI_SPICR, cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    cfg = SPICR_MASTER | SPICR_MANUAL_SS | SPICR_SPE | SPICR_TRANS_INHIBIT;
    mmio_.write(QSPI_SPICR, cfg);
    mmio_.write(QSPI_SPISSR, 0xFFFFFFFFu);
}

std::vector<std::uint8_t> QuadSpiMaster::xfer(const std::vector<std::uint8_t>& tx) {
    for (auto b : tx) mmio_.write(QSPI_SPIDTR, b & 0xFF);

    mmio_.write(QSPI_SPISSR, 0x00000000u);

    std::uint32_t cfg = SPICR_MASTER | SPICR_MANUAL_SS | SPICR_SPE;
    mmio_.write(QSPI_SPICR, cfg);

    auto deadline = Clock::now() + std::chrono::milliseconds(100);
    while (!(mmio_.read(QSPI_SPISR) & SPISR_TX_EMPTY)) {
        if (Clock::now() > deadline)
            throw std::runtime_error("QuadSpiMaster: SPI TX timeout");
    }

    cfg = SPICR_MASTER | SPICR_MANUAL_SS | SPICR_SPE | SPICR_TRANS_INHIBIT;
    mmio_.write(QSPI_SPICR, cfg);
    mmio_.write(QSPI_SPISSR, 0xFFFFFFFFu);

    std::vector<std::uint8_t> rx;
    while (!(mmio_.read(QSPI_SPISR) & SPISR_RX_EMPTY))
        rx.push_back(static_cast<std::uint8_t>(mmio_.read(QSPI_SPIDRR) & 0xFF));
    return rx;
}

// ===========================================================================
// OdinDriver
// ===========================================================================
namespace {
int openOdinMemFd() {
    if (!ODIN_REGISTERS_CONFIGURED) {
        throw std::logic_error(
            "OdinDriver: include/wann/hw/OdinRegisters.h still has placeholder "
            "(0x0) IP base addresses. Get the real physical addresses from "
            "ov.ip_dict on the board, fill them in, set "
            "ODIN_REGISTERS_CONFIGURED = true, then rebuild.");
    }
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0)
        throw std::runtime_error("OdinDriver: cannot open /dev/mem (run as root)");
    return fd;
}
}

// NOTE: this assumes the FPGA is already programmed with odin.bit and its
// AXI GPIO channel directions already configured — both normally done by
// PYNQ's Overlay() in Python, which this driver deliberately doesn't
// reimplement (bitstream download / clock & reset sequencing is
// board-specific and PYNQ already does it correctly). Run the one-time
// Python bootstrap (see odin_bootstrap.py) before this binary.
OdinDriver::OdinDriver()
    : memFd_(openOdinMemFd())
    , aerIn_(memFd_, ODIN_AER_IN_BASE_ADDR, ODIN_IP_WINDOW)
    , aerInAck_(memFd_, ODIN_AER_IN_ACK_BASE_ADDR, ODIN_IP_WINDOW)
    , aerOut_(memFd_, ODIN_AER_OUT_BASE_ADDR, ODIN_IP_WINDOW)
    , aerOutAck_(memFd_, ODIN_AER_OUT_ACK_BASE_ADDR, ODIN_IP_WINDOW)
    , spiMmio_(memFd_, ODIN_SPI_BASE_ADDR, ODIN_IP_WINDOW)
    , spi_(spiMmio_)
{
}

OdinDriver::~OdinDriver() {
    if (memFd_ >= 0) close(memFd_);
}

// ------------------------------------------------------------------
// Addressing (mirrors odin.py _addr_global / _addr_neuron_byte / _addr_synapse)
// ------------------------------------------------------------------
std::uint32_t OdinDriver::addrGlobal(int index) {
    return static_cast<std::uint32_t>(index) & 0x1F;
}

std::uint32_t OdinDriver::addrNeuronByte(int neuronId, int byteOffset) {
    return (1u << 18) | (1u << 16) |
           ((static_cast<std::uint32_t>(byteOffset) & 0xF) << 8) |
           (static_cast<std::uint32_t>(neuronId) & 0xFF);
}

std::uint32_t OdinDriver::addrSynapse(int pre, int post) {
    std::uint32_t a = 0;
    a |= (static_cast<std::uint32_t>(post) >> 3) & 0x1F;
    a |= static_cast<std::uint32_t>(pre) << 5;
    a |= ((static_cast<std::uint32_t>(post) >> 1) & 0x3) << 13;
    return (1u << 18) | (2u << 16) | (a & 0xFFFF);
}

void OdinDriver::spiSend40(std::uint32_t addr20, std::uint32_t data20) {
    std::uint64_t pkt = (static_cast<std::uint64_t>(addr20 & 0xFFFFF) << 20) |
                        (data20 & 0xFFFFF);
    std::vector<std::uint8_t> tx = {
        static_cast<std::uint8_t>((pkt >> 32) & 0xFF),
        static_cast<std::uint8_t>((pkt >> 24) & 0xFF),
        static_cast<std::uint8_t>((pkt >> 16) & 0xFF),
        static_cast<std::uint8_t>((pkt >>  8) & 0xFF),
        static_cast<std::uint8_t>( pkt        & 0xFF),
    };
    spi_.xfer(tx);
}

// ------------------------------------------------------------------
// Init / stop / start
// ------------------------------------------------------------------
void OdinDriver::init(bool verbose) {
    auto t0 = Clock::now();

    writeMasked(aerIn_, GPIO_DATA, 0xFFFFFFFFu, 0xFFFFFFFFu);
    writeMasked(aerIn_, GPIO2_DATA, 0xFFFFFFFFu, 0xFFFFFFFFu);
    writeMasked(aerOutAck_, GPIO_DATA, 0xFFFFFFFFu, 0xFFFFFFFFu);
    if (verbose) std::cout << "  [1/5] GPIOs limpiados\n";

    spiSend40(addrGlobal(0), 1);  // GATE_ACTIVITY=1 (stop)
    if (verbose) std::cout << "  [2/5] Chip parado (GATE_ACTIVITY=1)\n";

    if (verbose) std::cout << "  [3/5] Reseteando 65536 sinapsis (tardara ~30s)...\n";
    resetAllSynapses();
    if (verbose) std::cout << "        Sinapsis limpias.\n";

    if (verbose) std::cout << "  [4/5] Reseteando 256 neuronas (16 bytes c/u)...\n";
    resetAllNeurons();
    if (verbose) std::cout << "        Neuronas en estado seguro (thr=255, v_mem=0)\n";

    for (int i = 1; i <= 25; ++i) spiSend40(addrGlobal(i), 0);
    synSignCache_.fill(0);
    if (verbose) std::cout << "  [5/5] Registros globales limpiados\n";

    if (verbose)
        std::cout << "\nODIN listo. Tiempo total: "
                  << elapsedUs(t0) / 1e6 << "s\n";
}

void OdinDriver::stop()  { spiSend40(addrGlobal(0), 1); }
void OdinDriver::start() { spiSend40(addrGlobal(0), 0); }

void OdinDriver::resetAllNeurons() {
    for (int n = 0; n < ODIN_MAX_NEURONS; ++n)
        neuronLif(n, /*thr=*/255, /*leakStr=*/0, /*leakEn=*/0);
}

void OdinDriver::resetAllSynapses() {
    for (int pre = 0; pre < ODIN_MAX_NEURONS; ++pre)
        for (int post = 0; post < ODIN_MAX_NEURONS; ++post)
            synapse(pre, post, /*weight=*/0, /*mapped=*/0);
}

// ------------------------------------------------------------------
// Neurons (LIF) — bit layout confirmed against ODIN's official doc
// (section 3.3.2) and odin.py's neuron_lif().
// ------------------------------------------------------------------
void OdinDriver::neuronLif(int neuronId, int thr, int leakStr, int leakEn,
                           int caEn, int thetamem, int caTheta1, int caTheta2,
                           int caTheta3, int caLeak) {
    std::array<int, 128> bits{};
    auto setb = [&](int pos, int val, int w) {
        for (int i = 0; i < w; ++i) bits[pos + i] = (val >> i) & 1;
    };
    setb(0, 1, 1);              // lif_izh_sel = 1 (LIF)
    setb(1, leakStr, 7);
    setb(8, leakEn, 1);
    setb(9, thr, 8);
    setb(17, caEn, 1);
    setb(18, thetamem, 8);
    setb(26, caTheta1, 3);
    setb(29, caTheta2, 3);
    setb(32, caTheta3, 3);
    setb(35, caLeak, 5);

    for (int byteIdx = 0; byteIdx < 16; ++byteIdx) {
        int val = 0;
        for (int b = 0; b < 8; ++b) val |= (bits[byteIdx * 8 + b] & 1) << b;
        spiSend40(addrNeuronByte(neuronId, byteIdx), static_cast<std::uint32_t>(val));
    }
}

// ------------------------------------------------------------------
// Synapses
// ------------------------------------------------------------------
void OdinDriver::synapse(int pre, int post, int weight, int mapped) {
    if (weight < 0 || weight > 7)
        throw std::invalid_argument("OdinDriver::synapse: weight must be in [0,7]");
    int nibble = ((mapped & 1) << 3) | (weight & 0x7);
    int dataByte, mask;
    if (post & 1) { dataByte = nibble << 4; mask = 0x0F; }
    else          { dataByte = nibble;      mask = 0xF0; }
    std::uint32_t data20 = (static_cast<std::uint32_t>(mask) << 8) |
                           static_cast<std::uint32_t>(dataByte);
    spiSend40(addrSynapse(pre, post), data20);
}

void OdinDriver::setSynSign(int neuronId, bool inhibitory) {
    int wordIdx = neuronId >> 4;
    int bitIdx  = neuronId & 0xF;
    if (inhibitory) synSignCache_[wordIdx] |= (1u << bitIdx);
    else            synSignCache_[wordIdx] &= ~(1u << bitIdx);
    spiSend40(addrGlobal(2 + wordIdx), synSignCache_[wordIdx]);
}

// ------------------------------------------------------------------
// High-level: load an exported WANN network config
// ------------------------------------------------------------------
void OdinDriver::loadConfig(const wann::OdinNetworkConfig& cfg, bool verbose) {
    init(verbose);
    for (const auto& n : cfg.neurons) {
        // Default LIF params — see OdinExport.h/OdinDriver.h: Izhikevich
        // mode exists in the RTL but isn't wired up in this driver yet, so
        // every neuron currently runs as a generic LIF spiker regardless of
        // which of the 20 Izhikevich behaviours WANN evolved for it.
        neuronLif(n.addr, /*thr=*/14, /*leakStr=*/10, /*leakEn=*/1);
    }

    // Sign is per source neuron (Dale's law) — buildOdinConfig already
    // rejects genomes where one node's outgoing connections mix signs, so
    // every synapse sharing a src here is guaranteed consistent.
    for (const auto& s : cfg.synapses) {
        synapse(s.src, s.dst, s.weight3bit, /*mapped=*/1);
        setSynSign(s.src, /*inhibitory=*/!s.excitatory);
    }
}

// ------------------------------------------------------------------
// AER IN — 4-phase handshake (odin.py's _aer_send_raw)
// ------------------------------------------------------------------
std::uint32_t OdinDriver::aer17(int bit16, int hi8, int lo8) {
    return (static_cast<std::uint32_t>(bit16 & 1) << 16) |
           ((static_cast<std::uint32_t>(hi8) & 0xFF) << 8) |
           (static_cast<std::uint32_t>(lo8) & 0xFF);
}

int OdinDriver::aerSendRaw(std::uint32_t addr17, int timeoutUs) {
    auto deadline = [&] { return Clock::now() + std::chrono::microseconds(timeoutUs); };

    auto dl = deadline();
    while (aerInAck_.read(GPIO_DATA) & 0x1)
        if (Clock::now() > dl) return -1;

    writeMasked(aerIn_, GPIO_DATA, addr17 & 0x1FFFF, 0x1FFFF);
    writeMasked(aerIn_, GPIO2_DATA, 0x1, 0x1);

    dl = deadline();
    while (!(aerInAck_.read(GPIO_DATA) & 0x1)) {
        if (Clock::now() > dl) {
            writeMasked(aerIn_, GPIO2_DATA, 0x0, 0x1);
            return -2;
        }
    }
    writeMasked(aerIn_, GPIO2_DATA, 0x0, 0x1);

    dl = deadline();
    while (aerInAck_.read(GPIO_DATA) & 0x1)
        if (Clock::now() > dl) return -3;

    return 0;
}

int OdinDriver::sendVirtual(int neuronId, int weight, int leakBit, int inhibBit, int timeoutUs) {
    if (weight < 0 || weight > 7)
        throw std::invalid_argument("OdinDriver::sendVirtual: weight must be in [0,7]");
    int lo = ((weight & 0x7) << 5) | ((inhibBit & 1) << 4) | ((leakBit & 1) << 3) | 0x01;
    return aerSendRaw(aer17(0, neuronId, lo), timeoutUs);
}

// ------------------------------------------------------------------
// AER OUT drain
// ------------------------------------------------------------------
std::vector<int> OdinDriver::drainSpikes(int timeoutUs) {
    std::vector<int> fired;
    auto deadline = Clock::now() + std::chrono::microseconds(timeoutUs);
    while (Clock::now() < deadline) {
        if (aerOut_.read(GPIO2_DATA) & 0x1) {
            int addr = static_cast<int>(aerOut_.read(GPIO_DATA) & 0xFF);
            fired.push_back(addr);
            writeMasked(aerOutAck_, GPIO_DATA, 0x1, 0x1);
            int w = 0;
            while ((aerOut_.read(GPIO2_DATA) & 0x1) && w < 10000) ++w;
            writeMasked(aerOutAck_, GPIO_DATA, 0x0, 0x1);
        }
    }
    return fired;
}

void OdinDriver::aerBusReset() {
    writeMasked(aerIn_, GPIO_DATA, 0, 0xFFFFFFFFu);
    writeMasked(aerIn_, GPIO2_DATA, 0, 0xFFFFFFFFu);
    writeMasked(aerOutAck_, GPIO_DATA, 0, 0xFFFFFFFFu);
    drainSpikes(50'000);
}

} // namespace wann::hw

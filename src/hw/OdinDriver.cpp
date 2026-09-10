#include "../../include/wann/hw/OdinDriver.h"
#include "../../include/wann/hw/OdinRegisters.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>
#include <thread>

namespace wann::hw {

OdinDriver::OdinDriver() {
    if (!ODIN_REGISTERS_CONFIGURED) {
        throw std::logic_error(
            "OdinDriver: include/wann/hw/OdinRegisters.h still has placeholder "
            "(0x0) register offsets. Fill in the real AXI-Lite map from your "
            "Vivado block design / overlay .hwh, set ODIN_REGISTERS_CONFIGURED "
            "= true, then rebuild.");
    }

    memFd_ = open("/dev/mem", O_RDWR | O_SYNC);
    if (memFd_ < 0)
        throw std::runtime_error("OdinDriver: cannot open /dev/mem (run as root, "
                                  "or use the UIO device your overlay exposes instead)");

    mmapBase_ = mmap(nullptr, ODIN_AXI_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      memFd_, static_cast<off_t>(ODIN_AXI_BASE_ADDR));
    if (mmapBase_ == MAP_FAILED) {
        close(memFd_);
        throw std::runtime_error("OdinDriver: mmap of ODIN AXI-Lite window failed");
    }
}

OdinDriver::~OdinDriver() {
    if (mmapBase_ && mmapBase_ != MAP_FAILED) munmap(mmapBase_, ODIN_AXI_SPAN);
    if (memFd_ >= 0) close(memFd_);
}

std::uint32_t OdinDriver::readReg(std::uintptr_t offset) const {
    auto* p = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(mmapBase_) + offset);
    return *p;
}

void OdinDriver::writeReg(std::uintptr_t offset, std::uint32_t value) {
    auto* p = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(mmapBase_) + offset);
    *p = value;
}

void OdinDriver::loadConfig(const wann::OdinNetworkConfig& /*cfg*/) {
    // TODO: bit-bang the SPI-emulated config port (REG_CONFIG_SPI_DATA /
    // REG_CONFIG_SPI_CTRL) to shift in each neuron's 128-bit word and each
    // synapse's 3-bit weight. The exact shift order/framing (neuron-major
    // vs. synapse-major, MSB/LSB first, any start/stop framing bits) is
    // defined by ODIN's config scan chain and isn't in this repo yet.
    throw std::logic_error("OdinDriver::loadConfig: SPI config shift-in not "
                            "implemented — needs the real scan-chain protocol");
}

void OdinDriver::sendSpike(int /*addr*/) {
    // TODO: write REG_AER_IN_ADDR, toggle REG_AER_IN_REQ per REQ_ACTIVE_HIGH /
    // FOUR_PHASE_HANDSHAKE, and poll REG_AER_IN_ACK until the core accepts it.
    throw std::logic_error("OdinDriver::sendSpike: aer_in handshake not "
                            "implemented — needs req/ack polarity confirmed");
}

std::vector<int> OdinDriver::pollOutputs(int /*timeoutUs*/) {
    // TODO: poll REG_AER_OUT_REQ, latch REG_AER_OUT_ADDR, pulse REG_AER_OUT_ACK
    // per FOUR_PHASE_HANDSHAKE, repeat until `timeoutUs` of silence.
    throw std::logic_error("OdinDriver::pollOutputs: aer_out handshake not "
                            "implemented — needs req/ack polarity confirmed");
}

} // namespace wann::hw

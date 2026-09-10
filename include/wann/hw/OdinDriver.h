#pragma once
#include "../OdinExport.h"

#include <cstdint>
#include <vector>

namespace wann::hw {

// Thin AXI-Lite driver for the ODIN neuromorphic core on the ZCU104 PL
// fabric, talked to from the PS over /dev/mem (mmap). Only built when
// -DWANN_ODIN_HW=ON, since it's Linux/ARM-specific and only meaningful on
// the board itself.
//
// The mmap/register plumbing here is standard Zynq/PYNQ practice and is
// implemented for real. What's NOT implemented yet is the ODIN-specific
// protocol on top of it (the SPI-emulated config shift sequence, and the
// exact aer_in/aer_out req/ack handshake) — those depend on register
// offsets and polarity this repo doesn't know (see OdinRegisters.h). Every
// method that needs them throws std::logic_error until
// wann::hw::ODIN_REGISTERS_CONFIGURED is true and the TODOs in
// OdinDriver.cpp are resolved against the real block design.
class OdinDriver {
public:
    OdinDriver();
    ~OdinDriver();

    OdinDriver(const OdinDriver&) = delete;
    OdinDriver& operator=(const OdinDriver&) = delete;

    // Shifts the neuron parameter words + synapse weight table from `cfg`
    // (as produced by wann::buildOdinConfig / wann_car_odin_export) into
    // ODIN over the SPI-emulated config port.
    void loadConfig(const wann::OdinNetworkConfig& cfg);

    // Drives one AER_IN request for input neuron address `addr` and blocks
    // for the req/ack handshake to complete.
    void sendSpike(int addr);

    // Drains AER_OUT for up to `timeoutUs` microseconds of silence,
    // returning the addresses of every neuron that fired.
    std::vector<int> pollOutputs(int timeoutUs);

private:
    void*  mmapBase_ = nullptr;
    int    memFd_    = -1;

    std::uint32_t readReg(std::uintptr_t offset) const;
    void          writeReg(std::uintptr_t offset, std::uint32_t value);
};

} // namespace wann::hw

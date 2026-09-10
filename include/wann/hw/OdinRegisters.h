#pragma once
#include <cstdint>

// -----------------------------------------------------------------------
// PENDING: fill these in from your Vivado block design / PYNQ overlay
// before OdinDriver.cpp can talk to real hardware. Nothing in this repo
// can derive them — they depend entirely on how ODIN's AXI-Lite bridge,
// SPI-emulated config port, and aer_in/aer_out req/ack lines were wired
// into your block design (register offsets, address widths, and whether
// req/ack are active-high or active-low, 2-phase or 4-phase handshake).
//
// Until this is filled in, OdinDriver refuses to run (see OdinDriver.cpp).
// -----------------------------------------------------------------------
namespace wann::hw {

// Physical (or /dev/mem) base address of the ODIN AXI-Lite block, as seen
// from the PS. Get this from the overlay's .hwh / `Overlay().ip_dict`.
inline constexpr std::uintptr_t ODIN_AXI_BASE_ADDR = 0x0;  // TODO
inline constexpr std::size_t    ODIN_AXI_SPAN      = 0x0;  // TODO (bytes)

// Byte offsets within the AXI-Lite window. TODO: replace with the real
// offsets from your block design.
inline constexpr std::uintptr_t REG_CONFIG_SPI_DATA = 0x0;  // TODO
inline constexpr std::uintptr_t REG_CONFIG_SPI_CTRL = 0x0;  // TODO

inline constexpr std::uintptr_t REG_AER_IN_ADDR = 0x0;  // TODO
inline constexpr std::uintptr_t REG_AER_IN_REQ  = 0x0;  // TODO
inline constexpr std::uintptr_t REG_AER_IN_ACK  = 0x0;  // TODO

inline constexpr std::uintptr_t REG_AER_OUT_ADDR = 0x0;  // TODO
inline constexpr std::uintptr_t REG_AER_OUT_REQ  = 0x0;  // TODO
inline constexpr std::uintptr_t REG_AER_OUT_ACK  = 0x0;  // TODO

// req/ack polarity + handshake protocol — confirm against your design
// before trusting OdinDriver's handshake loop.
inline constexpr bool REQ_ACTIVE_HIGH = true;   // TODO confirm
inline constexpr bool FOUR_PHASE_HANDSHAKE = true;  // TODO confirm (vs. 2-phase)

// Set to true once every value above has been replaced with real numbers.
inline constexpr bool ODIN_REGISTERS_CONFIGURED = false;

} // namespace wann::hw

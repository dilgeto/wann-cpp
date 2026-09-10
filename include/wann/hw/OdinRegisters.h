#pragma once
#include <cstddef>
#include <cstdint>

// -----------------------------------------------------------------------
// Confirmed against:
//   - ODIN's official doc (github.com/ChFrenkel/ODIN, doc/README.md
//     sections 2.1/2.2/3.1/3.2) — SPI 20-bit addressing, AER 17-bit input
//     event encoding, neuron/synapse memory layout.
//   - The user's own working PYNQ driver
//     (jupyter_notebooks/neat_comun_ranc_odin/odin.py on the ZCU104),
//     which this repo's OdinDriver is a direct C++ port of.
//
// PENDING: the five physical (mmap) base addresses below are specific to
// this ZCU104 overlay's Vivado address map and can't be derived from
// anything in this repo — get them from the board with:
//   from pynq import Overlay
//   ov = Overlay('odin.bit')
//   print(ov.ip_dict['axi_gpio_0']['phys_addr'])   # ... _1, _2, _3, axi_quad_spi_0
// Fill in the five ODIN_*_BASE_ADDR constants below, set
// ODIN_REGISTERS_CONFIGURED = true, and rebuild.
// -----------------------------------------------------------------------
namespace wann::hw {

// AXI-Lite window size mmap'd per IP (standard 4K page covers every offset
// used below for both AXI GPIO and AXI Quad SPI).
inline constexpr std::size_t ODIN_IP_WINDOW = 0x1000;

// TODO: fill in from ov.ip_dict on the board (see comment above).
inline constexpr std::uintptr_t ODIN_AER_IN_BASE_ADDR      = 0x0;  // axi_gpio_0
inline constexpr std::uintptr_t ODIN_AER_IN_ACK_BASE_ADDR  = 0x0;  // axi_gpio_1
inline constexpr std::uintptr_t ODIN_AER_OUT_BASE_ADDR     = 0x0;  // axi_gpio_2
inline constexpr std::uintptr_t ODIN_AER_OUT_ACK_BASE_ADDR = 0x0;  // axi_gpio_3
inline constexpr std::uintptr_t ODIN_SPI_BASE_ADDR         = 0x0;  // axi_quad_spi_0

// Set to true once every *_BASE_ADDR above has been replaced with the real
// physical address from ov.ip_dict.
inline constexpr bool ODIN_REGISTERS_CONFIGURED = false;

// ------------------------------------------------------------------
// AXI GPIO (Xilinx LogiCORE PG144) register offsets — standard, fixed.
// Dual-channel core: channel 1 is the primary data/direction pair,
// channel 2 the secondary one (used here for REQ/ACK alongside data).
// ------------------------------------------------------------------
inline constexpr std::uintptr_t GPIO_DATA  = 0x00;
inline constexpr std::uintptr_t GPIO_TRI   = 0x04;  // 1=input, 0=output per bit
inline constexpr std::uintptr_t GPIO2_DATA = 0x08;
inline constexpr std::uintptr_t GPIO2_TRI  = 0x0C;

// ------------------------------------------------------------------
// AXI Quad SPI (Xilinx LogiCORE PG153) register offsets + SPICR/SPISR
// bits — standard, fixed. Mirrors odin.py's QuadSpiMaster exactly.
// ------------------------------------------------------------------
inline constexpr std::uintptr_t QSPI_SRR        = 0x40;
inline constexpr std::uintptr_t QSPI_SPICR      = 0x60;
inline constexpr std::uintptr_t QSPI_SPISR      = 0x64;
inline constexpr std::uintptr_t QSPI_SPIDTR     = 0x68;
inline constexpr std::uintptr_t QSPI_SPIDRR     = 0x6C;
inline constexpr std::uintptr_t QSPI_SPISSR     = 0x70;
inline constexpr std::uintptr_t QSPI_TXFIFO_OCY = 0x74;
inline constexpr std::uintptr_t QSPI_RXFIFO_OCY = 0x78;

inline constexpr std::uint32_t SPICR_LOOP          = (1u << 0);
inline constexpr std::uint32_t SPICR_SPE           = (1u << 1);
inline constexpr std::uint32_t SPICR_MASTER        = (1u << 2);
inline constexpr std::uint32_t SPICR_CPOL          = (1u << 3);
inline constexpr std::uint32_t SPICR_CPHA          = (1u << 4);
inline constexpr std::uint32_t SPICR_TXFIFO_RST    = (1u << 5);
inline constexpr std::uint32_t SPICR_RXFIFO_RST    = (1u << 6);
inline constexpr std::uint32_t SPICR_MANUAL_SS     = (1u << 7);
inline constexpr std::uint32_t SPICR_TRANS_INHIBIT = (1u << 8);
inline constexpr std::uint32_t SPICR_LSB_FIRST     = (1u << 9);

inline constexpr std::uint32_t SPISR_RX_EMPTY = (1u << 0);
inline constexpr std::uint32_t SPISR_RX_FULL  = (1u << 1);
inline constexpr std::uint32_t SPISR_TX_EMPTY = (1u << 2);
inline constexpr std::uint32_t SPISR_TX_FULL  = (1u << 3);
inline constexpr std::uint32_t SPISR_MODF     = (1u << 4);

} // namespace wann::hw

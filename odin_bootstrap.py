#!/usr/bin/env python3
"""odin_bootstrap.py - one-time PYNQ setup before running wann_car_odin_eval.

OdinDriver (include/wann/hw/OdinDriver.h) talks to the ODIN core directly
over /dev/mem via mmap, bypassing PYNQ/Python entirely for performance and
simplicity — but it deliberately does NOT reimplement bitstream download or
AXI GPIO direction configuration, since PYNQ's Overlay() already does both
correctly for this board (clock/reset sequencing, device tree overlay, and
setting each axi_gpio_N channel's direction from the block design's IP
config) and re-deriving that from scratch in C++ would just be a second,
easier-to-get-wrong copy of working infrastructure.

Run this once (or again after any bitstream reload / power cycle) before
wann_car_odin_eval — it downloads the overlay and touches every IP the C++
driver will later mmap directly, which is what makes PYNQ configure their
GPIO directions:

    python3 odin_bootstrap.py /path/to/odin.bit

The bitstream stays loaded and the GPIO/SPI register state persists in the
PL fabric after this process exits — the C++ binary run afterwards (as a
separate process, possibly much later) sees the same configured hardware.
"""

import sys

from pynq import Overlay


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} /ruta/a/odin.bit", file=sys.stderr)
        return 1

    bitfile = sys.argv[1]
    print(f"Cargando overlay: {bitfile}")
    ov = Overlay(bitfile)

    # Touching each IP is what makes PYNQ configure its GPIO tri-state
    # (direction) registers based on the block design's IP config — the
    # values themselves aren't used here, just the side effect.
    for ip_name in ("axi_gpio_0", "axi_gpio_1", "axi_gpio_2", "axi_gpio_3",
                     "axi_quad_spi_0"):
        ip = getattr(ov, ip_name)
        phys_addr = ov.ip_dict[ip_name]["phys_addr"]
        print(f"  {ip_name}: phys_addr=0x{phys_addr:x}  (ok)")

    print("\nODIN overlay cargado y GPIOs configurados.")
    print("Copia los phys_addr de arriba a include/wann/hw/OdinRegisters.h "
          "(ODIN_AER_IN_BASE_ADDR, etc.) si aun no lo hiciste, y deja el "
          "chip programado corriendo wann_car_odin_eval en otra terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

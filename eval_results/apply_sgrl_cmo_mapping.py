#!/usr/bin/env python3
"""Agrega `--reramcmo-mapping legacy|pair|single` a ../surrogate-gradient-rl/sgrl/noise.py (el lado SG de
`eval_results/run_paper_noise_cmo_mapping.sh`). Reemplaza al antiguo sgrl_cmo_mapping.patch: en vez de un diff (que
`git apply` rechaza si el archivo difiere un poco, p. ej. con o sin sgrl_occupancy.patch) edita por anclas de texto.

  python eval_results/apply_sgrl_cmo_mapping.py [ruta/al/repo/surrogate-gradient-rl] [--check] [--revert-info]

* Funciona con o sin sgrl_occupancy.patch aplicado (las anclas no tocan sus líneas).
* Idempotente y por pasos: paso 1 = mapeo pair/single; paso 2 = réplica exacta de aihwkit en pair/single (ventana de aceptación
  del ruido de programación y sin recorte de g_prog). Aplica solo lo que falta, así que sirve también si ya corriste la versión
  anterior del script (solo paso 1).
* Todo o nada: si falta o se repite alguna ancla no escribe nada y dice cuál.
* --check solo informa el estado (no escribe).
No cambia el comportamiento por defecto (mapping = "legacy").
"""
import argparse
import sys
from pathlib import Path

EDITS = [
    # (nombre, ancla única, reemplazo)
    ("ReRamCMOModel.mapping",
     """    n_levels: int = 0

    def program(self, g_t, gen):
        coeff = self.prog_coeff.get(""",
     """    n_levels: int = 0
    # Mapeo peso -> conductancia (mismo significado que CmoMapping de memristor_noise.hpp / --reramcmo-mapping):
    #   legacy  par diferencial, rama inactiva a 0 y lectura recortada a >= g_min por rama (histórico: aun con ruido 0 anula
    #           los pesos con g < g_min y achica el resto)
    #   pair    par diferencial con g_min en ambas ramas: el piso se cancela (como SinglePairConductanceConverter de aihwkit)
    #   single  un dispositivo por peso, afín (min de la matriz -> g_min, max -> g_max; SingleDeviceConductanceConverter,
    #           el de ReRamCMONoiseModel en aihwkit). La ocupación (max|w|/w_max) centra y recorta el rango usado.
    mapping: str = "legacy"

    @property
    def g_offset(self) -> float:  # conductancia de reposo de cada rama del par (solo mapping == "pair")
        return self.g_min if self.mapping == "pair" else 0.0

    def program(self, g_t, gen):
        coeff = self.prog_coeff.get("""),
    ("NoiseConfig.reramcmo_mapping",
     """    reramcmo_g_min: float = 1.0

    def scales(self):""",
     """    reramcmo_g_min: float = 1.0
    reramcmo_mapping: str = "legacy"  # legacy | pair | single (ver ReRamCMOModel)

    def scales(self):"""),
    ("build_model",
     """read_noise_scale=r, drift_scale=d, n_levels=cfg.n_levels)
        if not (0.0 <= m.g_min < m.g_max):""",
     """read_noise_scale=r, drift_scale=d, n_levels=cfg.n_levels,
                          mapping=cfg.reramcmo_mapping)
        if cfg.reramcmo_mapping not in ("legacy", "pair", "single"):
            raise ValueError("reramcmo_mapping debe ser legacy, pair o single")
        if not (0.0 <= m.g_min < m.g_max):"""),
    ("add_noise_args",
     """    g.add_argument(f"--{prefix}reramcmo-g-min", type=float, default=d.reramcmo_g_min)
""",
     """    g.add_argument(f"--{prefix}reramcmo-g-min", type=float, default=d.reramcmo_g_min)
    g.add_argument(f"--{prefix}reramcmo-mapping", choices=("legacy", "pair", "single"), default=d.reramcmo_mapping,
                   help="ReRAM-CMO peso->conductancia: legacy (hist., el piso g_min deforma los pesos), pair (par diferencial "
                        "con g_min en ambas ramas) o single (un dispositivo afín, como aihwkit)")
"""),
    ("Crossbar.__init__",
     """        g = (weight.abs() / self.w_max).clamp(0.0, 1.0) * model.g_max
        zero = torch.zeros_like(g)
        self.g_pos, self.g_neg = torch.where(weight > 0, g, zero), torch.where(weight < 0, g, zero)
        self.state = None""",
     """        self.single = getattr(model, "mapping", "legacy") == "single"
        if self.single:
            # un dispositivo por peso: afín en [min(W), max(W)] de la matriz (aihwkit lo hace por tile)
            lo, hi, amax = float(weight.min()), float(weight.max()), float(weight.abs().max())
            if not hi - lo > 1e-12:  # un solo valor: aihwkit dividiría por 0; rango simétrico +-|w|
                m = amax if amax > 0 else 1.0
                lo, hi = -m, m
            occ = min(1.0, max(0.0, amax / self.w_max)) if amax > 0 else 1.0
            total = model.g_max - model.g_min
            self.w_lo, self.g_lo, self.sc = lo, model.g_min + 0.5 * (1.0 - occ) * total, occ * total / (hi - lo)
            self.g_pos, self.g_neg = self.g_lo + (weight - lo) * self.sc, None
        else:
            # offset = g_min en "pair" (rama inactiva en g_min, rango útil g_max - g_min); en "legacy" 0 y g_max
            self.offset = float(getattr(model, "g_offset", 0.0))
            self.span = model.g_max - self.offset
            g = (weight.abs() / self.w_max).clamp(0.0, 1.0) * self.span
            zero = torch.zeros_like(g)
            self.g_pos = torch.where(weight > 0, g, zero) + self.offset
            self.g_neg = torch.where(weight < 0, g, zero) + self.offset
        self.state = None"""),
    ("Crossbar.program",
     """        self.state = (self.model.program(self.g_pos, gen), self.model.program(self.g_neg, gen))""",
     """        self.state = (self.model.program(self.g_pos, gen),
                      None if self.g_neg is None else self.model.program(self.g_neg, gen))"""),
    ("Crossbar.read",
     """        gp, gn = (self.model.read(s, t, gen) for s in self.state)
        return (gp - gn) * (self.w_max / self.model.g_max)""",
     """        if self.single:
            return self.w_lo + (self.model.read(self.state[0], t, gen) - self.g_lo) / self.sc
        gp, gn = (self.model.read(s, t, gen) for s in self.state)
        return (gp - gn) * (self.w_max / self.span)"""),
]

# Paso 2 (réplica exacta de aihwkit con los mapeos pair/single; "legacy" queda bit a bit como antes). Las líneas que toca
# existen igual con o sin el paso 1, así que se aplica sobre cualquiera de los dos estados.
EDITS_V2 = [
    ("ReRamCMOModel.program: ventana de aceptación",
     """        coeff = self.prog_coeff.get(self.acceptance_range, self.prog_coeff[min(self.prog_coeff)])
""",
     """        if self.mapping == "legacy":
            coeff = self.prog_coeff.get(self.acceptance_range, self.prog_coeff[min(self.prog_coeff)])
        else:  # aihwkit: claves en PORCENTAJE (0.2, 2); el 2e-2 por defecto no es clave -> cae a min() = ventana del 0.2 %
            coeff = next((c for k, c in self.prog_coeff.items() if abs(k * 100.0 - self.acceptance_range) < 1e-9),
                         self.prog_coeff[min(self.prog_coeff)])
"""),
    ("ReRamCMOModel.program: recorte de g_prog",
     """        g = (g_t + self.prog_noise_scale * sig * _randn(g_t, gen)).clamp(0.0, self.g_max)
        return {"g": _quantize(g, self.g_max, self.n_levels)}""",
     """        g = g_t + self.prog_noise_scale * sig * _randn(g_t, gen)
        if self.mapping == "legacy" or self.n_levels > 1:  # aihwkit no recorta g_prog (solo la lectura, a [g_min, g_max])
            g = g.clamp(0.0, self.g_max)
        return {"g": _quantize(g, self.g_max, self.n_levels)}"""),
]
V2_MARK = "aihwkit: claves en PORCENTAJE"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", default=str(Path(__file__).resolve().parent.parent.parent / "surrogate-gradient-rl"))
    ap.add_argument("--check", action="store_true", help="solo informar el estado")
    a = ap.parse_args()
    path = Path(a.repo) / "sgrl" / "noise.py"
    if not path.is_file():
        sys.exit(f"No existe {path}")
    s = path.read_text(encoding="utf-8")
    occ = "occupancy" in s
    print(f"{path}\n  sgrl_occupancy.patch: {'aplicado' if occ else 'NO aplicado'} (necesario para run_paper_noise_*.sh, no para este script)")
    v1 = "reramcmo_mapping" in s and "g_offset" in s and "self.single" in s
    v2 = V2_MARK in s
    if not v1 and ("reramcmo_mapping" in s or "self.single" in s):
        sys.exit("  mapeo CMO: aplicado A MEDIAS (hay restos pero no todo). Restaurá el archivo (git checkout sgrl/noise.py) "
                 "y vuelve a correr.")
    if v2 and not v1:
        sys.exit("  paso 2 presente sin el paso 1: restaurá el archivo (git checkout sgrl/noise.py) y vuelve a correr.")
    todo = ([] if v1 else EDITS) + ([] if v2 else EDITS_V2)
    print(f"  paso 1 (mapeo pair/single): {'ya aplicado' if v1 else 'pendiente'};  "
          f"paso 2 (réplica exacta de aihwkit en pair/single): {'ya aplicado' if v2 else 'pendiente'}")
    if not todo:
        print("  nada que hacer")
        return
    bad = [(n, s.count(anchor)) for n, anchor, _ in todo if s.count(anchor) != 1]
    if bad:
        for n, c in bad:
            print(f"  ancla '{n}': {c} coincidencias (se esperaba 1)")
        sys.exit("  No se escribió nada: este noise.py difiere de la versión esperada. Pasame `git log -1 --oneline` y "
                 "`git diff --stat` de ese repo.")
    print("  todas las anclas encontradas")
    if a.check:
        return
    for _, anchor, new in todo:
        s = s.replace(anchor, new)
    path.write_text(s, encoding="utf-8")
    print(f"  aplicado ({len(todo)} ediciones). Revisá con: git -C {a.repo} diff --stat")


if __name__ == "__main__":
    main()

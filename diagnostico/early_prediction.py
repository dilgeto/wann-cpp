#!/usr/bin/env python3
"""¿Cuánto predice lo temprano el resultado final? (dependencia de la semilla)

Para cada generación g de una lista, correlaciona entre corridas (Spearman, IC 95%
bootstrap sobre corridas) un predictor medido en la gen g con el nivel final
(media del fitness del élite en las últimas 20 gens):

  elite_fit    fitness del élite en la gen g (ruidoso: una sola evaluación)
  running_best récord acumulado hasta g
  top5_mean    media del 5% mejor de la población en g (menos sensible a una
               evaluación con suerte)
  n_conn       conexiones del élite en g

Si la correlación ya es alta con g pequeño, el destino de la corrida se decide
pronto (hipótesis de dependencia de las mutaciones iniciales). Una correlación
que crece despacio con g dice lo contrario: el resultado se construye durante la
evolución. Todo son correlaciones ENTRE corridas de una misma configuración; con
pocas corridas los intervalos serán anchos (se avisa si hay menos de 8).

Uso:
  python diagnostico/early_prediction.py --glob 'log/diag_rank00/seed*_lineage.csv' \\
      [--out diag_results/prediction] [--gens 5,10,25,50,100,200,300] [--outcome final_level|runbest]
Salidas: <out>_prediction.csv y <out>_prediction.png (si hay matplotlib).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diag_lib as dl

dl.anchor_to_repo_root()

PREDICTORS = ["elite_fit", "running_best", "top5_mean", "n_conn"]


def features_at(lin: dl.Lineage, g: int) -> dict[str, float]:
    if g > lin.last_gen:
        return {p: float("nan") for p in PREDICTORS}
    k = max(1, int(round(0.05 * lin.pop)))
    top = np.sort(lin.fitness[g])[-k:]
    return {
        "elite_fit":    float(lin.elite_fit()[g]),
        "running_best": float(lin.running_best()[g]),
        "top5_mean":    float(top.mean()),
        "n_conn":       float(lin.n_conn[g, lin.elite_idx()[g]]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", required=True, help="patrón de *_lineage.csv")
    ap.add_argument("--out", default="diag_results/prediction", help="prefijo de salida")
    ap.add_argument("--gens", default="5,10,25,50,100,200,300")
    ap.add_argument("--outcome", choices=["final_level", "runbest"], default="final_level",
                    help="variable a predecir (default: media del élite en las últimas 20 gens)")
    ap.add_argument("--boot", type=int, default=2000, help="remuestreos bootstrap")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    lins = []
    for path in files:
        try:
            lins.append(dl.load_lineage(path))
        except Exception as exc:                      # noqa: BLE001
            print(f"  (omitido {path}: {exc})", file=sys.stderr)
    if len(lins) < 3:
        print(f"Se necesitan al menos 3 corridas utilizables (hay {len(lins)}).", file=sys.stderr)
        return 1
    if len(lins) < 8:
        print(f"AVISO: solo {len(lins)} corridas; los intervalos serán muy anchos.\n")

    outcome = np.array([l.final_level() if args.outcome == "final_level"
                        else float(l.running_best()[-1]) for l in lins])
    gens = [int(x) for x in args.gens.split(",") if x]

    rows = []
    for g in gens:
        feats = [features_at(l, g) for l in lins]
        for p in PREDICTORS:
            x = np.array([f[p] for f in feats])
            rho, lo, hi, n = dl.spearman_ci(x, outcome, n_boot=args.boot)
            rows.append({"gen": g, "predictor": p, "n_runs": n, "rho": rho, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(f"{out}_prediction.csv", index=False)

    print(f"{len(lins)} corridas; resultado = {args.outcome}; "
          f"nivel final: mediana {np.median(outcome):.1f}, rango [{outcome.min():.1f}, {outcome.max():.1f}]\n")
    print("Spearman entre corridas (predictor en la gen g vs resultado final), IC 95%:")
    header = f"{'gen':>5}  " + "  ".join(f"{p:<26}" for p in PREDICTORS)
    print(header)
    for g in gens:
        cells = []
        for p in PREDICTORS:
            r = res[(res["gen"] == g) & (res["predictor"] == p)].iloc[0]
            cells.append((f"{'sin datos (g > gens)':<26}" if int(r["n_runs"]) == 0 else f"{'n/d (constante)':<26}")
                         if not np.isfinite(r["rho"]) else
                         f"{r['rho']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] n={int(r['n_runs']):<3}")
        print(f"{g:>5}  " + "  ".join(cells))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for p in PREDICTORS:
            s = res[res["predictor"] == p].dropna(subset=["rho"])
            if s.empty:
                continue
            ax.plot(s["gen"], s["rho"], marker="o", label=p)
            ax.fill_between(s["gen"], s["ci_lo"], s["ci_hi"], alpha=0.12)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("generación g")
        ax.set_ylabel(f"Spearman con {args.outcome}")
        ax.set_title("¿Cuánto predice lo temprano el resultado final?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out}_prediction.png", dpi=150)
        print(f"\nEscrito: {out}_prediction.csv, {out}_prediction.png")
    except ImportError:
        print(f"\nEscrito: {out}_prediction.csv (sin matplotlib: no se generó el gráfico)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

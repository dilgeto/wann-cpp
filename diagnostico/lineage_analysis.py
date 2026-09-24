#!/usr/bin/env python3
"""Análisis de linaje de corridas de wann_car con snapshot_interval > 0.

Responde: ¿cuándo se "decide" una corrida? Como no hay crossover estructural, cada
individuo desciende de UN ancestro de la generación 0, y la diversidad estructural
solo puede perderse. Por corrida mide:

  * founders_at_g / gen_single_founder: cuántos ancestros distintos de la gen 0
    siguen vivos en la generación g, y cuándo queda uno solo (efecto fundador).
  * mrca_gen: última generación con un único ancestro de TODA la población final.
  * anc_pct_at_g: percentil de fitness, en la gen g, del ancestro del élite final.
    ~1.0 desde temprano = el ganador ya lideraba; ~0.5 = emergió después.
  * gen_last_improve / stagnation_len: como early_stop_patience (récord acumulado),
    y slope_last100 (¿el élite todavía sube?) con el fitness del élite suavizado.
  * por operador (early vs late): tasa de hijos que superan al élite anterior, que
    es el "descubrimiento realizado" para comparar con P(mejora) de N1.

Ojo con el ruido: cada generación evalúa con otra semilla y los élites se
re-evalúan, así que el récord acumulado incluye evaluaciones con suerte. Por eso
el nivel final (final_level) es la media del élite en las últimas 20 gens.
Con prob_crossover > 0 la ascendencia usa solo el padre más apto (parent).

Uso:
  python diagnostico/lineage_analysis.py --glob 'log/diag_rank00/seed*_lineage.csv' \\
      [--out diag_results/lineage] [--gens 10,25,50,100] [--early 50] [--late 100]
Salidas: <out>_runs.csv, <out>_curves.csv, <out>_ops.csv
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


def analyse_run(lin: dl.Lineage, gens: list[int]) -> tuple[dict, pd.DataFrame]:
    G = lin.last_gen
    P = lin.pop
    elite_i = lin.elite_idx()
    elite = lin.elite_fit()
    best = lin.running_best()
    f = dl.founders(lin)

    n_found = np.array([len(np.unique(f[g])) for g in range(lin.n_gen)])
    top_share = np.array([np.bincount(f[g], minlength=P).max() / P for g in range(lin.n_gen)])

    improved = np.flatnonzero(best[1:] > best[:-1]) + 1
    last_improve = int(improved[-1]) if len(improved) else 0

    single = np.flatnonzero(n_found == 1)
    le5 = np.flatnonzero(n_found <= 5)

    line = dl.ancestor_line(lin, G, int(elite_i[G]))
    row = {
        "run": lin.name,
        "gens": lin.n_gen,
        "pop": P,
        "final_level": lin.final_level(),
        "final_best_running": float(best[-1]),
        "gen_last_improve": last_improve,
        "stagnation_len": G - last_improve,
        "slope_last100": (float(elite[-20:].mean() - elite[-120:-100].mean())
                          if G >= 120 else float("nan")),
        "mrca_gen": dl.mrca_gen(lin),
        "gen_founders_le5": float(le5[0]) if len(le5) else float("nan"),
        "gen_single_founder": float(single[0]) if len(single) else float("nan"),
        "top_founder_share_final": float(top_share[-1]),
        "n_conn_final_elite": int(lin.n_conn[G, elite_i[G]]),
        "founder_gen0_fit_pct": dl.percentile_in_gen(lin, 0, int(line[0])),
    }
    for g in gens:
        row[f"founders_at_{g}"] = float(n_found[g]) if g <= G else float("nan")
        row[f"anc_pct_at_{g}"] = (dl.percentile_in_gen(lin, g, int(line[g]))
                                  if g <= G else float("nan"))

    curves = pd.DataFrame({
        "run": lin.name,
        "gen": np.arange(lin.n_gen),
        "n_founders": n_found,
        "top_founder_share": top_share,
        "elite_fit": elite,
        "running_best": best,
        "mean_fit": lin.fitness.mean(axis=1),
        "elite_n_conn": lin.n_conn[np.arange(lin.n_gen), elite_i],
    })
    return row, curves


def ops_table(lin: dl.Lineage, early: int, late: int) -> list[dict]:
    """Por operador y fase: n, tasa de aplicación, Δ medio hijo-padre y fracción de
    hijos que superan al élite de la generación anterior."""
    G = lin.last_gen
    elite = lin.elite_fit()
    phases = {"early": range(1, min(early, G) + 1),
              "late":  range(max(1, G - late + 1), G + 1)}
    rows = []
    for phase, gens in phases.items():
        gens = list(gens)
        if not gens:
            continue
        for op, name in enumerate(dl.OP_NAMES):
            n = applied = beats = 0
            deltas: list[np.ndarray] = []
            for g in gens:
                m = lin.op[g] == op
                if not m.any():
                    continue
                child = lin.fitness[g][m]
                deltas.append(child - lin.fitness[g - 1][lin.parent[g][m]])
                n += int(m.sum())
                applied += int(lin.applied[g][m].sum())
                beats += int((child > elite[g - 1]).sum())
            if n == 0:
                continue
            d = np.concatenate(deltas)
            rows.append({"run": lin.name, "phase": phase, "op": name, "n": n,
                         "applied_rate": applied / n, "mean_delta": float(d.mean()),
                         "frac_delta_pos": float((d > 0).mean()),
                         "frac_beats_prev_elite": beats / n})
    return rows


def fmt_ci(rho, lo, hi, n) -> str:
    if not np.isfinite(rho):
        return f"n/d: constante o n<3 (n={n})"
    return f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={n})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", required=True, help="patrón de *_lineage.csv")
    ap.add_argument("--out", default="diag_results/lineage", help="prefijo de salida")
    ap.add_argument("--gens", default="10,25,50,100", help="generaciones a reportar")
    ap.add_argument("--early", type=int, default=50, help="gens de la fase 'early'")
    ap.add_argument("--late", type=int, default=100, help="gens de la fase 'late'")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"Sin archivos para {args.glob!r}", file=sys.stderr)
        return 1
    gens = [int(x) for x in args.gens.split(",") if x]

    rows, curves, ops = [], [], []
    for path in files:
        try:
            lin = dl.load_lineage(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"  (omitido {path}: {exc})", file=sys.stderr)
            continue
        r, c = analyse_run(lin, gens)
        rows.append(r)
        curves.append(c)
        ops.extend(ops_table(lin, args.early, args.late))

    if not rows:
        print("Ninguna corrida utilizable.", file=sys.stderr)
        return 1
    runs = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(f"{out}_runs.csv", index=False)
    pd.concat(curves).to_csv(f"{out}_curves.csv", index=False)
    ops_df = pd.DataFrame(ops)
    ops_df.to_csv(f"{out}_ops.csv", index=False)

    print(f"{len(runs)} corridas (pop {int(runs['pop'].iloc[0])}, "
          f"{int(runs['gens'].min())}-{int(runs['gens'].max())} gens)\n")
    keys = ["final_level", "stagnation_len", "slope_last100", "mrca_gen",
            "gen_founders_le5", "gen_single_founder", "top_founder_share_final"]
    q = runs[keys].quantile([0.25, 0.5, 0.75]).T
    q.columns = ["p25", "mediana", "p75"]
    q["n_nan"] = runs[keys].isna().sum()
    print("Resumen entre corridas:")
    print(q.round(2).to_string(), "\n")

    pct = runs["founder_gen0_fit_pct"]
    print(f"Ancestro del élite final en el top-10% de fitness de la gen 0: "
          f"{(pct > 0.9).mean():.0%} de las corridas (azar puro ≈ 10%)\n")

    print("Spearman con final_level (IC 95% bootstrap sobre corridas):")
    predictors = ["mrca_gen", "gen_single_founder"] + [f"founders_at_{g}" for g in gens] \
                 + [f"anc_pct_at_{g}" for g in gens]
    for col in predictors:
        print(f"  {col:<24} {fmt_ci(*dl.spearman_ci(runs[col], runs['final_level']))}")

    if len(ops_df):
        late = ops_df[ops_df["phase"] == "late"]
        if len(late):
            tab = late.groupby("op")[["applied_rate", "frac_beats_prev_elite"]].median()
            print(f"\nFase 'late' (últimas {args.late} gens), mediana entre corridas:")
            print(tab.round(4).to_string())
    print(f"\nEscrito: {out}_runs.csv, {out}_curves.csv, {out}_ops.csv")
    if len(runs) < 8:
        print("AVISO: con menos de 8 corridas las correlaciones son solo orientativas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

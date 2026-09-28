#!/usr/bin/env python3
"""¿Las mejoras "confirmadas" de wann_car_neighborhood son reales o azar?

Con --eps bajo (p. ej. 0) cada padre entrega decenas de candidatos, y cada uno pasa
el criterio de confirmación (ventaja media > 2 errores estándar con V semillas
nuevas) con probabilidad ~4 % aunque no valga nada. Este script compara, por
generación, las confirmaciones observadas con las que daría el azar puro:

  pct_conf    % de candidatos confirmados
  pct_azar    % esperado si ningún candidato mejorara (cola de una t de Student
              con V-1 gl en t = 2)
  p_binom     P(>= confirmados observados | azar); pequeño = hay señal real
  d_cribado   Δ medio de los candidatos en el cribado (una semilla)
  d_valid     Δ medio con semillas nuevas. Si cae a ~0, era el ruido del cribado.
  pct_pos     % de candidatos con Δ validado > 0 (azar puro ≈ 50 %)
  bonf        confirmados que resisten Bonferroni dentro de su padre (α = 0.05 / nº
              de candidatos del padre) — la corrección que el binario no aplica
  bonf_prac   de esos, los que ganan al menos --min-effect (default 2 %) del fitness
              del padre: significativos Y con efecto que importa
  P_mediana   mediana entre padres de P(mutación produce un vecino confirmado)

CONTROL POSITIVO: las generaciones tempranas (el élite todavía sube) deberían
mostrar señal (p_binom pequeño, d_valid > 0, P_mediana alta). Si allí tampoco la
hay, la herramienta es insensible y no se puede concluir nada de las tardías.

Uso:
  python diagnostico/confirm_report.py --dirs log/diag_neighborhood_control log/diag_neighborhood_eps0 \\
      [--runs-csv diag_results/lineage_runs.csv --split-at 400] [--out diag_results/confirm]
Salidas: <out>_report.csv y <out>_report.png (si hay matplotlib).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diag_lib as dl

dl.anchor_to_repo_root()

NAME_RE = re.compile(r"^(?P<run>.+)_g(?P<gen>\d+)_confirm\.csv$")


def make_sf(df: int):
    """P(T > t) para una t de Student con `df` grados de libertad (numpy, sin scipy)."""
    x = np.linspace(-200.0, 200.0, 400001)
    pdf = (math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
           * (1.0 + x ** 2 / df) ** (-(df + 1) / 2))
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * (x[1] - x[0]))])
    return lambda t: 1.0 - np.interp(t, x, cdf)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) para X ~ Binomial(n, p), en escala logarítmica (n grande)."""
    if k <= 0:
        return 1.0
    if n <= 0 or p <= 0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def load(dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(candidatos, padres). Un *_confirm.csv por padre; puede estar vacío."""
    cands, parents = [], []
    for d in dirs:
        for f in sorted(Path(d).glob("*_confirm.csv")):
            m = NAME_RE.match(f.name)
            pf = f.with_name(f.name.replace("_confirm.csv", "_parents.csv"))
            if not m or not pf.exists():
                continue
            par = pd.read_csv(pf)
            par = par[par["step"] == 0].iloc[0]
            row = {"run": m.group("run"), "gen": int(m.group("gen")), "fit_mean": float(par["fit_mean"]),
                   "n_candidates": int(par.get("n_candidates", 0)),
                   "p_confirmed": float(par.get("p_confirmed", 0.0))}
            parents.append(row)
            c = pd.read_csv(f)
            if len(c):
                c["run"], c["gen"], c["fit_mean"] = row["run"], row["gen"], row["fit_mean"]
                cands.append(c)
    if not parents:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()), pd.DataFrame(parents)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dirs", nargs="+", required=True, help="carpetas con *_confirm.csv y *_parents.csv")
    ap.add_argument("--runs-csv", default="", help="*_runs.csv de lineage_analysis.py")
    ap.add_argument("--split-at", type=float, default=None, help="separa alto/bajo por final_level")
    ap.add_argument("--min-effect", type=float, default=0.02, help="efecto mínimo relevante, fracción del fitness")
    ap.add_argument("--out", default="diag_results/confirm")
    args = ap.parse_args()

    C, P = load(args.dirs)
    if P.empty:
        print("Sin resultados.", file=sys.stderr)
        return 1
    P["group"] = "todas"
    if args.runs_csv and args.split_at is not None:
        lv = pd.read_csv(args.runs_csv).set_index("run")["final_level"]
        P["group"] = np.where(P["run"].map(lv) >= args.split_at, "alto", "bajo")
    if not C.empty:
        C = C.merge(P[["run", "gen", "group"]], on=["run", "gen"])
        V = int(C["val_seeds"].iloc[0])
        sf = make_sf(V - 1)
        C["t"] = C["val_d"] / C["val_se"].where(C["val_se"] > 0)
        n_par = C.groupby(["run", "gen"])["t"].transform("size")
        C["p_t"] = sf(C["t"].fillna(0.0).to_numpy())
        C.loc[C["t"].isna(), "p_t"] = 1.0
        C["bonf"] = (C["p_t"] < 0.05 / n_par) & (C["val_d"] > 0)
        C["bonf_prac"] = C["bonf"] & (C["val_d"] >= args.min_effect * C["fit_mean"])
        p_null = float(sf(2.0))
    else:
        p_null = float("nan")

    rows = []
    for (gen, grp), pg in P.groupby(["gen", "group"]):
        cg = C[(C["gen"] == gen) & (C["group"] == grp)] if not C.empty else C
        n, k = len(cg), int(cg["confirmed"].sum()) if len(cg) else 0
        rows.append({
            "gen": gen, "group": grp, "n_padres": len(pg), "candidatos": n, "confirmados": k,
            "pct_conf": 100 * k / n if n else np.nan, "pct_azar": 100 * p_null,
            "p_binom": binom_sf(k, n, p_null) if n else np.nan,
            "d_cribado": cg["screen_d"].mean() if n else np.nan, "d_valid": cg["val_d"].mean() if n else np.nan,
            "pct_pos": 100 * (cg["val_d"] > 0).mean() if n else np.nan,
            "bonf": int(cg["bonf"].sum()) if n else 0, "bonf_prac": int(cg["bonf_prac"].sum()) if n else 0,
            "P_mediana": float(pg["p_confirmed"].median()),
        })
    R = pd.DataFrame(rows).sort_values(["gen", "group"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(f"{out}_report.csv", index=False)

    print(f"{len(P)} padres, {len(C)} candidatos re-evaluados"
          + (f" con {V} semillas nuevas; azar esperado = {100 * p_null:.1f}% de confirmados\n" if len(C) else "\n"))
    show = R.copy()
    for col in ("pct_conf", "pct_azar", "d_cribado", "d_valid", "pct_pos"):
        show[col] = show[col].round(1)
    show["p_binom"] = show["p_binom"].map(lambda v: "-" if pd.isna(v) else f"{v:.2g}")
    show["P_mediana"] = show["P_mediana"].map(lambda v: f"{v:.2g}")
    print(show.to_string(index=False))
    print("\nLectura: pct_conf ≈ pct_azar, d_valid ≈ 0 y pct_pos ≈ 50 => lo 'confirmado' es ruido. "
          "Señal real = p_binom pequeño, d_valid > 0 y bonf_prac > 0.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for grp, s in R.groupby("group"):
            ax[0].plot(s["gen"], s["pct_conf"], marker="o", label=grp)
            ax[1].plot(s["gen"], s["d_valid"], marker="o", label=grp)
        ax[0].axhline(100 * p_null, color="gray", ls="--", label="azar")
        ax[1].axhline(0, color="gray", ls="--")
        ax[0].set_ylabel("% de candidatos confirmados")
        ax[1].set_ylabel("Δ validado medio de los candidatos")
        for a in ax:
            a.set_xscale("symlog", linthresh=25)
            a.set_xlabel("generación")
            a.legend()
        fig.tight_layout()
        fig.savefig(f"{out}_report.png", dpi=150)
        print(f"\nEscrito: {out}_report.csv, {out}_report.png")
    except ImportError:
        print(f"\nEscrito: {out}_report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

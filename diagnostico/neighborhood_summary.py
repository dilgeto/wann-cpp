#!/usr/bin/env python3
"""Resume los análisis de vecindad (salidas de wann_car_neighborhood) de muchas
corridas y generaciones, y los agrupa por resultado final.

Lee ``<dir>/<corrida>_g<gen>_parents.csv`` (y ``_neighbors.csv`` para N2), tal
como los deja run_neighborhood.py. Clasifica cada padre (paso 0):

  optimo_local_N1         ningún vecino supera eps en el cribado
  mejoras_no_confirmadas  hubo candidatos pero ninguno se confirma con semillas nuevas
  mejoras_confirmadas     al menos un vecino se confirma
  mejoras_sin_confirmar   el análisis se corrió con --confirm 0

Para los padres sin mejora confirmada, con N2 muestreado (--n2-mids), cuenta los
nietos que superan eps: vía un intermedio neutro o mejor (alcanzable) o vía uno
peor (valle). OJO: las mejoras de N2 son del cribado (una semilla), sin confirmar.

Con --runs-csv (el *_runs.csv de lineage_analysis.py) agrupa las corridas en
terciles de nivel final (bajo/medio/alto) para contrastar semillas buenas y malas.

Uso:
  python diagnostico/neighborhood_summary.py --dir log/diag_neighborhood \\
      [--runs-csv diag_results/lineage_runs.csv] [--out diag_results/neighborhood]
Salidas: <out>_parents.csv (un padre por fila) y <out>_groups.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diag_lib as dl

dl.anchor_to_repo_root()

NAME_RE = re.compile(r"^(?P<run>.+)_g(?P<gen>\d+)_parents\.csv$")
DEFAULTS = {"n_candidates": 0, "n_confirm_tested": 0, "n_confirmed": 0, "p_confirmed": 0.0}


def category(r: pd.Series) -> str:
    if r["stuck"] == 1:
        return "optimo_local_N1"
    if r["n_confirm_tested"] > 0:
        return "mejoras_confirmadas" if r["n_confirmed"] > 0 else "mejoras_no_confirmadas"
    return "mejoras_sin_confirmar"


def n2_counts(neigh_path: Path, parents: pd.DataFrame) -> pd.DataFrame:
    """n2_total / n2_better / n2_via_neutral / n2_via_valley por (label, idx, step)."""
    if not neigh_path.exists():
        return pd.DataFrame()
    nb = pd.read_csv(neigh_path, usecols=["label", "idx", "step", "order", "mid_d_mean", "d_mean"])
    nb = nb[nb["order"] == 2]
    if nb.empty:
        return pd.DataFrame()
    nb = nb.merge(parents[["label", "idx", "step", "eps"]], on=["label", "idx", "step"])
    better = nb["d_mean"] > nb["eps"]
    nb = nb.assign(better=better,
                   via_neutral=better & (nb["mid_d_mean"] >= -nb["eps"]),
                   via_valley=better & (nb["mid_d_mean"] < -nb["eps"]))
    g = nb.groupby(["label", "idx", "step"])
    return pd.DataFrame({"n2_total": g.size(), "n2_better": g["better"].sum(),
                         "n2_via_neutral": g["via_neutral"].sum(),
                         "n2_via_valley": g["via_valley"].sum()}).reset_index()


def load_all(directory: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(directory.glob("*_parents.csv")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        for col, val in DEFAULTS.items():
            if col not in df:
                df[col] = val
        n2 = n2_counts(directory / p.name.replace("_parents.csv", "_neighbors.csv"), df)
        if len(n2):
            df = df.merge(n2, on=["label", "idx", "step"], how="left")
        df["run"] = m.group("run")
        df["gen_req"] = int(m.group("gen"))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["step"] == 0].copy()                      # ignora pasos de --climb
    for col in ("n2_total", "n2_better", "n2_via_neutral", "n2_via_valley"):
        if col not in out:
            out[col] = np.nan
    out["category"] = out.apply(category, axis=1)
    out["identical_share"] = out["n_identical"] / out["n_neighbors"].clip(lower=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="carpeta con los *_parents.csv")
    ap.add_argument("--runs-csv", default="", help="*_runs.csv de lineage_analysis.py (columna final_level)")
    ap.add_argument("--split-at", type=float, default=None,
                    help="en vez de terciles, separa las corridas en 'alto' (final_level >= valor) y "
                         "'bajo' (< valor); requiere --runs-csv")
    ap.add_argument("--out", default="diag_results/neighborhood")
    args = ap.parse_args()

    df = load_all(Path(args.dir))
    if df.empty:
        print(f"Sin resultados en {args.dir}", file=sys.stderr)
        return 1

    df["group"] = "todas"
    if args.split_at is not None and not args.runs_csv:
        print("--split-at requiere --runs-csv", file=sys.stderr)
        return 1
    if args.runs_csv:
        runs = pd.read_csv(args.runs_csv)[["run", "final_level"]]
        df = df.merge(runs, on="run", how="left")
        n_runs = df["run"].nunique()
        if args.split_at is not None:
            df["group"] = np.where(df["final_level"] >= args.split_at, "alto", "bajo")
            df.loc[df["final_level"].isna(), "group"] = "sin_nivel"
        elif n_runs >= 6 and df["final_level"].notna().any():
            lvl = df.drop_duplicates("run").set_index("run")["final_level"]
            terc = pd.qcut(lvl, 3, labels=["bajo", "medio", "alto"], duplicates="drop")
            df["group"] = df["run"].map(terc).astype(str)
        else:
            print("AVISO: menos de 6 corridas con final_level; no se agrupa por terciles.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}_parents.csv", index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        stuck_like = g[g["category"].isin(["optimo_local_N1", "mejoras_no_confirmadas"])]
        n2 = stuck_like.dropna(subset=["n2_total"])
        return pd.Series({
            "n_padres": n,
            "pct_optimo_local": 100 * (g["category"] == "optimo_local_N1").mean(),
            "pct_no_confirmadas": 100 * (g["category"] == "mejoras_no_confirmadas").mean(),
            "pct_confirmadas": 100 * (g["category"] == "mejoras_confirmadas").mean(),
            "mediana_P_confirmada": g.loc[g["n_confirmed"] > 0, "p_confirmed"].median(),
            "pct_vecinos_identicos": 100 * g["identical_share"].median(),
            "n2_padres": len(n2),
            "n2_pct_con_mejora": (100 * (n2["n2_better"] > 0).mean()) if len(n2) else np.nan,
            "n2_pct_via_valle": (100 * ((n2["n2_via_valley"] > 0) & (n2["n2_via_neutral"] == 0)).mean())
                                 if len(n2) else np.nan,
        })

    grouped = df.groupby(["gen_req", "group"])
    try:
        groups = grouped.apply(agg, include_groups=False).reset_index()
    except TypeError:                       # pandas < 2.2 no tiene include_groups
        groups = grouped.apply(agg).reset_index()
    groups.to_csv(f"{out}_groups.csv", index=False)

    gens = sorted(int(g) for g in df["gen_req"].unique())
    print(f"{len(df)} padres ({df['run'].nunique()} corridas, gens {gens})\n")
    shown = groups.round(1).astype({"n_padres": int, "n2_padres": int})
    print(shown.to_string(index=False))
    print("\nLectura: 'optimo_local' = ningún vecino supera eps; 'no_confirmadas' = candidatos que no se "
          "sostienen con semillas nuevas; 'confirmadas' = existe un vecino mejor que la evolución "
          "no ha usado. n2_pct_via_valle = padres cuya única salida en N2 pasa por un intermedio peor.")
    print(f"\nEscrito: {out}_parents.csv, {out}_groups.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

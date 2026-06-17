#!/usr/bin/env python3
"""
plot_screening.py — Genera gráficos desde los resultados del screening.

Lee los CSV/JSONL de screening_reduce/ y screening_full/.
No requiere C++ ni estudio Optuna activo.

Uso
---
  python plot_screening.py                          # todos los run keys
  python plot_screening.py --run acrobot_ttfs_rate  # un run key específico
  python plot_screening.py --phase reduce           # solo reduce
  python plot_screening.py --phase full             # solo full
  python plot_screening.py --out mis_graficos/      # dir de salida

Salida
------
  plots/
    reduce/
      {run_key}_importance.png       # importancia de HPs (Gini + Spearman ρ)
      {run_key}_fitness_rounds.png   # distribución de fitness por round
      {run_key}_space_reduction.png  # reducción del espacio de búsqueda
    full/
      {run_key}_importance.png       # importancia HPs (Phase 2)
      {run_key}_parallel.png         # coordenadas paralelas (Phase 2)
      {run_key}_phase3.png           # validación Phase 3 (mean ± std)
    phase3_summary.png               # comparación de todas las combinaciones
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as ss

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance as sk_perm_imp
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

REDUCE_DIR = Path("screening_reduce")
FULL_DIR   = Path("screening_full")

# Espacio inicial (para comparar reducción)
INITIAL_SPACE: dict[str, tuple] = {
    "alg_probMoo":           ("float", 0.05, 0.70),
    "prob_addConn":          ("float", 0.05, 0.50),
    "prob_addNode":          ("float", 0.05, 0.40),
    "prob_enable":           ("log",   0.005, 0.25),
    "prob_mutAct":           ("float", 0.10, 0.70),
    "prob_toggleExcitatory": ("float", 0.02, 0.30),
    "prob_initEnable":       ("float", 0.20, 0.80),
    "select_cullRatio":      ("float", 0.05, 0.50),
    "select_eliteRatio":     ("float", 0.05, 0.40),
    "select_tournSize":      ("int",   2,    16),
}

# Etiquetas cortas para los ejes
SHORT = {
    "alg_probMoo":            "probMoo",
    "prob_addConn":           "addConn",
    "prob_addNode":           "addNode",
    "prob_enable":            "enable",
    "prob_mutAct":            "mutAct",
    "prob_toggleExcitatory":  "toggleExc",
    "prob_initEnable":        "initEnable",
    "select_cullRatio":       "cullRatio",
    "select_eliteRatio":      "eliteRatio",
    "select_tournSize":       "tournSize",
}

TASKS = ["mountain_car", "acrobot", "pendulum", "car"]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def hp_cols_from(df: pd.DataFrame) -> list[str]:
    exclude = {"round", "trial", "peak_fitness", "elapsed_s",
               "rank", "seed_idx"}
    return [c for c in df.columns if c not in exclude
            and c in INITIAL_SPACE]


def parse_run_key(run_key: str) -> tuple[str, str]:
    """Returns (task, enc_dec). e.g. 'mountain_car_ttfs_rate' → ('mountain_car','ttfs_rate')"""
    for t in sorted(TASKS, key=len, reverse=True):
        if run_key.startswith(t):
            suffix = run_key[len(t):]
            enc_dec = suffix.lstrip("_")
            return t, enc_dec
    return run_key, ""


# ── Importancia ───────────────────────────────────────────────────────────────

def _compute_importance(df_ok: pd.DataFrame, hp_cols: list[str]):
    X = df_ok[hp_cols].values.astype(float)
    y = df_ok["peak_fitness"].values.astype(float)

    rho_d, pval_d = {}, {}
    for i, col in enumerate(hp_cols):
        rho, pval = ss.spearmanr(X[:, i], y)
        rho_d[col]  = float(rho)
        pval_d[col] = float(pval)

    gini_d = {c: 0.0 for c in hp_cols}
    perm_d = {c: 0.0 for c in hp_cols}
    if HAS_SKLEARN and len(df_ok) >= 10:
        rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        n_rep = min(30, max(10, len(df_ok) // 3))
        pi    = sk_perm_imp(rf, X, y, n_repeats=n_rep, random_state=0)
        gini_d = dict(zip(hp_cols, rf.feature_importances_))
        perm_d = dict(zip(hp_cols, pi.importances_mean))

    order = sorted(hp_cols, key=lambda c: gini_d[c])  # ascending → barh
    return order, gini_d, perm_d, rho_d, pval_d


def plot_importance(run_key: str, df_ok: pd.DataFrame,
                    hp_cols: list[str], out_path: Path,
                    title_prefix: str = "") -> None:
    if len(df_ok) < 5:
        print(f"  skip importance {run_key}: <5 trials")
        return

    order, gini, perm, rho, pval = _compute_importance(df_ok, hp_cols)
    labels = [SHORT.get(c, c) for c in order]
    y = np.arange(len(order))

    has_rf = HAS_SKLEARN and len(df_ok) >= 10 and max(gini.values()) > 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle(f"{title_prefix}{run_key}", fontweight="bold")

    # Gini
    axes[0].set_title("RF Gini importance")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    if has_rf:
        max_gini = max(gini.values())
        clr_g = ["#d62728" if abs(gini[c] - max_gini) < 1e-9 else "#4C72B0"
                  for c in order]
        bars = axes[0].barh(y, [gini[c] for c in order],
                            color=clr_g, height=0.65)
        for bar, col in zip(bars, order):
            w = bar.get_width()
            if w > 0.005:
                axes[0].text(w + 0.003, bar.get_y() + bar.get_height() / 2,
                             f"{w:.3f}", va="center", fontsize=7)
        axes[0].set_xlabel("Importance")
    else:
        msg = ("sklearn no instalado\n(pip install scikit-learn)"
               if not HAS_SKLEARN else
               f"RF requiere ≥10 trials\n(hay {len(df_ok)})")
        axes[0].text(0.5, 0.5, msg, ha="center", va="center",
                     transform=axes[0].transAxes,
                     fontsize=10, color="gray",
                     bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
        axes[0].set_xlim(0, 1)

    # Spearman ρ
    clr_r = ["#2ca02c" if rho[c] >= 0 else "#d62728" for c in order]
    axes[1].barh(y, [rho[c] for c in order], color=clr_r,
                 height=0.65, alpha=0.85)
    axes[1].set_title("Spearman ρ  (* p<0.05  ** p<0.01)")
    axes[1].set_xlabel("ρ")
    axes[1].axvline(0, color="black", linewidth=0.8)
    for i, col in enumerate(order):
        xv  = rho[col]
        sig = "**" if pval[col] < 0.01 else ("*" if pval[col] < 0.05 else "")
        off = 0.02 if xv >= 0 else -0.02
        label = f"{xv:+.2f}{sig}"
        axes[1].text(xv + off, i, label, va="center",
                     ha="left" if xv >= 0 else "right",
                     fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Fitness por round ─────────────────────────────────────────────────────────

def plot_fitness_rounds(run_key: str, df: pd.DataFrame,
                        out_path: Path) -> None:
    rounds = sorted(df["round"].unique())
    data   = [df.loc[df["round"] == r, "peak_fitness"].dropna().values
               for r in rounds]

    fig, ax = plt.subplots(figsize=(max(4, len(rounds) * 1.4), 4))
    tick_labels = [f"R{r}" for r in rounds]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    ax.set_xticks(range(1, len(rounds) + 1))
    ax.set_xticklabels(tick_labels)
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.35, 0.80, len(rounds)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    ax.set_title(f"{run_key}  — Fitness por round (reduce)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Peak fitness")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Reducción del espacio ─────────────────────────────────────────────────────

def plot_space_reduction(run_key: str, space_log_path: Path,
                         out_path: Path) -> None:
    entries = []
    with open(space_log_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        print(f"  skip space_reduction {run_key}: space_log vacío")
        return

    last_space = entries[-1]["space"]
    params     = [p for p in INITIAL_SPACE if p in last_space]

    shrink = []
    for p in params:
        _, lo0, hi0 = INITIAL_SPACE[p]
        _, lo1, hi1 = last_space[p]
        span0 = hi0 - lo0 + 1e-12
        span1 = hi1 - lo1 + 1e-12
        shrink.append(max(0.0, 1.0 - span1 / span0) * 100)

    order  = sorted(range(len(params)), key=lambda i: shrink[i], reverse=True)
    labels = [SHORT.get(params[i], params[i]) for i in order]
    vals   = [shrink[i] for i in order]
    colors = ["#d62728" if v >= 20 else "#4C72B0" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Reducción del espacio (%)")
    ax.set_title(f"{run_key}  — Reducción del espacio de búsqueda\n"
                 f"({len(entries)} rounds, INITIAL_SPACE → R{entries[-1]['round']})")
    ax.axvline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, max(max(vals) * 1.25, 15))

    # Etiqueta con % en cada barra
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)

    # Si todo es 0, mostrar aviso
    if max(vals) < 0.5:
        ax.text(0.5, 0.5, "Sin reducción significativa\n(todos los bins sobrevivieron)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Coordenadas paralelas ─────────────────────────────────────────────────────

def plot_parallel_coords(run_key: str, df_ok: pd.DataFrame,
                         hp_cols: list[str], out_path: Path) -> None:
    if len(df_ok) < 4:
        return

    df_plot = df_ok.nlargest(max(4, len(df_ok)), "peak_fitness").copy()

    df_norm = df_plot[hp_cols].copy().astype(float)
    for col in hp_cols:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - mn) / (mx - mn + 1e-12)

    fit      = df_plot["peak_fitness"].values.astype(float)
    norm_fit = (fit - fit.min()) / (fit.max() - fit.min() + 1e-12)
    cmap     = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(12, 4))
    x_pos = np.arange(len(hp_cols))

    # Dibuja las líneas de peor a mejor (para que las buenas queden encima)
    idx_order = np.argsort(norm_fit)
    for i in idx_order:
        ax.plot(x_pos, df_norm.iloc[i].values,
                color=cmap(norm_fit[i]), alpha=0.5, linewidth=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT.get(c, c) for c in hp_cols],
                       rotation=35, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Valor normalizado [0, 1]")
    ax.set_title(f"{run_key}  — Coordenadas paralelas (Phase 2)\n"
                 "Verde = mayor fitness, rojo = menor fitness")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(float(fit.min()), float(fit.max())))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="peak_fitness", shrink=0.85)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Phase 3 barras ────────────────────────────────────────────────────────────

def plot_phase3(run_key: str, summary: pd.DataFrame,
                out_path: Path) -> None:
    df = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(4, len(df) * 1.5), 4))
    colors = ["#d62728" if i == 0 else "#4C72B0" for i in range(len(df))]
    ax.bar(x, df["mean"], yerr=df["std"], capsize=6,
           color=colors, edgecolor="black", alpha=0.85,
           error_kw=dict(elinewidth=1.5, ecolor="black"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {int(r)}" for r in df["rank"]])
    ax.set_ylabel("Peak fitness  (mean ± std, N seeds)")
    ax.set_title(f"{run_key}  — Phase 3: validación multi-semilla\n"
                 "(rojo = mejor config)")

    ylim = ax.get_ylim()
    rng  = ylim[1] - ylim[0]
    for i, row in df.iterrows():
        ax.text(i, row["mean"] + row["std"] + rng * 0.015,
                f"{row['mean']:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Resumen Phase 3 (todas las combinaciones) ─────────────────────────────────

def plot_phase3_summary(records: list[dict], out_path: Path) -> None:
    if not records:
        return

    tasks = [t for t in TASKS if any(r["task"] == t for r in records)]
    ncols = len(tasks)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)

    for ax, task in zip(axes[0], tasks):
        recs = sorted([r for r in records if r["task"] == task],
                      key=lambda r: r["mean"], reverse=True)
        if not recs:
            ax.set_visible(False)
            continue

        labels = [r["enc_dec"] or "default" for r in recs]
        means  = [r["mean"] for r in recs]
        stds   = [r["std"]  for r in recs]
        colors = ["#d62728" if i == 0 else "#4C72B0"
                  for i in range(len(recs))]
        x = np.arange(len(recs))

        ax.bar(x, means, yerr=stds, capsize=5, color=colors,
               edgecolor="black", alpha=0.85,
               error_kw=dict(elinewidth=1.5, ecolor="black"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(task, fontweight="bold")
        ax.set_ylabel("Phase 3 mean fitness (mejor rank)")

        ylim = ax.get_ylim()
        rng  = ylim[1] - ylim[0]
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + rng * 0.015, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Comparación Phase 3 — mejor config validada por combinación",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Procesamiento por fase ────────────────────────────────────────────────────

def process_reduce(run_key: str, plots_dir: Path) -> None:
    out_dir  = plots_dir / "reduce"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path   = REDUCE_DIR / run_key / "results.csv"
    space_path = REDUCE_DIR / run_key / "space_log.jsonl"

    if not csv_path.exists():
        print(f"  [reduce] {run_key}: sin resultados")
        return

    df      = pd.read_csv(csv_path)
    df_ok   = df.dropna(subset=["peak_fitness"])
    hp_cols = hp_cols_from(df_ok)

    print(f"\n[reduce] {run_key}: {len(df_ok)}/{len(df)} trials exitosos")

    plot_importance(run_key, df_ok, hp_cols,
                    out_dir / f"{run_key}_importance.png",
                    title_prefix="[reduce] ")

    if "round" in df.columns and df["round"].nunique() > 1:
        plot_fitness_rounds(run_key, df,
                            out_dir / f"{run_key}_fitness_rounds.png")

    if space_path.exists():
        plot_space_reduction(run_key, space_path,
                             out_dir / f"{run_key}_space_reduction.png")


def process_full(run_key: str, plots_dir: Path) -> dict | None:
    out_dir = plots_dir / "full"
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_path = FULL_DIR / run_key / "p2_results.csv"
    p3_path = FULL_DIR / run_key / "p3_summary.csv"

    if not p2_path.exists():
        print(f"  [full] {run_key}: sin Phase 2")
        return None

    try:
        p2 = pd.read_csv(p2_path)
    except Exception:
        return None

    p2_ok   = p2.dropna(subset=["peak_fitness"])
    hp_cols = hp_cols_from(p2_ok)

    print(f"\n[full] {run_key}: {len(p2_ok)}/{len(p2)} Phase 2 exitosos")

    plot_importance(run_key, p2_ok, hp_cols,
                    out_dir / f"{run_key}_importance.png",
                    title_prefix="[full P2] ")

    plot_parallel_coords(run_key, p2_ok, hp_cols,
                         out_dir / f"{run_key}_parallel.png")

    if not p3_path.exists():
        return None

    try:
        summary = pd.read_csv(p3_path).dropna(subset=["mean"])
    except Exception:
        return None

    if len(summary) == 0:
        return None

    plot_phase3(run_key, summary, out_dir / f"{run_key}_phase3.png")

    best = summary.loc[summary["mean"].idxmax()]
    task, enc_dec = parse_run_key(run_key)
    return {
        "run_key": run_key,
        "task":    task,
        "enc_dec": enc_dec,
        "mean":    float(best["mean"]),
        "std":     float(best["std"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run",   default=None,
                    help="Run key específico (ej. acrobot_ttfs_rate)")
    ap.add_argument("--phase", default="both",
                    choices=["reduce", "full", "both"],
                    help="Qué fase graficar (default: both)")
    ap.add_argument("--out",   default="plots",
                    help="Directorio de salida (default: plots/)")
    args = ap.parse_args()

    plots_dir = Path(args.out)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Recolectar run keys disponibles
    if args.run:
        run_keys = [args.run]
    else:
        seen: set[str] = set()
        run_keys: list[str] = []
        for base in (REDUCE_DIR, FULL_DIR):
            if base.exists():
                for d in sorted(base.iterdir()):
                    if d.is_dir() and d.name not in seen:
                        seen.add(d.name)
                        run_keys.append(d.name)

    print(f"Run keys encontrados: {run_keys}\n")

    p3_records: list[dict] = []

    for rk in run_keys:
        if args.phase in ("reduce", "both"):
            process_reduce(rk, plots_dir)
        if args.phase in ("full", "both"):
            rec = process_full(rk, plots_dir)
            if rec:
                p3_records.append(rec)

    if p3_records and args.phase in ("full", "both"):
        plot_phase3_summary(p3_records, plots_dir / "phase3_summary.png")

    print(f"\nTodos los gráficos guardados en: {plots_dir}/")


if __name__ == "__main__":
    main()

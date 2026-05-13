#!/usr/bin/env python3
"""
screening.py  –  Multi-fidelity random-search hyperparameter screening
                 for WANN-SNN tasks.

Three operational modes
-----------------------
  generate-only  (run locally, Python required)
    Samples N configs via LHS, writes JSON config files and a manifest CSV.
    Transfer the configs to the cluster, run screening_run.sh there, then
    transfer peaks.csv back and analyse locally.

  analyse-only   (run locally after retrieving peaks.csv from cluster)
    Reads manifest.csv + peaks.csv, runs RF / Spearman / Kruskal-Wallis
    analysis, prints and saves the report.

  full           (run entirely locally, Python required on machine)
    generate + run subprocesses + analyse in one shot.

Typical cluster workflow
------------------------
  # 1. Generate configs on your PC
  python screening.py --task mountain_car --mode generate-only --n 64 --seed 42

  # 2. Send to cluster (adjust host/path as needed)
  rsync -av screening/mountain_car/configs/ cluster:wann-cpp/screening/mountain_car/configs/
  scp screening_run.sh cluster:wann-cpp/

  # 3. On the cluster
  bash screening_run.sh --task mountain_car --jobs 16 --omp 12

  # 4. Transfer results back
  rsync -av cluster:wann-cpp/screening/mountain_car/peaks.csv \\
             screening/mountain_car/

  # 5. Analyse on your PC
  python screening.py --task mountain_car --mode analyse-only

Dependencies (pip, local machine only)
---------------------------------------
  pip install numpy scipy pandas scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as ss

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    from scipy.stats.qmc import LatinHypercube
    HAS_LHS = True
except ImportError:
    HAS_LHS = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance as sk_perm_imp
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Per-task defaults ─────────────────────────────────────────────────────────

TASK_DEFAULTS: dict[str, dict] = {
    "mountain_car": {
        "executable":  "./build/wann_mountain_car",
        "base_config": "p/mountain_car_snn.json",
        "fidelity": {
            "maxGen":    64,
            "popSize":   64,
            "alg_nVals": 4,
            "alg_nReps": 2,
            "bestReps":  5,
            "save_mod":  99999,
        },
    },
    "acrobot": {
        "executable":  "./build/wann_acrobot",
        "base_config": "p/acrobot_snn.json",
        "fidelity": {
            "maxGen":    128,
            "popSize":   96,
            "alg_nVals": 4,
            "alg_nReps": 2,
            "bestReps":  5,
            "save_mod":  99999,
        },
    },
    "car": {
        "executable":  "./build/wann_car",
        "base_config": "p/car_snn.json",
        "fidelity": {
            "maxGen":    512,
            "popSize":   128,
            "alg_nVals": 4,
            "alg_nReps": 3,
            "bestReps":  5,
            "save_mod":  99999,
        },
    },
    "pendulum": {
        "executable":  "./build/wann_snn",
        "base_config": "p/pendulum_snn.json",
        "fidelity": {
            "maxGen":    64,
            "popSize":   64,
            "alg_nVals": 4,
            "alg_nReps": 2,
            "bestReps":  5,
            "save_mod":  99999,
        },
    },
}

# ── Search space ──────────────────────────────────────────────────────────────
#
# Format:  param_name → (kind, lo, hi)  or  ("choice", [v1, v2, ...])
#   "uniform" – sample uniformly in [lo, hi]
#   "log"     – sample log-uniformly in [lo, hi]  (good for small probabilities)
#   "int"     – sample integer in {lo, ..., hi}  (inclusive)
#   "bool"    – sample from {True, False}
#   "choice"  – sample uniformly from a list

SEARCH_SPACE: dict[str, tuple] = {
    "prob_addConn":            ("uniform", 0.05, 0.50),
    "prob_addNode":            ("uniform", 0.05, 0.40),
    "prob_mutAct":             ("uniform", 0.10, 0.70),
    "prob_enable":             ("log",     0.005, 0.25),
    "prob_initEnable":         ("uniform", 0.20, 0.80),
    "prob_toggleExcitatory":   ("uniform", 0.02, 0.30),
    "alg_probMoo":             ("uniform", 0.05, 0.70),
    "select_cullRatio":        ("uniform", 0.05, 0.50),
    "select_eliteRatio":       ("uniform", 0.05, 0.40),
    "select_tournSize":        ("int",     2,    16),
}

# Extra parameters active only for specific tasks
TASK_SPACE_EXTRA: dict[str, dict[str, tuple]] = {
    "mountain_car": {
        "reward_shaping_scale": ("uniform", 0.0, 20.0),
    },
}


# ── Sampling ──────────────────────────────────────────────────────────────────

def _lhs_unit(n: int, d: int, seed: int) -> np.ndarray:
    """(n, d) Latin-hypercube samples in [0,1], or pure-random fallback."""
    if HAS_LHS:
        return LatinHypercube(d=d, seed=seed).random(n=n)
    return np.random.default_rng(seed).uniform(0.0, 1.0, (n, d))


def sample_configs(n: int, task: str, seed: int) -> list[dict[str, Any]]:
    """Draw n configs: LHS for continuous dims, independent random for discrete."""
    rng   = np.random.default_rng(seed)
    space = {**SEARCH_SPACE, **TASK_SPACE_EXTRA.get(task, {})}
    params = list(space.keys())

    continuous = [p for p in params if space[p][0] in ("uniform", "log")]
    discrete   = [p for p in params if space[p][0] not in ("uniform", "log")]

    lhs = _lhs_unit(n, len(continuous), seed)

    configs: list[dict] = []
    for i in range(n):
        cfg: dict[str, Any] = {}

        for j, p in enumerate(continuous):
            spec = space[p]
            u = float(lhs[i, j])
            if spec[0] == "uniform":
                cfg[p] = spec[1] + u * (spec[2] - spec[1])
            else:  # log
                lo, hi = np.log(spec[1]), np.log(spec[2])
                cfg[p] = float(np.exp(lo + u * (hi - lo)))

        for p in discrete:
            spec = space[p]
            if spec[0] == "int":
                cfg[p] = int(rng.integers(spec[1], spec[2] + 1))
            elif spec[0] == "bool":
                cfg[p] = bool(rng.integers(0, 2))
            elif spec[0] == "choice":
                cfg[p] = str(rng.choice(spec[1]))

        configs.append(cfg)

    return configs


# ── Generate-only mode ────────────────────────────────────────────────────────

def cmd_generate(task: str, n: int, seed: int, fidelity: dict) -> None:
    """Write JSON config files and a manifest CSV. No training is performed."""
    out_dir = Path("screening") / task
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(n, task, seed)

    for i, cfg in enumerate(configs):
        # JSON seen by the C++ executable: fidelity on top of sampled params
        merged = {**cfg, **fidelity}
        (cfg_dir / f"{i:04d}.json").write_text(json.dumps(merged, indent=2))

    # Manifest: only the sampled hyperparams (what varies), one row per config
    manifest = pd.DataFrame([{"idx": i, **cfg} for i, cfg in enumerate(configs)])
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    td = TASK_DEFAULTS[task]
    print(f"Generated {n} configs → {cfg_dir}/")
    print(f"Manifest  → {manifest_path}")
    print()
    print("── Next steps ──")
    print(f"  # Transfer configs to cluster")
    print(f"  rsync -av screening/{task}/configs/ cluster:wann-cpp/screening/{task}/configs/")
    print(f"  scp screening_run.sh cluster:wann-cpp/")
    print()
    print(f"  # Run on cluster (adjust --jobs / --omp to match available CPUs)")
    print(f"  bash screening_run.sh --task {task} --jobs 16 --omp 12 --seed {seed}")
    print()
    print(f"  # Transfer results back")
    print(f"  rsync -av cluster:wann-cpp/screening/{task}/peaks.csv screening/{task}/")
    print()
    print(f"  # Analyse locally")
    print(f"  python screening.py --task {task} --mode analyse-only")


# ── Analyse-only mode ─────────────────────────────────────────────────────────

def cmd_analyse(task: str, n_top: int) -> None:
    """Merge manifest.csv + peaks.csv and run statistical analysis."""
    out_dir = Path("screening") / task
    manifest_path = out_dir / "manifest.csv"
    peaks_path    = out_dir / "peaks.csv"

    for p in (manifest_path, peaks_path):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            if p == manifest_path:
                print("Run:  python screening.py --task ... --mode generate-only", file=sys.stderr)
            else:
                print("Transfer peaks.csv from the cluster first.", file=sys.stderr)
            sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    peaks    = pd.read_csv(peaks_path, header=None,
                           names=["idx", "peak_fitness", "elapsed_s"])
    peaks["peak_fitness"] = pd.to_numeric(peaks["peak_fitness"], errors="coerce")
    peaks["elapsed_s"]    = pd.to_numeric(peaks["elapsed_s"],    errors="coerce")

    df = manifest.merge(peaks, on="idx", how="left")

    report = analyse(df, task, n_top=n_top)
    print(report)

    rpt_path = out_dir / "analysis.txt"
    rpt_path.write_text(report)
    print(f"Analysis → {rpt_path}")


# ── Full mode (local, subprocess-based) ───────────────────────────────────────

STATS_COLS = ["evals", "fitMed", "fitMax", "fitTop", "fitPeak", "nodeMed", "connMed"]


def _read_peak(stats_path: Path) -> float | None:
    try:
        df = pd.read_csv(stats_path, header=None, names=STATS_COLS)
        val = df["fitPeak"].max()
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def run_one(args: tuple) -> tuple:
    (idx, task, cfg, fidelity, base_config, executable, omp_threads, seed) = args

    merged = {**cfg, **fidelity}

    cfg_dir = Path("screening") / task / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{idx:04d}.json"
    cfg_path.write_text(json.dumps(merged, indent=2))

    prefix = f"scr_{task}/{idx:04d}"
    (Path("log") / f"scr_{task}").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [executable, "-d", base_config, "-p", str(cfg_path),
           "-o", prefix, "-s", str(seed)]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=3600)
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-500:].strip()
            return (idx, cfg, None, elapsed, err)
    except subprocess.TimeoutExpired:
        return (idx, cfg, None, time.monotonic() - t0, "TIMEOUT")
    except Exception as exc:
        return (idx, cfg, None, time.monotonic() - t0, str(exc))

    stats_file = Path("log") / f"scr_{task}" / f"{idx:04d}_stats.out"
    peak = _read_peak(stats_file)
    return (idx, cfg, peak, time.monotonic() - t0, None)


def cmd_full(task: str, n: int, seed: int, fidelity: dict, base_config: str,
             executable: str, jobs: int, omp: int, n_top: int) -> None:
    """Generate, run, and analyse in one shot (requires local executables)."""
    if not Path(executable).exists():
        print(f"ERROR: executable not found: {executable}", file=sys.stderr)
        sys.exit(1)

    configs = sample_configs(n, task, seed)
    out_dir = Path("screening") / task
    out_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (i, task, cfg, fidelity, base_config, executable, omp, seed * 100_000 + i)
        for i, cfg in enumerate(configs)
    ]

    records: list[dict] = []
    t_wall = time.monotonic()

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_one, w): w[0] for w in work}
        done = 0
        for fut in as_completed(futures):
            idx, cfg, peak, elapsed, err = fut.result()
            done += 1
            status = f"peak={peak:9.4f}" if peak is not None else f"FAIL  {(err or '')[:60]}"
            print(f"  [{done:3d}/{n}]  #{idx:04d}  {status}  ({elapsed:.1f}s)")
            row: dict[str, Any] = {"idx": idx, "peak_fitness": peak,
                                   "elapsed_s": round(elapsed, 1)}
            row.update(cfg)
            records.append(row)

    wall = time.monotonic() - t_wall
    print(f"\nWall time: {wall:.1f}s  ({wall/60:.1f} min)")

    df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)

    # Write peaks.csv in the same format as screening_run.sh produces
    peaks = df[["idx", "peak_fitness"]]
    peaks.to_csv(out_dir / "peaks.csv", index=False, header=False)

    # Write manifest.csv (hyperparams only, no fidelity)
    manifest = pd.DataFrame([{"idx": i, **cfg} for i, cfg in enumerate(configs)])
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    report = analyse(df, task, n_top=n_top)
    print(report)
    (out_dir / "analysis.txt").write_text(report)


# ── Statistical analysis ──────────────────────────────────────────────────────

def analyse(df: pd.DataFrame, task: str, n_top: int = 5) -> str:
    W = 66
    lines: list[str] = ["=" * W, f"  Screening analysis — task: {task}", "=" * W, ""]

    df_ok = df.dropna(subset=["peak_fitness"]).copy()
    n_ok, n_all = len(df_ok), len(df)

    if n_ok == 0:
        lines.append("ERROR: no successful runs — nothing to analyse.")
        return "\n".join(lines)

    lines.append(
        f"Successful runs : {n_ok}/{n_all}   "
        f"peak range : [{df_ok['peak_fitness'].min():.4f}, "
        f"{df_ok['peak_fitness'].max():.4f}]"
    )
    lines.append("")

    skip = {"peak_fitness", "elapsed_s"}
    param_cols = [c for c in df_ok.columns if c not in skip]
    hp_cols    = [c for c in param_cols if c != "idx"]

    # ── Top-K ─────────────────────────────────────────────────────────────
    lines.append(f"── Top {n_top} configs by peak_fitness ──")
    top = df_ok.nlargest(n_top, "peak_fitness")[["idx", "peak_fitness"] + hp_cols]
    lines.append(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # ── Encode for numeric analysis ────────────────────────────────────────
    df_enc = df_ok[hp_cols + ["peak_fitness"]].copy()
    cat_cols: list[str] = []
    for col in hp_cols:
        kind = df_enc[col].dtype.kind
        is_cat = kind in ("O", "b") or hasattr(df_enc[col].dtype, "categories")
        if is_cat:
            df_enc[col] = df_enc[col].astype("category").cat.codes.astype(float)
            cat_cols.append(col)

    X = df_enc[hp_cols].values.astype(float)
    y = df_enc["peak_fitness"].values.astype(float)

    # ── Spearman ρ ─────────────────────────────────────────────────────────
    lines.append("── Spearman ρ  (|ρ| descending) ──")
    lines.append(f"  {'Parameter':<36} {'ρ':>7}  {'p-val':>8}  sig")
    corrs = [(col, *ss.spearmanr(X[:, i], y)) for i, col in enumerate(hp_cols)]
    for col, rho, pval in sorted(corrs, key=lambda x: abs(x[1]), reverse=True):
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "  ")
        lines.append(f"  {col:<36} {rho:+7.3f}  {pval:8.4f}  {sig}")
    lines.append("")

    # ── Kruskal-Wallis ─────────────────────────────────────────────────────
    if cat_cols:
        lines.append("── Kruskal-Wallis H-test  (categorical / boolean) ──")
        lines.append(f"  {'Parameter':<36} {'H':>8}  {'p-val':>8}  sig")
        for col in cat_cols:
            raw    = df_ok[col]
            groups = [df_ok.loc[raw == v, "peak_fitness"].values
                      for v in raw.unique() if (raw == v).sum() >= 2]
            if len(groups) < 2:
                lines.append(f"  {col:<36} (insufficient groups)")
                continue
            H, pval = ss.kruskal(*groups)
            sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "  ")
            lines.append(f"  {col:<36} {H:8.2f}  {pval:8.4f}  {sig}")
        lines.append("")

    # ── Random Forest ──────────────────────────────────────────────────────
    if HAS_SKLEARN and n_ok >= 10:
        lines.append("── Random-Forest importance  (Gini + permutation) ──")
        rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        n_rep = min(30, max(10, n_ok // 3))
        pi    = sk_perm_imp(rf, X, y, n_repeats=n_rep, random_state=0)
        rows  = sorted(
            zip(hp_cols, rf.feature_importances_,
                pi.importances_mean, pi.importances_std),
            key=lambda x: x[1], reverse=True
        )
        lines.append(f"  {'Parameter':<36} {'Gini':>7}  {'Perm':>9}  {'±std':>7}")
        for col, gini, pm, ps in rows:
            lines.append(f"  {col:<36} {gini:7.4f}  {pm:+9.4f}  {ps:7.4f}")
        lines.append("")
    elif not HAS_SKLEARN:
        lines.append("sklearn not installed — RF analysis skipped.")
        lines.append("Install with:  pip install scikit-learn\n")
    else:
        lines.append(f"RF needs ≥10 successful runs (have {n_ok}) — skipped.\n")

    lines.append("=" * W)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", required=True, action="append",
                    choices=list(TASK_DEFAULTS), metavar="TASK",
                    help=f"Task to screen (repeat for multiple). "
                         f"Choices: {list(TASK_DEFAULTS)}")
    ap.add_argument("--mode", default="full",
                    choices=["full", "generate-only", "analyse-only"],
                    help="'generate-only': write configs+manifest then exit. "
                         "'analyse-only': read manifest+peaks, run analysis. "
                         "'full': generate+run+analyse locally (default).")
    ap.add_argument("--n",    type=int, default=64,
                    help="Random configs per task (default: 64)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="Parallel processes for 'full' mode (default: 8)")
    ap.add_argument("--omp",  type=int, default=None,
                    help="OMP_NUM_THREADS per run in 'full' mode "
                         "(default: cpu_count // jobs)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Global RNG seed (default: 0)")
    ap.add_argument("--top",  type=int, default=5,
                    help="Top-K configs to show in report (default: 5)")
    ap.add_argument("--base", default=None,
                    help="Override base JSON path (single-task)")
    ap.add_argument("--exe",  default=None,
                    help="Override executable path (single-task)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    omp  = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)

    if not HAS_LHS and args.mode != "analyse-only":
        print("WARNING: scipy.stats.qmc not found — using pure random sampling "
              "(pip install scipy>=1.7 for LHS).", file=sys.stderr)

    for task in args.task:
        td          = TASK_DEFAULTS[task]
        executable  = args.exe  or td["executable"]
        base_config = args.base or td["base_config"]
        fidelity    = td["fidelity"]

        print(f"\n{'='*66}")
        print(f"  Task: {task}   mode: {args.mode}   n: {args.n}   seed: {args.seed}")
        print(f"{'='*66}")

        if args.mode == "generate-only":
            cmd_generate(task, args.n, args.seed, fidelity)

        elif args.mode == "analyse-only":
            cmd_analyse(task, args.top)

        else:  # full
            print(f"  parallel: {args.jobs}   OMP/run: {omp}")
            print(f"  Fidelity : {fidelity}")
            cmd_full(task, args.n, args.seed, fidelity, base_config,
                     executable, args.jobs, omp, args.top)


if __name__ == "__main__":
    main()

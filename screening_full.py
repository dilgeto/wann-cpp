#!/usr/bin/env python3
"""
screening_full.py  –  Full-budget search (Phase 2) and validation (Phase 3).

Phase 2  –  Optuna TPE with full training budget in the reduced space
            produced by screening_reduce.py. Fewer trials are needed because
            the space is already narrowed to promising regions.

Phase 3  –  Validates the top-K configs from Phase 2 by re-running each
            with N different seeds. Reports mean ± std to confirm results
            are reproducible and not due to random luck.

Workflow
--------
  # Both phases in sequence (typical use)
  python screening_full.py --task acrobot --mode both \\
      --n 20 --top 3 --seeds 5 --jobs 10 --omp 24

  # Phase 2 only
  python screening_full.py --task acrobot --mode phase2 --n 20 --jobs 10 --omp 24

  # Phase 3 only (requires p2_results.csv)
  python screening_full.py --task acrobot --mode phase3 --top 3 --seeds 5 --jobs 15 --omp 16

  # Analysis only — local PC, no C++ needed
  python screening_full.py --task acrobot --mode analyse

Space source (in order of priority)
-------------------------------------
  1. screening_reduce/{task}/space_log.jsonl  (last round of iterative reduction)
  2. INITIAL_SPACE from screening_reduce.py   (if reduction was not run)

Dependencies
------------
  pip install optuna numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as ss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
except ImportError:
    print("ERROR: pip install optuna", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance as sk_perm_imp
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from screening_reduce import (
    INITIAL_SPACE,
    TASK_DEFAULTS,
    VALID_ENCODERS,
    VALID_DECODERS,
    suggest_from_space,
    STATS_COLS,
    _read_peak,
    make_run_key,
    encoder_nInput,
)

# ── Full fidelity (from each task's base JSON) ────────────────────────────────
# Phase 2 does NOT override maxGen/popSize — it lets the base config determine
# the training budget. We only pass the 10 hyperparameters as overrides.
# Phase 3 additionally sets save_mod=10 to keep intermediate snapshots.

PHASE3_SAVE_MOD = 10   # save every 10 gens during validation runs

# ── Load reduced space ────────────────────────────────────────────────────────

def load_reduced_space(run_key: str) -> tuple[dict[str, tuple], str]:
    """
    Load the final reduced space from screening_reduce results.
    run_key is '{task}' or '{task}_{encoder}_{decoder}'.
    Falls back to INITIAL_SPACE if no reduction log is found.
    Returns (space, source_description).
    """
    space_log = Path("screening_reduce") / run_key / "space_log.jsonl"
    if not space_log.exists():
        return dict(INITIAL_SPACE), "INITIAL_SPACE (no reduction log found)"

    last_entry = None
    with open(space_log) as f:
        for line in f:
            if line.strip():
                last_entry = json.loads(line)

    if last_entry is None:
        return dict(INITIAL_SPACE), "INITIAL_SPACE (empty reduction log)"

    space = {p: tuple(v) for p, v in last_entry["space"].items()}
    source = (f"screening_reduce/{run_key}/space_log.jsonl "
              f"(round {last_entry['round']})")
    return space, source

# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_subprocess(
    params:      dict[str, Any],
    extra_cfg:   dict,          # merged on top of params (e.g. save_mod)
    base_config: str,
    executable:  str,
    omp:         int,
    seed:        int,
    log_prefix:  str,           # relative prefix passed to -o
    cfg_path:    Path,
) -> float | None:
    """Write config, launch C++ subprocess, return peak fitness."""
    merged = {**params, **extra_cfg}
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(merged, indent=2))

    log_dir = Path("log") / Path(log_prefix).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    cmd = [executable, "-d", base_config, "-p", str(cfg_path),
           "-o", log_prefix, "-s", str(seed)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=7200)
        if proc.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, Exception):
        return None

    stats_file = Path("log") / (log_prefix + "_stats.out")
    return _read_peak(stats_file)

# ── Phase 2 ───────────────────────────────────────────────────────────────────

def run_phase2(
    task:            str,
    run_key:         str,
    space:           dict[str, tuple],
    n_trials:        int,
    jobs:            int,
    omp:             int,
    seed:            int,
    base_config:     str,
    executable:      str,
    out_dir:         Path,
    fixed_overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Optuna TPE search with full training budget (no fidelity overrides).
    No pruning — each trial runs to completion.
    """
    fixed_overrides = fixed_overrides or {}
    cfg_dir = out_dir / "p2_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            constant_liar=True,
            n_startup_trials=max(5, n_trials // 4),
        ),
        # No pruner: full-budget runs are too expensive to cut short
    )

    results: list[dict] = []
    lock = threading.Lock()
    done_counter = [0]

    def objective(trial: optuna.Trial) -> float:
        params     = suggest_from_space(trial, space)
        cfg_path   = cfg_dir / f"{trial.number:04d}.json"
        log_prefix = f"full_p2_{run_key}/{trial.number:04d}"

        # save_mod=99999: only final save (avoids cluttering log/)
        # fixed_overrides applied last (encoder/decoder cannot be overridden by TPE)
        peak = _run_subprocess(
            params, {"save_mod": 99999, **fixed_overrides},
            base_config, executable, omp,
            seed + trial.number, log_prefix, cfg_path,
        )

        with lock:
            done_counter[0] += 1
            best_str = ""
            if study.best_trials:
                try:
                    best_str = f"  best={study.best_value:.4f}"
                except Exception:
                    pass
            peak_str = f"peak={peak:.4f}" if peak is not None else "FAIL"
            print(f"  [{done_counter[0]:3d}/{n_trials}]  "
                  f"#{trial.number:04d}  {peak_str}{best_str}")

            if peak is not None:
                results.append({"trial": trial.number, "peak_fitness": peak, **params})

        if peak is None:
            raise optuna.exceptions.OptunaError("run failed")
        return peak

    study.optimize(objective, n_trials=n_trials, n_jobs=jobs, catch=(Exception,))

    # Always build df with explicit columns so downstream code never sees a
    # column-less df even when all trials failed.
    col_order = ["trial", "peak_fitness"] + list(space.keys())
    if results:
        df = pd.DataFrame(results)[col_order]
    else:
        df = pd.DataFrame(columns=col_order)
    csv_path = out_dir / "p2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPhase 2 complete: {len(df)} successful / {n_trials} trials")
    if len(df) > 0:
        print(f"Best: {df['peak_fitness'].max():.4f}")
    print(f"Results → {csv_path}")
    return df

# ── Phase 3 ───────────────────────────────────────────────────────────────────

def run_phase3(
    task:            str,
    run_key:         str,
    p2_df:           pd.DataFrame,
    top_k:           int,
    n_seeds:         int,
    jobs:            int,
    omp:             int,
    base_config:     str,
    executable:      str,
    out_dir:         Path,
    fixed_overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Re-run the top-K configs from Phase 2 with N different seeds.
    Parallelizes all (K × N) runs simultaneously.
    """
    fixed_overrides = fixed_overrides or {}
    hp_cols = [c for c in p2_df.columns
               if c not in {"trial", "peak_fitness", "elapsed_s"}]

    top_configs = (p2_df
                   .nlargest(top_k, "peak_fitness")
                   .reset_index(drop=True))

    total = top_k * n_seeds
    print(f"\nPhase 3: validating top {top_k} configs × {n_seeds} seeds "
          f"= {total} runs")
    print(f"Configs:")
    for i, row in top_configs.iterrows():
        print(f"  Rank {i}  peak_p2={row['peak_fitness']:.4f}  "
              f"trial=#{int(row['trial']):04d}")
    print()

    cfg_dir = out_dir / "p3_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build all (rank, seed_idx) work items
    work: list[tuple[int, int, dict]] = []
    for rank, row in top_configs.iterrows():
        params = {col: row[col] for col in hp_cols if col in row}
        for si in range(n_seeds):
            work.append((int(rank), si, params))

    results:  list[dict] = []
    lock      = threading.Lock()
    done      = [0]

    def run_one(rank: int, si: int, params: dict) -> None:
        run_seed   = rank * 10000 + si * 100
        cfg_path   = cfg_dir / f"rank{rank:02d}_seed{si:02d}.json"
        log_prefix = f"full_p3_{run_key}/rank{rank:02d}_seed{si:02d}"

        # Phase 3: save every PHASE3_SAVE_MOD gens for analysis
        # fixed_overrides applied last (encoder/decoder cannot be overridden)
        peak = _run_subprocess(
            params, {"save_mod": PHASE3_SAVE_MOD, **fixed_overrides},
            base_config, executable, omp,
            run_seed, log_prefix, cfg_path,
        )

        with lock:
            done[0] += 1
            peak_str = f"peak={peak:.4f}" if peak is not None else "FAIL"
            print(f"  [{done[0]:3d}/{total}]  rank={rank}  seed={si}  {peak_str}")
            results.append({
                "rank":         rank,
                "seed_idx":     si,
                "peak_fitness": peak,
                **params,
            })

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, r, si, p) for r, si, p in work]
        concurrent.futures.wait(futures)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "p3_validation.csv", index=False)

    # Summary per rank
    summary = (df.dropna(subset=["peak_fitness"])
                 .groupby("rank")["peak_fitness"]
                 .agg(["mean", "std", "min", "max", "count"])
                 .reset_index())
    summary.columns = ["rank", "mean", "std", "min", "max", "n_ok"]
    summary["std"] = summary["std"].fillna(0.0)

    # Attach best config params
    for col in hp_cols:
        summary[col] = summary["rank"].map(
            top_configs.set_index(top_configs.index)[col].to_dict()
        )

    summary.to_csv(out_dir / "p3_summary.csv", index=False)
    return df, summary  # type: ignore[return-value]

# ── Analysis ──────────────────────────────────────────────────────────────────

def cmd_analyse(run_key: str, n_top: int = 5) -> None:
    out_dir = Path("screening_full") / run_key
    W = 66

    lines: list[str] = ["=" * W,
                         f"  Full-budget analysis — run: {run_key}",
                         "=" * W, ""]

    # ── Phase 2 ───────────────────────────────────────────────────────────
    p2_path = out_dir / "p2_results.csv"
    if not p2_path.exists():
        lines.append("Phase 2 results not found.")
    else:
        p2 = pd.read_csv(p2_path)
        hp_cols = [c for c in p2.columns
                   if c not in {"trial", "peak_fitness", "elapsed_s"}]
        p2_ok = p2.dropna(subset=["peak_fitness"])

        lines.append(f"Phase 2: {len(p2_ok)}/{len(p2)} successful trials   "
                     f"peak range [{p2_ok['peak_fitness'].min():.4f}, "
                     f"{p2_ok['peak_fitness'].max():.4f}]")
        lines.append("")

        if len(p2_ok) == 0:
            lines.append("No successful trials — skipping analysis.")
            lines.append("")
        else:
            lines.append(f"── Top {n_top} configs (Phase 2) ──")
            top = p2_ok.nlargest(n_top, "peak_fitness")[
                ["trial", "peak_fitness"] + hp_cols]
            lines.append(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            lines.append("")

        # Spearman
        if len(p2_ok) >= 5:
            X = p2_ok[hp_cols].values.astype(float)
            y = p2_ok["peak_fitness"].values.astype(float)
            lines.append("── Spearman ρ  (Phase 2, |ρ| descending) ──")
            lines.append(f"  {'Parameter':<36} {'ρ':>7}  {'p-val':>8}  sig")
            corrs = [(col, *ss.spearmanr(X[:, i], y))
                     for i, col in enumerate(hp_cols)]
            for col, rho, pval in sorted(corrs, key=lambda x: abs(x[1]), reverse=True):
                sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "  ")
                lines.append(f"  {col:<36} {rho:+7.3f}  {pval:8.4f}  {sig}")
            lines.append("")

        # RF
        if HAS_SKLEARN and len(p2_ok) >= 10:
            X = p2_ok[hp_cols].values.astype(float)
            y = p2_ok["peak_fitness"].values.astype(float)
            rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
            rf.fit(X, y)
            n_rep = min(30, max(10, len(p2_ok) // 3))
            pi    = sk_perm_imp(rf, X, y, n_repeats=n_rep, random_state=0)
            rows  = sorted(
                zip(hp_cols, rf.feature_importances_,
                    pi.importances_mean, pi.importances_std),
                key=lambda x: x[1], reverse=True,
            )
            lines.append("── Random-Forest importance  (Phase 2) ──")
            lines.append(f"  {'Parameter':<36} {'Gini':>7}  {'Perm':>9}  {'±std':>7}")
            for col, gini, pm, ps in rows:
                lines.append(f"  {col:<36} {gini:7.4f}  {pm:+9.4f}  {ps:7.4f}")
            lines.append("")

    # ── Phase 3 ───────────────────────────────────────────────────────────
    p3_path = out_dir / "p3_summary.csv"
    if not p3_path.exists():
        lines.append("Phase 3 validation not found.")
    else:
        summary = pd.read_csv(p3_path)
        hp_cols_p3 = [c for c in summary.columns
                      if c not in {"rank", "mean", "std", "min", "max", "n_ok"}]

        lines.append("── Phase 3 Validation ──")
        lines.append(f"  {'Rank':<6} {'Mean':>9} {'±Std':>8} {'Min':>9} "
                     f"{'Max':>9} {'n_ok':>5}")
        for _, row in summary.sort_values("mean", ascending=False).iterrows():
            lines.append(
                f"  {int(row['rank']):<6} {row['mean']:9.4f} "
                f"{row['std']:8.4f} {row['min']:9.4f} "
                f"{row['max']:9.4f} {int(row['n_ok']):5d}"
            )
        lines.append("")

        # Best config detail
        valid_means = summary["mean"].dropna()
        if len(valid_means) == 0:
            lines.append("No successful validation runs.")
            lines.append("")
        else:
            best = summary.loc[valid_means.idxmax()]
            lines.append(f"── Best validated config (rank {int(best['rank'])}) ──")
            lines.append(f"  mean={best['mean']:.4f}  std={best['std']:.4f}  "
                         f"min={best['min']:.4f}  max={best['max']:.4f}")
            for col in hp_cols_p3:
                val = best[col]
                fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                lines.append(f"  {col:<36}  {fmt}")
            lines.append("")

    lines.append("=" * W)
    report = "\n".join(lines)
    print(report)

    rpt_path = out_dir / "analysis.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    rpt_path.write_text(report)
    print(f"Analysis → {rpt_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task",  required=True,
                    choices=list(TASK_DEFAULTS), metavar="TASK")
    ap.add_argument("--mode",  default="both",
                    choices=["phase2", "phase3", "both", "analyse"],
                    help="Which phase(s) to run (default: both)")
    # Phase 2
    ap.add_argument("--n",     type=int, default=20,
                    help="Phase 2: Optuna trials (default: 20)")
    # Phase 3
    ap.add_argument("--top",   type=int, default=3,
                    help="Phase 3: top-K configs to validate (default: 3)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Phase 3: seeds per config (default: 5)")
    # Common
    ap.add_argument("--jobs",  type=int, default=8,
                    help="Parallel workers (default: 8)")
    ap.add_argument("--omp",   type=int, default=None,
                    help="OMP_NUM_THREADS per run (default: cpu_count // jobs)")
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--top-analyse", type=int, default=5, dest="top_analyse",
                    help="Top-K configs in analysis report (default: 5)")
    ap.add_argument("--base",    default=None, help="Override base JSON")
    ap.add_argument("--exe",     default=None, help="Override executable")
    ap.add_argument("--encoder", default=None,
                    choices=sorted(VALID_ENCODERS),
                    help="Override snn_encoder (must match screening_reduce run)")
    ap.add_argument("--decoder", default=None,
                    choices=sorted(VALID_DECODERS),
                    help="Override snn_decoder (must match screening_reduce run)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    td   = TASK_DEFAULTS[args.task]
    omp  = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)
    rkey = make_run_key(args.task, args.encoder, args.decoder)

    # Fixed overrides passed to every C++ run
    fixed_overrides: dict = {}
    if args.encoder:
        fixed_overrides["snn_encoder"] = args.encoder
        n_input = encoder_nInput(args.encoder, TASK_DEFAULTS[args.task]["n_obs"])
        if n_input is not None:
            fixed_overrides["ann_nInput"] = n_input
    if args.decoder:
        fixed_overrides["snn_decoder"] = args.decoder

    if args.mode == "analyse":
        cmd_analyse(rkey, args.top_analyse)
        return

    executable  = args.exe  or td["executable"]
    base_config = args.base or td["base_config"]
    out_dir     = Path("screening_full") / rkey
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(executable).exists():
        print(f"ERROR: {executable} not found. Compile first.", file=sys.stderr)
        sys.exit(1)

    # Load reduced space from matching screening_reduce run
    space, source = load_reduced_space(rkey)
    print(f"\n{'='*66}")
    print(f"  Task: {args.task}   run key: {rkey}   mode: {args.mode}")
    if fixed_overrides:
        print(f"  Fixed overrides: {fixed_overrides}")
    print(f"  Space source: {source}")
    print(f"  Space ({len(space)} params):")
    for p, (kind, lo, hi) in space.items():
        print(f"    {p:<36} [{lo:.4g}, {hi:.4g}]  ({kind})")
    print(f"{'='*66}\n")

    p2_df = None

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if args.mode in ("phase2", "both"):
        print(f"── Phase 2: {args.n} trials, {args.jobs} parallel, OMP={omp} ──\n")
        p2_df = run_phase2(
            task            = args.task,
            run_key         = rkey,
            space           = space,
            n_trials        = args.n,
            jobs            = args.jobs,
            omp             = omp,
            seed            = args.seed,
            base_config     = base_config,
            executable      = executable,
            out_dir         = out_dir,
            fixed_overrides = fixed_overrides,
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────
    if args.mode in ("phase3", "both"):
        # Load phase 2 results if not already in memory
        if p2_df is None:
            p2_path = out_dir / "p2_results.csv"
            if not p2_path.exists():
                print(f"ERROR: {p2_path} not found. Run phase2 first.",
                      file=sys.stderr)
                sys.exit(1)
            try:
                p2_df = pd.read_csv(p2_path)
            except pd.errors.EmptyDataError:
                print(f"ERROR: {p2_path} is empty — run phase2 first.", file=sys.stderr)
                sys.exit(1)
            except Exception as exc:
                print(f"ERROR: could not read {p2_path}: {exc}", file=sys.stderr)
                sys.exit(1)

        if "peak_fitness" not in p2_df.columns or \
                len(p2_df.dropna(subset=["peak_fitness"])) < args.top:
            print(f"ERROR: need at least {args.top} successful phase-2 trials "
                  f"(have {len(p2_df.dropna(subset=['peak_fitness']))}).",
                  file=sys.stderr)
            sys.exit(1)

        print(f"\n── Phase 3: top {args.top} configs × {args.seeds} seeds, "
              f"{args.jobs} parallel, OMP={omp} ──")
        _, summary = run_phase3(
            task            = args.task,
            run_key         = rkey,
            p2_df           = p2_df,
            top_k           = args.top,
            n_seeds         = args.seeds,
            jobs            = args.jobs,
            omp             = omp,
            base_config     = base_config,
            executable      = executable,
            out_dir         = out_dir,
            fixed_overrides = fixed_overrides,
        )

        print("\n── Phase 3 summary ──")
        print(f"  {'Rank':<6} {'Mean':>9} {'±Std':>8} {'Min':>9} {'Max':>9}")
        for _, row in summary.sort_values("mean", ascending=False).iterrows():
            print(f"  {int(row['rank']):<6} {row['mean']:9.4f} "
                  f"{row['std']:8.4f} {row['min']:9.4f} {row['max']:9.4f}")

    cmd_analyse(rkey, args.top_analyse)


if __name__ == "__main__":
    main()

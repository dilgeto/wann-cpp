#!/usr/bin/env python3
"""
screening_reduce.py  –  Iterative search space reduction via Optuna TPE.

Algorithm (per round)
---------------------
  1. Run N trials with TPE in the current search space.
  2. Discretize each hyperparameter into K equal bins.
  3. Apply a one-sided Mann-Whitney U test per bin:
       H₀: this bin's fitness ≥ the rest.
       If rejected (p < α) AND bin median < global median → eliminate bin.
  4. Shrink each parameter's range to the bounding box of surviving bins.
  5. Repeat until no bins are eliminated or max_rounds is reached.

Runs entirely on the cluster (requires Python venv + compiled C++ executables).
Transfer results.csv to your local PC for final analysis.

Usage
-----
  # On the cluster (uses encoder/decoder from base JSON)
  python screening_reduce.py --task acrobot --rounds 4 --n 30 --jobs 20 --omp 12

  # With specific encoder/decoder (output goes to screening_reduce/acrobot_ttfs_rate/)
  python screening_reduce.py --task acrobot --encoder ttfs --decoder rate \\
      --rounds 4 --n 30 --jobs 20 --omp 12

  # On local PC (analysis only — no C++ needed)
  python screening_reduce.py --task acrobot --encoder ttfs --decoder rate --mode analyse

Valid encoders: current  poisson  rate  ttfs  ttfs_log  small  large
Valid decoders: spike_count  rate  first_spike  voting  wta

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

# ── Valid encoder / decoder values ───────────────────────────────────────────

VALID_ENCODERS = {"current", "poisson", "rate", "ttfs", "ttfs_log", "small", "large"}
VALID_DECODERS = {"spike_count", "rate", "first_spike", "voting", "wta"}


def make_run_key(task: str, encoder: str | None, decoder: str | None) -> str:
    """Return the directory key: '{task}_{encoder}_{decoder}' or just '{task}'."""
    if encoder and decoder:
        return f"{task}_{encoder}_{decoder}"
    return task


# ── Per-task defaults ─────────────────────────────────────────────────────────

TASK_DEFAULTS: dict[str, dict] = {
    "mountain_car": {
        "executable":  "./build/wann_mountain_car",
        "base_config": "p/mountain_car_snn.json",
        "n_obs": 2,   # number of observation variables
        "fidelity": {
            "maxGen": 64, "popSize": 64, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5,
        },
    },
    "acrobot": {
        "executable":  "./build/wann_acrobot",
        "base_config": "p/acrobot_snn.json",
        "n_obs": 6,
        "fidelity": {
            "maxGen": 128, "popSize": 96, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5,
        },
    },
    "car": {
        "executable":  "./build/wann_car",
        "base_config": "p/car_snn.json",
        "n_obs": 9,
        "fidelity": {
            "maxGen": 512, "popSize": 128, "alg_nVals": 4,
            "alg_nReps": 3, "bestReps": 5,
        },
    },
    "pendulum": {
        "executable":  "./build/wann_snn",
        "base_config": "p/pendulum_snn.json",
        "n_obs": 3,
        "fidelity": {
            "maxGen": 64, "popSize": 64, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5,
        },
    },
    "disc_mc": {
        "executable":  "./build/wann_disc_mc",
        "base_config": "p/disc_mc_snn.json",
        "n_obs": 2,
        "fidelity": {
            "maxGen": 64, "popSize": 64, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5,
        },
    },
}


def encoder_nInput(encoder: str | None, n_obs: int,
                   neurons_per_var: int = 5) -> int | None:
    """
    Return the ann_nInput required for population-coding encoders, or None
    if the encoder uses one neuron per observation variable (no change needed).

      small : 2 neurons per variable  →  n_obs * 2
      large : neurons_per_var per var →  n_obs * neurons_per_var
      others: 1 neuron per variable   →  no override needed
    """
    if encoder == "small":
        return n_obs * 2
    if encoder == "large":
        return n_obs * neurons_per_var
    return None

# ── Initial search space (10 hyperparameters) ─────────────────────────────────
# Format: param → (kind, lo, hi)
#   "float" – uniform in [lo, hi]
#   "log"   – log-uniform in [lo, hi]
#   "int"   – integer in {lo, ..., hi}

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

# ── Search space helpers ──────────────────────────────────────────────────────

def suggest_from_space(trial: optuna.Trial,
                       space: dict[str, tuple]) -> dict[str, Any]:
    """Suggest all parameters respecting the current (possibly narrowed) space."""
    params: dict[str, Any] = {}
    for name, (kind, lo, hi) in space.items():
        if kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
    return params


def _bin_edges(kind: str, lo: float, hi: float, K: int) -> np.ndarray:
    """K+1 bin edges in natural or log scale."""
    if kind == "log":
        return np.exp(np.linspace(np.log(lo), np.log(hi), K + 1))
    return np.linspace(lo, hi, K + 1)


def reduce_space(
    df: pd.DataFrame,
    space: dict[str, tuple],
    K: int   = 5,
    alpha: float = 0.05,
    min_per_bin: int = 3,
) -> tuple[dict[str, tuple], dict[str, dict]]:
    """
    Discretize each param into K bins. Eliminate bins whose fitness is
    significantly worse than the rest (Mann-Whitney U, one-sided, p < alpha
    AND bin median < global median).

    Returns:
        new_space  – updated (kind, lo, hi) per param
        report     – elimination details per param (for logging)

    Note: surviving bins are merged into a bounding-box interval.
    If eliminated bins leave a gap, the gap is included in the new range.
    This is a deliberate conservative choice: better to keep a slightly
    wider range than to discard potentially good regions with few samples.
    """
    df_ok = df.dropna(subset=["peak_fitness"])
    y     = df_ok["peak_fitness"].values
    global_median = float(np.median(y))

    new_space: dict[str, tuple] = {}
    report:    dict[str, dict]  = {}

    for param, (kind, lo, hi) in space.items():
        if param not in df_ok.columns:
            new_space[param] = (kind, lo, hi)
            continue

        x    = df_ok[param].values
        edges = _bin_edges(kind, lo, hi, K)

        # Assign to bins 0..K-1
        bin_of = np.clip(np.digitize(x, edges[1:-1]), 0, K - 1)

        eliminated: set[int] = set()
        for b in range(K):
            mask   = bin_of == b
            bin_y  = y[mask]
            rest_y = y[~mask]

            if len(bin_y) < min_per_bin or len(rest_y) < 2:
                continue  # not enough data → keep

            # One-sided: is bin significantly WORSE than the rest?
            _, pval = ss.mannwhitneyu(bin_y, rest_y, alternative="less")
            if pval < alpha and float(np.median(bin_y)) < global_median:
                eliminated.add(b)

        # Safety: keep at least one bin (the one with best median)
        surviving = [b for b in range(K) if b not in eliminated]
        if not surviving:
            medians = [
                float(np.median(y[bin_of == b])) if (bin_of == b).any() else -np.inf
                for b in range(K)
            ]
            surviving = [int(np.argmax(medians))]

        # New bounds: bounding box of surviving bins
        new_lo = float(edges[min(surviving)])
        new_hi = float(edges[max(surviving) + 1])

        if kind == "int":
            new_lo = max(lo, float(int(np.ceil(new_lo))))
            new_hi = min(hi, float(int(np.floor(new_hi))))
            if new_lo > new_hi:
                new_lo, new_hi = lo, hi  # fallback: keep original

        new_space[param] = (kind, new_lo, new_hi)

        report[param] = {
            "old":            (lo, hi),
            "new":            (new_lo, new_hi),
            "eliminated_bins": sorted(eliminated),
            "surviving_bins":  surviving,
            "bin_counts":     [int((bin_of == b).sum()) for b in range(K)],
            "bin_medians":    [
                float(np.median(y[bin_of == b])) if (bin_of == b).any() else None
                for b in range(K)
            ],
            "global_median":  global_median,
        }

    return new_space, report

# ── C++ subprocess objective ──────────────────────────────────────────────────

STATS_COLS = ["evals", "fitMed", "fitMax", "fitTop", "fitPeak", "nodeMed", "connMed"]


def _read_peak(path: Path) -> float | None:
    try:
        df  = pd.read_csv(path, header=None, names=STATS_COLS)
        val = df["fitPeak"].max()
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _make_objective(
    run_key:         str,
    space:           dict[str, tuple],
    fidelity:        dict,
    base_config:     str,
    executable:      str,
    omp:             int,
    seed:            int,
    round_idx:       int,
    results:         list,
    lock:            threading.Lock,
    fixed_overrides: dict | None = None,
):
    """Return the Optuna objective closure for one round."""
    # Enable pruning by checkpointing every maxGen/8 generations
    checkpoint_every = max(4, fidelity["maxGen"] // 8)
    fidelity_with_ckpt = {**fidelity, "save_mod": checkpoint_every}
    fixed_overrides = fixed_overrides or {}

    cfg_dir = Path("screening_reduce") / run_key / "configs"
    log_dir = Path("log") / f"reduce_{run_key}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params    = suggest_from_space(trial, space)
        trial_key = f"r{round_idx:02d}_t{trial.number:04d}"
        # fixed_overrides (e.g. snn_encoder, snn_decoder) applied last so they
        # cannot be overridden by TPE suggestions.
        merged  = {**params, **fidelity_with_ckpt, **fixed_overrides}

        cfg_path   = cfg_dir / f"{trial_key}.json"
        prefix     = f"reduce_{run_key}/{trial_key}"
        stats_file = Path("log") / f"reduce_{run_key}" / f"{trial_key}_stats.out"

        cfg_path.write_text(json.dumps(merged, indent=2))

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(omp)
        cmd = [executable, "-d", base_config, "-p", str(cfg_path),
               "-o", prefix, "-s", str(seed + round_idx * 100000 + trial.number)]

        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=env)
        except Exception as exc:
            raise optuna.exceptions.OptunaError(str(exc))

        last_step = 0
        while proc.poll() is None:
            time.sleep(1.5)
            if not stats_file.exists():
                continue
            try:
                df   = pd.read_csv(stats_file, header=None, names=STATS_COLS)
                step = len(df)
                if step > last_step:
                    last_step = step
                    peak_so_far = float(df["fitPeak"].max())
                    trial.report(peak_so_far, step)
                    if trial.should_prune():
                        proc.terminate()
                        proc.wait()
                        raise optuna.TrialPruned()
            except (optuna.TrialPruned, optuna.exceptions.OptunaError):
                raise
            except Exception:
                pass

        proc.wait()
        elapsed = time.monotonic() - t0

        if proc.returncode != 0:
            raise optuna.exceptions.OptunaError("non-zero exit")

        peak = _read_peak(stats_file)
        if peak is None:
            raise optuna.exceptions.OptunaError("no stats file")

        row = {"round": round_idx, "trial": trial.number,
               "peak_fitness": peak, "elapsed_s": round(elapsed, 1),
               **params}
        with lock:
            results.append(row)

        return peak

    return objective

# ── One optimization round ────────────────────────────────────────────────────

def run_round(
    run_key:         str,
    round_idx:       int,
    space:           dict[str, tuple],
    n:               int,
    jobs:            int,
    omp:             int,
    seed:            int,
    fidelity:        dict,
    base_config:     str,
    executable:      str,
    fixed_overrides: dict | None = None,
) -> list[dict]:
    """Run N trials in parallel using Optuna TPE + MedianPruner."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed + round_idx,
            constant_liar=True,
            n_startup_trials=max(5, n // 4),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
        ),
    )

    results: list[dict] = []
    lock = threading.Lock()

    objective = _make_objective(
        run_key, space, fidelity, base_config, executable,
        omp, seed, round_idx, results, lock, fixed_overrides,
    )

    study.optimize(
        objective,
        n_trials=n,
        n_jobs=jobs,
        catch=(Exception,),
    )

    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned   = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.PRUNED])

    if n_complete > 0:
        print(f"  Round {round_idx}: {n_complete} complete, {n_pruned} pruned  "
              f"best={study.best_value:.4f}")
    else:
        print(f"  Round {round_idx}: 0 complete, {n_pruned} pruned — "
              f"check executable / config")

    return results

# ── Space reduction report ────────────────────────────────────────────────────

def _print_reduction(round_idx: int, space_before: dict,
                     report: dict, K: int) -> None:
    print(f"\n── Space after round {round_idx} (K={K} bins, α=0.05) ──")
    any_change = False
    for param, info in report.items():
        lo_old, hi_old = info["old"]
        lo_new, hi_new = info["new"]
        elim = info["eliminated_bins"]
        if not elim:
            continue
        any_change = True
        pct = 100.0 * (1.0 - (hi_new - lo_new) / (hi_old - lo_old + 1e-12))
        kind = space_before[param][0]
        unit = "" if kind != "int" else " (int)"
        print(f"  {param:<36}  [{lo_old:.4g}, {hi_old:.4g}] → "
              f"[{lo_new:.4g}, {hi_new:.4g}]{unit}  "
              f"(-{pct:.0f}%)  elim bins: {elim}")
    if not any_change:
        print("  No bins eliminated — space unchanged.")

# ── Main loop ─────────────────────────────────────────────────────────────────

def cmd_full(
    task:            str,
    max_rounds:      int,
    n:               int,
    jobs:            int,
    omp:             int,
    seed:            int,
    K:               int,
    alpha:           float,
    n_top:           int,
    base_config:     str,
    executable:      str,
    encoder:         str | None = None,
    decoder:         str | None = None,
) -> None:
    fidelity = dict(TASK_DEFAULTS[task]["fidelity"])
    rkey = make_run_key(task, encoder, decoder)

    # Fixed overrides: encoder/decoder (+ ann_nInput for population encoders)
    fixed_overrides: dict = {}
    if encoder:
        fixed_overrides["snn_encoder"] = encoder
        n_input = encoder_nInput(encoder, TASK_DEFAULTS[task]["n_obs"])
        if n_input is not None:
            fixed_overrides["ann_nInput"] = n_input
    if decoder:
        fixed_overrides["snn_decoder"] = decoder

    out_dir = Path("screening_reduce") / rkey
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv  = out_dir / "results.csv"
    space_log    = out_dir / "space_log.jsonl"

    # Load existing results (resume support)
    all_results: list[dict] = []
    if results_csv.exists():
        try:
            all_results = pd.read_csv(results_csv).to_dict("records")
            print(f"Resuming: loaded {len(all_results)} existing results.")
        except Exception:
            print("Resuming: existing results.csv is empty or corrupt — starting fresh.")

    # Restore narrowed space from previous rounds if resuming
    space = dict(INITIAL_SPACE)
    if space_log.exists():
        with open(space_log) as f:
            last_entry = None
            for line in f:
                if line.strip():
                    last_entry = json.loads(line)
        if last_entry is not None:
            space = {p: tuple(v) for p, v in last_entry["space"].items()}
            print(f"Resuming: restored space from round {last_entry['round']}.")

    print(f"\n{'='*66}")
    print(f"  Task: {task}   run key: {rkey}")
    if fixed_overrides:
        print(f"  Fixed overrides: {fixed_overrides}")
    print(f"  rounds: {max_rounds}   n/round: {n}")
    print(f"  parallel: {jobs}   OMP/run: {omp}   K bins: {K}   α: {alpha}")
    print(f"  Fidelity: {fidelity}")
    print(f"{'='*66}\n")

    for r in range(max_rounds):
        print(f"Round {r}  —  space:")
        for p, (kind, lo, hi) in space.items():
            print(f"  {p:<36} [{lo:.4g}, {hi:.4g}]  ({kind})")
        print()

        new_results = run_round(
            rkey, r, space, n, jobs, omp, seed,
            fidelity, base_config, executable, fixed_overrides,
        )
        all_results.extend(new_results)

        # Save after every round (crash-safe)
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(results_csv, index=False)

        if len(df_all.dropna(subset=["peak_fitness"])) < 3:
            print("  Too few successful trials to reduce space — skipping reduction.")
            continue

        space_before = dict(space)
        space, report = reduce_space(df_all, space, K=K, alpha=alpha)

        # Log space evolution
        log_entry = {
            "round": r,
            "n_trials": len(new_results),
            "space": {p: list(v) for p, v in space.items()},
            "eliminated": {
                p: info["eliminated_bins"]
                for p, info in report.items()
                if info["eliminated_bins"]
            },
        }
        with open(space_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        _print_reduction(r, space_before, report, K)

        # Convergence check
        any_eliminated = any(info["eliminated_bins"] for info in report.values())
        if not any_eliminated:
            print(f"\nConverged at round {r} — no bins eliminated.")
            break

        print()

    print(f"\nTotal trials: {len(all_results)}")
    print(f"Results → {results_csv}")
    cmd_analyse(rkey, n_top)

# ── Analysis (local or cluster) ───────────────────────────────────────────────

def cmd_analyse(run_key: str, n_top: int = 5) -> None:
    out_dir     = Path("screening_reduce") / run_key
    results_csv = out_dir / "results.csv"
    space_log   = out_dir / "space_log.jsonl"

    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(results_csv)
    df_ok = df.dropna(subset=["peak_fitness"])

    hp_cols = [c for c in df_ok.columns
               if c not in {"round", "trial", "peak_fitness", "elapsed_s"}]

    W = 66
    lines: list[str] = ["=" * W, f"  Reduction analysis — run: {run_key}", "=" * W, ""]
    lines.append(f"Total trials: {len(df)}   successful: {len(df_ok)}   "
                 f"rounds: {df['round'].nunique()}")
    lines.append(f"Peak range: [{df_ok['peak_fitness'].min():.4f}, "
                 f"{df_ok['peak_fitness'].max():.4f}]")
    lines.append("")

    # ── Space evolution ────────────────────────────────────────────────────
    if space_log.exists():
        lines.append("── Space evolution (eliminated bins per round) ──")
        with open(space_log) as f:
            for line in f:
                entry = json.loads(line)
                r     = entry["round"]
                elim  = entry.get("eliminated", {})
                if elim:
                    lines.append(f"  Round {r}:")
                    for p, bins in elim.items():
                        _, lo, hi = entry["space"][p]
                        lines.append(f"    {p:<36} bins {bins} → [{lo:.4g}, {hi:.4g}]")
                else:
                    lines.append(f"  Round {r}: no bins eliminated")
        lines.append("")

    # ── Top-K ─────────────────────────────────────────────────────────────
    lines.append(f"── Top {n_top} configs by peak_fitness ──")
    top = df_ok.nlargest(n_top, "peak_fitness")[
        ["round", "trial", "peak_fitness"] + hp_cols]
    lines.append(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # ── Spearman ρ ─────────────────────────────────────────────────────────
    X = df_ok[hp_cols].values.astype(float)
    y = df_ok["peak_fitness"].values.astype(float)

    lines.append("── Spearman ρ  (|ρ| descending) ──")
    lines.append(f"  {'Parameter':<36} {'ρ':>7}  {'p-val':>8}  sig")
    corrs = [(col, *ss.spearmanr(X[:, i], y)) for i, col in enumerate(hp_cols)]
    for col, rho, pval in sorted(corrs, key=lambda x: abs(x[1]), reverse=True):
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "  ")
        lines.append(f"  {col:<36} {rho:+7.3f}  {pval:8.4f}  {sig}")
    lines.append("")

    # ── Random Forest ──────────────────────────────────────────────────────
    if HAS_SKLEARN and len(df_ok) >= 10:
        lines.append("── Random-Forest importance  (Gini + permutation) ──")
        rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        n_rep = min(30, max(10, len(df_ok) // 3))
        pi    = sk_perm_imp(rf, X, y, n_repeats=n_rep, random_state=0)
        rows  = sorted(
            zip(hp_cols, rf.feature_importances_,
                pi.importances_mean, pi.importances_std),
            key=lambda x: x[1], reverse=True,
        )
        lines.append(f"  {'Parameter':<36} {'Gini':>7}  {'Perm':>9}  {'±std':>7}")
        for col, gini, pm, ps in rows:
            lines.append(f"  {col:<36} {gini:7.4f}  {pm:+9.4f}  {ps:7.4f}")
        lines.append("")
    elif not HAS_SKLEARN:
        lines.append("sklearn not installed — RF skipped.  pip install scikit-learn\n")

    lines.append("=" * W)
    report = "\n".join(lines)
    print(report)

    rpt_path = out_dir / "analysis.txt"
    rpt_path.write_text(report)
    print(f"\nAnalysis → {rpt_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task",   required=True,
                    choices=list(TASK_DEFAULTS), metavar="TASK")
    ap.add_argument("--mode",   default="full",
                    choices=["full", "analyse"],
                    help="'full': run on cluster. 'analyse': report only (local).")
    ap.add_argument("--rounds", type=int, default=4,
                    help="Max reduction rounds (default: 4)")
    ap.add_argument("--n",      type=int, default=30,
                    help="Trials per round (default: 30)")
    ap.add_argument("--jobs",   type=int, default=8,
                    help="Parallel workers (default: 8)")
    ap.add_argument("--omp",    type=int, default=None,
                    help="OMP_NUM_THREADS per run (default: cpu_count // jobs)")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--K",      type=int, default=5,
                    help="Bins per parameter for reduction (default: 5)")
    ap.add_argument("--alpha",  type=float, default=0.05,
                    help="Mann-Whitney significance threshold (default: 0.05)")
    ap.add_argument("--top",     type=int, default=5)
    ap.add_argument("--base",    default=None, help="Override base JSON")
    ap.add_argument("--exe",     default=None, help="Override executable")
    ap.add_argument("--encoder", default=None,
                    choices=sorted(VALID_ENCODERS),
                    help="Override snn_encoder (also changes output dir)")
    ap.add_argument("--decoder", default=None,
                    choices=sorted(VALID_DECODERS),
                    help="Override snn_decoder (also changes output dir)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    td   = TASK_DEFAULTS[args.task]
    omp  = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)
    rkey = make_run_key(args.task, args.encoder, args.decoder)

    if args.mode == "analyse":
        cmd_analyse(rkey, args.top)
        return

    executable  = args.exe  or td["executable"]
    base_config = args.base or td["base_config"]

    if not Path(executable).exists():
        print(f"ERROR: {executable} not found. Compile first.", file=sys.stderr)
        sys.exit(1)

    cmd_full(
        task        = args.task,
        max_rounds  = args.rounds,
        n           = args.n,
        jobs        = args.jobs,
        omp         = omp,
        seed        = args.seed,
        K           = args.K,
        alpha       = args.alpha,
        n_top       = args.top,
        base_config = base_config,
        executable  = executable,
        encoder     = args.encoder,
        decoder     = args.decoder,
    )


if __name__ == "__main__":
    main()

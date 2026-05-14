#!/usr/bin/env python3
"""
screening.py  –  Bayesian hyperparameter screening for WANN-SNN tasks.
                 Optuna TPE sampler · constant-liar parallelism · MedianPruner.

Cluster workflow (no Python on cluster) — repeat for multiple rounds:
---------------------------------------------------------------------
  # 1. Generate next batch of configs locally (TPE-guided after round 0)
  python screening.py --task acrobot --mode suggest --n 20

  # 2. Transfer configs to cluster
  rsync -av screening/acrobot/configs/ cluster:wann-cpp/screening/acrobot/configs/
  scp screening_run.sh cluster:wann-cpp/          # only needed once

  # 3. Run on cluster (bash, no Python needed)
  bash screening_run.sh --task acrobot --jobs 20 --omp 12

  # 4. Transfer results back
  rsync -av cluster:wann-cpp/screening/acrobot/peaks.csv screening/acrobot/

  # 5. Tell Optuna the results  (updates TPE model for next round)
  python screening.py --task acrobot --mode observe

  # Repeat steps 1-5 as many rounds as desired, then:
  python screening.py --task acrobot --mode analyse

Local workflow (compiled exes available):
-----------------------------------------
  python screening.py --task acrobot --mode full --n-trials 60 --jobs 8

Dependencies:
  pip install optuna numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as ss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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
            "maxGen": 64, "popSize": 64, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5, "save_mod": 99999,
        },
    },
    "acrobot": {
        "executable":  "./build/wann_acrobot",
        "base_config": "p/acrobot_snn.json",
        "fidelity": {
            "maxGen": 128, "popSize": 96, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5, "save_mod": 99999,
        },
    },
    "car": {
        "executable":  "./build/wann_car",
        "base_config": "p/car_snn.json",
        "fidelity": {
            "maxGen": 512, "popSize": 128, "alg_nVals": 4,
            "alg_nReps": 3, "bestReps": 5, "save_mod": 99999,
        },
    },
    "pendulum": {
        "executable":  "./build/wann_snn",
        "base_config": "p/pendulum_snn.json",
        "fidelity": {
            "maxGen": 64, "popSize": 64, "alg_nVals": 4,
            "alg_nReps": 2, "bestReps": 5, "save_mod": 99999,
        },
    },
}

# ── Search space (10 fixed hyperparameters) ───────────────────────────────────

def suggest_params(trial: Any) -> dict[str, Any]:
    """Map an Optuna trial to the 10 hyperparameters under study."""
    return {
        "alg_probMoo":           trial.suggest_float("alg_probMoo",           0.05, 0.70),
        "prob_addConn":          trial.suggest_float("prob_addConn",          0.05, 0.50),
        "prob_addNode":          trial.suggest_float("prob_addNode",          0.05, 0.40),
        "prob_enable":           trial.suggest_float("prob_enable",           0.005, 0.25, log=True),
        "prob_mutAct":           trial.suggest_float("prob_mutAct",           0.10, 0.70),
        "prob_toggleExcitatory": trial.suggest_float("prob_toggleExcitatory", 0.02, 0.30),
        "prob_initEnable":       trial.suggest_float("prob_initEnable",       0.20, 0.80),
        "select_cullRatio":      trial.suggest_float("select_cullRatio",      0.05, 0.50),
        "select_eliteRatio":     trial.suggest_float("select_eliteRatio",     0.05, 0.40),
        "select_tournSize":      trial.suggest_int(  "select_tournSize",      2,    16),
    }

# ── Optuna study factory ──────────────────────────────────────────────────────

def _study_storage(study_dir: Path) -> str:
    return f"sqlite:///{study_dir}/study.db"


def create_or_load_study(task: str, seed: int, study_dir: Path) -> Any:
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        constant_liar=True,   # treat in-progress trials as done → diverse batch
        n_startup_trials=10,  # pure random for first 10 trials, then TPE
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # don't prune before 5 complete trials
        n_warmup_steps=2,     # don't prune in first 2 intermediate steps
        interval_steps=1,
    )
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=_study_storage(study_dir),
        study_name=task,
        load_if_exists=True,
    )

# ── Mode: suggest ─────────────────────────────────────────────────────────────

def cmd_suggest(task: str, n: int, fidelity: dict, seed: int) -> None:
    """Ask Optuna for n new trials and write their JSON configs."""
    study_dir = Path("screening") / task
    cfg_dir   = study_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    study = create_or_load_study(task, seed, study_dir)

    n_done = sum(1 for t in study.trials
                 if t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.PRUNED))

    new_trials: list[tuple[int, dict]] = []
    for _ in range(n):
        trial  = study.ask()
        params = suggest_params(trial)
        new_trials.append((trial.number, params))

    for trial_num, params in new_trials:
        merged = {**params, **fidelity}
        (cfg_dir / f"{trial_num:04d}.json").write_text(json.dumps(merged, indent=2))

    nums = [num for num, _ in new_trials]
    print(f"Generated {n} configs  (trials #{nums[0]:04d}–#{nums[-1]:04d})")
    print(f"Study: {n_done} completed/pruned trials before this round")
    if n_done >= 1:
        try:
            print(f"Best so far: {study.best_value:.4f}  (trial #{study.best_trial.number:04d})")
        except Exception:
            pass
    print()
    print("── Next steps ──")
    print(f"  rsync -av screening/{task}/configs/ cluster:wann-cpp/screening/{task}/configs/")
    print(f"  bash screening_run.sh --task {task} --jobs 20 --omp 12 --seed {seed}")
    print(f"  rsync -av cluster:wann-cpp/screening/{task}/peaks.csv screening/{task}/")
    print(f"  python screening.py --task {task} --mode observe")

# ── Mode: observe ─────────────────────────────────────────────────────────────

def cmd_observe(task: str) -> None:
    """Read peaks.csv from the cluster and tell results to Optuna."""
    study_dir  = Path("screening") / task
    peaks_path = study_dir / "peaks.csv"

    if not peaks_path.exists():
        print(f"ERROR: {peaks_path} not found. Transfer it from the cluster first.",
              file=sys.stderr)
        sys.exit(1)

    study = optuna.load_study(study_name=task, storage=_study_storage(study_dir))

    peaks = pd.read_csv(peaks_path, header=None,
                        names=["idx", "peak_fitness", "elapsed_s"])
    peaks["idx"]          = peaks["idx"].apply(lambda x: int(str(x).strip()))
    peaks["peak_fitness"] = pd.to_numeric(peaks["peak_fitness"], errors="coerce")

    running = {t.number for t in study.trials
               if t.state == optuna.trial.TrialState.RUNNING}

    n_ok = n_fail = n_skip = 0
    for _, row in peaks.iterrows():
        num  = int(row["idx"])
        peak = row["peak_fitness"]
        if num not in running:
            n_skip += 1
            continue
        if pd.isna(peak):
            study.tell(num, state=optuna.trial.TrialState.FAIL)
            n_fail += 1
        else:
            study.tell(num, float(peak))
            n_ok += 1

    n_complete = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"Told Optuna: {n_ok} complete, {n_fail} failed, {n_skip} skipped")
    print(f"Study total: {n_complete} complete trials")
    if n_complete > 0:
        best = study.best_trial
        print(f"Best so far: {study.best_value:.4f}  (trial #{best.number:04d})")
        print(f"  params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in best.params.items()} }")
    print()
    print("── Next steps ──")
    print(f"  # Another round (TPE now informed by results)")
    print(f"  python screening.py --task {task} --mode suggest --n 20")
    print(f"  # Or final analysis")
    print(f"  python screening.py --task {task} --mode analyse")

# ── Mode: analyse ─────────────────────────────────────────────────────────────

STATS_COLS = ["evals", "fitMed", "fitMax", "fitTop", "fitPeak", "nodeMed", "connMed"]


def _read_peak(stats_path: Path) -> float | None:
    try:
        df  = pd.read_csv(stats_path, header=None, names=STATS_COLS)
        val = df["fitPeak"].max()
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def cmd_analyse(task: str, n_top: int) -> None:
    """Load Optuna study, run Spearman + RF analysis, print and save report."""
    study_dir = Path("screening") / task
    db_path   = study_dir / "study.db"

    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run --mode suggest first.", file=sys.stderr)
        sys.exit(1)

    study    = optuna.load_study(study_name=task, storage=_study_storage(study_dir))
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED]

    if not complete:
        print("No complete trials to analyse.")
        return

    print(f"Study: {len(complete)} complete, {len(pruned)} pruned, "
          f"{len(study.trials)} total")

    rows = [{"idx": t.number, "peak_fitness": t.value, **t.params}
            for t in complete]
    df = pd.DataFrame(rows)

    report = _analyse(df, task, n_top=n_top)
    print(report)

    rpt_path = study_dir / "analysis.txt"
    rpt_path.write_text(report)
    df.to_csv(study_dir / "results.csv", index=False)
    print(f"Analysis → {rpt_path}")
    print(f"Results  → {study_dir}/results.csv")

# ── Mode: full (local, with pruning) ─────────────────────────────────────────

def _run_trial(trial: Any, task: str, params: dict, fidelity: dict,
               base_config: str, executable: str, omp: int, seed: int) -> float | None:
    """
    Launch one C++ training subprocess.
    Polls the stats file for intermediate values so Optuna's MedianPruner
    can kill under-performing runs early (save_mod set to maxGen/8).
    """
    checkpoint_every = max(4, fidelity["maxGen"] // 8)
    merged = {**params, **fidelity, "save_mod": checkpoint_every}

    cfg_dir  = Path("screening") / task / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{trial.number:04d}.json"
    cfg_path.write_text(json.dumps(merged, indent=2))

    prefix = f"scr_{task}/{trial.number:04d}"
    (Path("log") / f"scr_{task}").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    cmd = [executable, "-d", base_config, "-p", str(cfg_path),
           "-o", prefix, "-s", str(seed + trial.number)]

    stats_file = Path("log") / f"scr_{task}" / f"{trial.number:04d}_stats.out"

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, env=env)
    except Exception:
        return None

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
                peak = float(df["fitPeak"].max())
                trial.report(peak, step)
                if trial.should_prune():
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception:
            pass

    proc.wait()
    if proc.returncode != 0:
        return None
    return _read_peak(stats_file)


def _worker(study: Any, task: str, fidelity: dict, base_config: str,
            executable: str, omp: int, seed: int) -> tuple[int, float | None, str]:
    """Ask one trial, run it, tell result. Returns (trial_num, peak, status)."""
    trial  = study.ask()
    params = suggest_params(trial)
    try:
        peak = _run_trial(trial, task, params, fidelity,
                          base_config, executable, omp, seed)
        if peak is None:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            return trial.number, None, "FAIL"
        study.tell(trial.number, float(peak))
        return trial.number, peak, "OK"
    except optuna.TrialPruned:
        study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
        return trial.number, None, "PRUNED"


def cmd_full(task: str, n_trials: int, fidelity: dict, base_config: str,
             executable: str, jobs: int, omp: int, seed: int, n_top: int) -> None:
    """Async parallel Optuna loop: each finished worker immediately asks for the next trial."""
    if not Path(executable).exists():
        print(f"ERROR: {executable} not found.", file=sys.stderr)
        sys.exit(1)

    study_dir = Path("screening") / task
    study_dir.mkdir(parents=True, exist_ok=True)
    study = create_or_load_study(task, seed, study_dir)

    n_submitted = 0
    done        = 0

    print(f"Running {n_trials} trials   parallel={jobs}   OMP/run={omp}")
    print(f"Pruner: MedianPruner (activates after 5 complete trials)")
    print()

    t_wall = time.monotonic()

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures: dict = {}

        # Submit initial batch
        for _ in range(min(jobs, n_trials)):
            f = pool.submit(_worker, study, task, fidelity,
                            base_config, executable, omp, seed)
            futures[f] = None
            n_submitted += 1

        for fut in as_completed(futures):
            trial_num, peak, status = fut.result()
            done += 1

            n_complete = sum(1 for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE)
            best_str = ""
            if n_complete > 0:
                best_str = f"  best={study.best_value:.4f}"

            peak_str = f"peak={peak:.4f}" if peak is not None else status
            print(f"  [{done:3d}/{n_trials}]  #{trial_num:04d}  {peak_str}{best_str}")

            # Submit next trial while there are remaining
            if n_submitted < n_trials:
                f = pool.submit(_worker, study, task, fidelity,
                                base_config, executable, omp, seed)
                futures[f] = None
                n_submitted += 1

    wall = time.monotonic() - t_wall
    print(f"\nWall time: {wall:.1f}s  ({wall/60:.1f} min)")
    cmd_analyse(task, n_top)

# ── Statistical analysis ──────────────────────────────────────────────────────

def _analyse(df: pd.DataFrame, task: str, n_top: int = 5) -> str:
    W = 66
    lines: list[str] = ["=" * W, f"  Screening analysis — task: {task}", "=" * W, ""]

    df_ok = df.dropna(subset=["peak_fitness"]).copy()
    n_ok, n_all = len(df_ok), len(df)

    if n_ok == 0:
        lines.append("No successful runs — nothing to analyse.")
        return "\n".join(lines)

    lines.append(
        f"Successful runs : {n_ok}/{n_all}   "
        f"peak range : [{df_ok['peak_fitness'].min():.4f}, "
        f"{df_ok['peak_fitness'].max():.4f}]"
    )
    lines.append("")

    skip    = {"peak_fitness", "elapsed_s"}
    hp_cols = [c for c in df_ok.columns if c not in skip and c != "idx"]

    # ── Top-K ─────────────────────────────────────────────────────────────
    lines.append(f"── Top {n_top} configs by peak_fitness ──")
    top = df_ok.nlargest(n_top, "peak_fitness")[["idx", "peak_fitness"] + hp_cols]
    lines.append(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # ── Encode for numeric analysis ────────────────────────────────────────
    df_enc = df_ok[hp_cols + ["peak_fitness"]].copy()
    cat_cols: list[str] = []
    for col in hp_cols:
        kind   = df_enc[col].dtype.kind
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

    # ── Kruskal-Wallis (categorical / boolean) ─────────────────────────────
    if cat_cols:
        lines.append("── Kruskal-Wallis H-test ──")
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
            key=lambda x: x[1], reverse=True,
        )
        lines.append(f"  {'Parameter':<36} {'Gini':>7}  {'Perm':>9}  {'±std':>7}")
        for col, gini, pm, ps in rows:
            lines.append(f"  {col:<36} {gini:7.4f}  {pm:+9.4f}  {ps:7.4f}")
        lines.append("")
    elif not HAS_SKLEARN:
        lines.append("sklearn not installed — RF skipped.  pip install scikit-learn\n")
    else:
        lines.append(f"RF needs ≥10 complete runs (have {n_ok}) — skipped.\n")

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
                    help=f"Task (repeat for multiple). Choices: {list(TASK_DEFAULTS)}")
    ap.add_argument("--mode", default="suggest",
                    choices=["suggest", "observe", "analyse", "full"],
                    help="suggest: generate configs via TPE. "
                         "observe: feed peaks.csv to Optuna. "
                         "analyse: final report. "
                         "full: local end-to-end with pruning.")
    ap.add_argument("--n",        type=int, default=20,
                    help="Configs per suggest round (default: 20)")
    ap.add_argument("--n-trials", type=int, default=60,
                    help="Total trials for 'full' mode (default: 60)")
    ap.add_argument("--jobs",     type=int, default=8,
                    help="Parallel workers for 'full' mode (default: 8)")
    ap.add_argument("--omp",      type=int, default=None,
                    help="OMP_NUM_THREADS per run (default: cpu_count // jobs)")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--top",      type=int, default=5)
    ap.add_argument("--base",     default=None, help="Override base JSON")
    ap.add_argument("--exe",      default=None, help="Override executable")
    return ap.parse_args()


def main() -> None:
    if not HAS_OPTUNA:
        print("ERROR: optuna not installed.  pip install optuna", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    omp  = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)

    for task in args.task:
        td          = TASK_DEFAULTS[task]
        executable  = args.exe  or td["executable"]
        base_config = args.base or td["base_config"]
        fidelity    = td["fidelity"]

        print(f"\n{'='*66}")
        print(f"  Task: {task}   mode: {args.mode}   seed: {args.seed}")
        print(f"{'='*66}")

        if args.mode == "suggest":
            cmd_suggest(task, args.n, fidelity, args.seed)
        elif args.mode == "observe":
            cmd_observe(task)
        elif args.mode == "analyse":
            cmd_analyse(task, args.top)
        elif args.mode == "full":
            print(f"  n-trials: {args.n_trials}   parallel: {args.jobs}   OMP/run: {omp}")
            cmd_full(task, args.n_trials, fidelity, base_config,
                     executable, args.jobs, omp, args.seed, args.top)


if __name__ == "__main__":
    main()

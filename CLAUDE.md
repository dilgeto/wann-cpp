# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

C++17 implementation of Weight Agnostic Neural Networks (WANN, NEAT-style topology
evolution with shared/scalar weights) driving Spiking Neural Network (SNN) controllers
on classic-control RL tasks (Acrobot, MountainCar, a racing-car track, Pendulum,
L2F/Crazyflie). Python scripts around the C++ binaries do hyperparameter screening,
statistical comparison (bootstrap CIs) against ANN/PPO baselines, and plotting.

## Agent policy — never run training

Claude Code must never itself start a WANN training run — only the user initiates
training. This means never executing `wann_<task>` / `wann_train` directly, and never
running `screening_reduce.py` or `screening_full.py` in a mode that trains (`phase1`,
`phase2`, `phase3`, `both`) — not even a short/small smoke-test run (e.g. `maxGen`
lowered, tiny `popSize`). If verification of a code change is needed, prefer compiling
and, where applicable, the project's non-training tools (`wann_<task>_eval`,
`wann_eval_weights_<task>`, `wann_car_replay`) or ask the user to run and share
results. Training is compute-heavy and expected to run on the user's own machine or
cluster, not be launched by the agent.

## Build

This project depends on a sibling repo, `../snn-simulator`, which must be built first
(it produces `libsnn_core.a`, `libsnn_encoding.a`, `libsnn_engine.a` under
`../snn-simulator/build_cmake`, plus the header-only `rl-tools` under
`../snn-simulator/external/`). Build it via its own `./build.sh` before building here.

```bash
./build.sh            # Release build (default) -> build/
./build.sh Debug       # Debug build
```

Equivalent to: `cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" && cmake --build build -j$(nproc)`.

`nlohmann_json` is picked up from the system package if available, otherwise fetched
via `FetchContent`. `OpenMP` is required (used to parallelise population evaluation
across individuals with `#pragma omp parallel for`).

There is no test suite (no `enable_testing()`/CTest target, no test binaries).
Verification is empirical: run training/eval binaries and inspect the resulting
CSVs/plots.

### Binaries (in `build/`)

Per task `<X>` in `{acrobot, car, mountain_car, disc_mc, l2f}` (plus a generic
`wann_snn` for Pendulum), three binaries share one pattern:

- `wann_<X>` — runs WANN evolution (ask/tell loop) with the SNN task as fitness
  function. `./wann_car [-d p/car_snn.json] [-p overrides.json] [-o out_prefix] [-s seed] [-v]`.
  `-p` accepts a file path or an inline JSON string (e.g. `-p '{"maxGen":1}'`).
- `wann_<X>_eval` — evaluates N episodes with the best saved network.
- `wann_eval_weights_<X>` (acrobot/car/disc_mc only) — loads a saved network and sweeps
  its `alg_nVals` shared-weight values across a list of seeds, printing per-seed CSV
  to stdout; see `eval_results/eval_p3_weights.py` for the orchestrating script. `--reward
  shaped|original` and `--episode-detail --weight-index N` control output.
  The task is fixed at compile time (`EVAL_TASK_ACROBOT`/`_CAR`/`_DISC_MC`) because
  linking two `SnnXxxTask.cpp` translation units together causes multiple-definition
  errors from rl-tools' non-inline symbols — this is why there's one binary per task
  instead of a runtime `--task` flag.
- `wann_car_replay` — runs one episode with the best (or a specific) weight and
  exports the trajectory to CSV for visualization: `./wann_car_replay -f
  log/snn_car_best.out -d p/car_snn.json -w best -s 0`.
- `wann_train` — task-agnostic entry point (no SNN), for the plain WANN algorithm.

**Terminology: "Mountain Car" (unqualified) always means the discrete task** — code
identifier `disc_mc` (Gymnasium `MountainCar-v0`, discrete actions; binaries
`wann_disc_mc`, `wann_eval_weights_disc_mc`, config `p/disc_mc_snn.json`). There is a
second, separate rl-tools *continuous* MountainCar task (code identifier
`mountain_car`; binary `wann_mountain_car`, config `p/mountain_car_snn.json`) that
also exists in this repo — always refer to it explicitly as "continuous Mountain Car"
to avoid ambiguity, never as plain "Mountain Car". This matches the convention
`eval_results/eval_p3_weights.py` already uses for its `--task mountain_car` flag (which maps to
the `disc_mc` binary/config internally — see the comment above its `TASKS` dict).

### Screening / evaluation pipeline (Python, orchestrates the C++ binaries)

Three-phase pipeline per (task, encoder, decoder) combination, run via `run_all.sh`
(or invoke phases individually):

```bash
python screening_reduce.py --task car --encoder ttfs --decoder first_spike \
    --rounds 4 --n 30 --jobs 8 --omp 64        # Phase 1: narrow hyperparam space
python screening_full.py --task car --encoder ttfs --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 11 --jobs 3 --omp 190   # Phase 2+3: full runs
python eval_results/eval_p3_weights.py --task car --seeds 11   # sweep shared weights over seeds
bash generate_graphs.sh car                              # training curves, Pareto front, topology plots
python bootstrap_results/bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike  # bootstrap CI vs ANN/PPO
```

`JOBS_*`/`OMP_*` env vars in `run_all.sh` control parallel process count vs. OpenMP
threads per process — tune jointly against available cores, they multiply.

Results land in per-combination directories: `screening_reduce/<run_key>/`,
`screening_full/<run_key>/` (includes `p3_configs/rank<NN>_seed<NN>.json`),
`eval_results/<task>_best.csv`, `log/full_p3_<run_key>/rank<NN>_seed<NN>_*`,
`bootstrap_results/<run_key>/`, `graficos/<Tarea>/`.

## Architecture

**Evolution core** (`src/Wann.cpp`, `include/wann/Wann.h`): ask/tell evolutionary
algorithm operating on `std::vector<Ind>`. `ask()` returns the current population for
the caller to evaluate; `tell(reward)` takes a `reward[i][j]` matrix (individual i,
shared-weight-value j) and drives selection/speciation/mutation for the next
generation. WANN uses a single species. Mutation operators (`mutAddConn`,
`mutAddNode`, `mutToggleExcitatory`) and `crossover`/`topoMutate` are private; NSGA-II
style multi-objective sorting lives in `src/NsgaSort.cpp`.

**Genome** (`include/wann/Ind.h`): `NodeGene` (input/output/hidden/bias + activation
index 1-11) and `ConnGene` (src/dst node id, weight — NaN when disabled to preserve
structure across mutations —, `enabled`, `excitatory`). `InnovRecord` tracks
structural innovations NEAT-style so identical topological mutations across the
population share the same innovation number.

**Task interface** (`include/wann/Task.h`, `ITask`): the one method that matters is
`getDistFitness(wVec, aVec, seed) -> vector<double>`. This is the "weight agnostic"
part — instead of evaluating one trained weight per connection, each topology is
tested with `alg_nVals` different *shared* scalar weight values (same scalar applied
to every enabled connection), averaged over `alg_nReps` repetitions; fitness is
reported per weight value, not collapsed to one number, so evolution can favor
topologies that work well across the whole weight distribution.

**Per-task adapters** (`src/Snn<Task>Task.cpp` / `include/wann/Snn<Task>Task.h`):
each implements `ITask` by bridging a WANN genome to the SNN simulator
(`../snn-simulator`, encoder → spiking network → decoder) driving an rl-tools or
Gymnasium environment. `snn_encoder` (`current`/`poisson`/`rate`/`ttfs`/`ttfs_log`/
`small`/`large`) turns continuous observations into spikes; `snn_decoder`
(`spike_count`/`rate`/`first_spike`/`voting`/`rate_argmax`) turns output spikes back
into an action. `snn_reset_between_steps` toggles whether membrane state persists
across env steps (implicit recurrence) or resets each step.

**Config** (`include/wann/Hyperparams.h`): all algorithm/task/SNN hyperparameters live
in one struct, loaded from JSON under `p/*.json` (one base config per task, e.g.
`p/car_snn.json`) and mergeable with a second override JSON (file path or inline
`{"key":val}` string) — the pattern used everywhere is `-d base.json -p
overrides.json`. Screening scripts generate override JSONs on the fly to sweep this
struct's fields. `early_stop_patience` (int, default `0` = disabled) stops training
once that many generations pass with no new `fitTop` record (running-best elite
fitness) — set it per JSON like any other hyperparameter. Currently only wired into
`src/main_car.cpp`'s generation loop (not the other tasks' `main_*.cpp`).

`snn_window_ms`/`snn_tau_exc`/`snn_tau_inh`/`snn_ttfs_threshold` are SNN-simulator
microparameters (`ms` per env step given to the SNN before decoding an action; AMPA/
GABA conductance decay time constants; TTFS no-spike cutoff) — **currently wired only
into `SnnCarTask`** (`snn_window_ms` replaced what used to be the compile-time
`WANN_CAR_SIM_WINDOW_MS` macro; the other `Snn<Task>Task.cpp` files still hardcode
their own `SIM_WINDOW_MS` and never touch `tau_exc`/`tau_inh`/TTFS threshold at all).
Do not add `DT` to a search alongside `snn_window_ms` (the FIRST_SPIKE decoder's
quantization step is `2*dt/window` — the two are collinear for that effect; `dt` stays
a fixed `1.0` in `SnnCarTask.cpp`), and do not add TTFS's `t_max_ratio` alongside
`snn_window_ms` either (`t_max = (window-dt)*t_max_ratio` is the same kind of product;
`t_max_ratio` stays fixed at `1.0`, not exposed as a hyperparameter).

**Logging** (`src/DataGatherer.cpp`): appends per-generation population statistics,
writes `<prefix>_best.gen`/`.wi` (best individual + which shared-weight index won)
and Pareto snapshots consumed by `graph.py`/`graph_network.py`. Mirrors a Python
`dataGatherer.py` from an earlier WANN implementation this project descends from.

## Directory map

- `include/wann/`, `src/` — the C++ library (`wann_lib`) and per-binary `main_*.cpp`.
- `p/` — base Hyperparams JSON configs, one per task.
- `log/` — training run outputs (`*_best.gen/.wi`, replay CSVs, per-run subdirs
  `full_p2_*`, `full_p3_*`, `reduce_*` matching screening `run_key`s).
- `screening_reduce/`, `screening_full/` — phase 1 / phase 2+3 screening outputs.
- `eval_results/` — holds both `eval_p3_weights.py` (the shared-weight sweep
  script — anchors its CWD to the repo root at startup, same as the
  `bootstrap_results/*.py` scripts, so it behaves identically run from the repo
  root, from inside `eval_results/`, or by absolute path) and its output:
  `<task>_best.csv`/`<task>_best_episodes.csv` at the top level when run without
  `--run-key` (sweeps every run_key of a task into one file), or nested under
  `eval_results/<run_key>/` when the revalidation step of a
  `bootstrap_results/bootstrap_compare_{car,acrobot,mountain_car}_auto.py`
  script calls it for one specific run_key (override with `--out-dir`/
  `--eval-out-dir`). Older ad-hoc `eval_p3_weights_*/` dirs at repo root
  predate this convention.
- `bootstrap_results/` — holds both the bootstrap comparison scripts
  (`bootstrap_compare_lib.py`, `bootstrap_auto_lib.py`, the `*_auto.py` scripts
  for car/acrobot/mountain_car, `bootstrap_compare_acrobot_ppo_experiment.py`,
  `bootstrap_compare.py`, `bootstrap_two_sample.py`, `replot_bootstrap.py`) and
  their output (`bootstrap_results/<run_key>/`, one subdir per comparison —
  `rewards.csv`, `bootstrap_samples.csv`, `summary_stats.csv`, `ci_results.csv`,
  a `.png`). Every script there anchors its CWD and `sys.path` to the repo root
  at startup (see the `_REPO_ROOT = Path(__file__).resolve().parent.parent`
  block near the top of each), so they behave identically whether invoked from
  the repo root, from inside `bootstrap_results/`, or by absolute path.
- `graficos/`, `plots/` — generated figures, organized by task.
- `lib/` — vendored front-end JS assets (`vis-9.1.2`, `tom-select`) unrelated to the
  C++/Python pipeline — not part of the WANN codebase proper.

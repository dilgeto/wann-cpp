# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

C++17 implementation of Weight Agnostic Neural Networks (WANN, NEAT-style topology
evolution with shared/scalar weights) driving Spiking Neural Network (SNN) controllers
on classic-control RL tasks (Acrobot, MountainCar, a racing-car track, Pendulum,
L2F/Crazyflie, BipedalWalker Hardcore). Python scripts around the C++ binaries do hyperparameter screening,
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

Per task `<X>` in `{acrobot, car, mountain_car, disc_mc, l2f, bipedal}` (plus a generic
`wann_snn` for Pendulum), three binaries share one pattern:

- `wann_<X>` — runs WANN evolution (ask/tell loop) with the SNN task as fitness
  function. `./wann_car [-d p/car_snn.json] [-p overrides.json] [-o out_prefix] [-s seed] [-v]`.
  `-p` accepts a file path or an inline JSON string (e.g. `-p '{"maxGen":1}'`).
- `wann_<X>_eval` — evaluates N episodes with the best saved network.
- `wann_eval_weights_<X>` (acrobot/car/disc_mc/bipedal only) — loads a saved network and sweeps
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
- `wann_car_neighborhood` — evaluation-only neighbourhood analysis: enumerates every
  single-mutation neighbour (N1) of an individual, evaluates them with the parent's
  seeds and reports better/neutral/worse per operator (optional sampled N2 and a greedy
  climb). Improving neighbours are re-evaluated with fresh seeds (`--confirm`) before being
  called improvements, to avoid the winner's curse of a single noisy screen. Input is a `snapshot_interval` population snapshot or a legacy `*_best.out`
  (genome reconstructed, approximate). See `BINARIOS.md` §4b. `Neighborhood.cpp`
  re-implements `Wann`'s private mutation operators to enumerate them — keep the two in
  sync if either changes.
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

`screening_full.py` in `phase3`/`both` mode automatically finishes by running the
evaluation + bootstrap step (`bootstrap_results/bootstrap_compare_<task>_auto.py
--run-key <run_key>`: shared-weight revalidation via `eval_p3_weights.py`, winner
selection, bootstrap CI vs. ANN/PPO) for `car`, `acrobot` and `disc_mc` (the latter
maps to `mountain_car` in the bootstrap scripts); other tasks skip it. It never trains,
and a failure there is non-fatal (the error prints the command to retry). Opt out with
`--no-post-eval`; tune with `--eval-seeds/--eval-nreps/--eval-jobs/--eval-omp/
--bootstrap-n/--bootstrap-resamples`. So the manual `eval_p3_weights.py`/`*_auto.py`
lines above are only needed for re-runs.

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
style multi-objective sorting lives in `src/NsgaSort.cpp`. `nsga_sort()` returns a
*strict* per-individual rank (crowding distance flattens each front into a total
order, so no two individuals ever share a rank).

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
(`spike_count`/`rate`/`first_spike`/`voting`/`rate_argmax`/`population_vector`) turns
output spikes back into an action. `snn_reset_between_steps` toggles whether
membrane state persists across env steps (implicit recurrence) or resets each step.

`population_vector` (**`SnnCarTask` only**) is a classical population-vector decode,
not PopSAN's literal learned linear readout — that would need per-connection trained
decoder weights, which doesn't fit WANN's weight-agnostic (shared-scalar-only)
design. Each of the two actions (throttle, steering) is decoded from a population of
`snn_neurons_per_var` output neurons (so `ann_nOutput` must be `2 * snn_neurons_per_var`,
not `2`); each neuron in a population has a **fixed** preferred value uniformly spaced
in `[-1,1]` (`populationVectorDecode` in `SnnCarTask.cpp`), and the action is the
spike-count-weighted average of preferred values (no spikes at all in a population
falls back to `0.0`). Screening scripts auto-set `ann_nOutput` via
`decoder_nOutput()` in `screening_reduce.py` when `--decoder population_vector` is
passed (mirrors `encoder_nInput()` for the `small`/`large` population-coding
encoders) — for a manual run, set `ann_nOutput` yourself in the override JSON.
`main_odin_export.cpp`'s firmware export path forwards `ann_nOutput` generically but
has not been checked against a population-decoded network — verify before exporting
one.

**BipedalWalker** (`SnnBipedalTask`, binaries `wann_bipedal*`, config
`p/bipedal_snn.json`): the Box2D-free C++ port in `../snn-simulator/include/rl/environments/
bipedal_walker/` (header-only, float32 physics as validated against Box2D there). Hardcore
by default; `bipedal_hardcore=false` switches to plain BipedalWalker-v3 (same walker/reward,
easier terrain — a curriculum stage). `bipedal_max_steps` (0 = env limit, 2000/1600) caps
episode length to cut cost; truncated rewards are not comparable to full-length ones.
Terrain is resampled per episode from the episode seed. 24 obs → 4 torques; `population_vector`
needs `ann_nOutput = 4 * snn_neurons_per_var`, `small`/`large` need `ann_nInput = 48` /
`24 * snn_neurons_per_var` (the constructor throws otherwise; screening scripts set them via
`TASK_DEFAULTS["bipedal"]["n_actions"]`). With `first_spike` a silent output means zero torque,
not −1. No `bootstrap_compare_*_auto.py` for it yet (screening_full skips that step).

**Config** (`include/wann/Hyperparams.h`): all algorithm/task/SNN hyperparameters live
in one struct, loaded from JSON under `p/*.json` (one base config per task, e.g.
`p/car_snn.json`) and mergeable with a second override JSON (file path or inline
`{"key":val}` string) — the pattern used everywhere is `-d base.json -p
overrides.json`. Screening scripts generate override JSONs on the fly to sweep this
struct's fields. Every non-integer swept hyperparameter is limited to 2 decimals
(`HP_DECIMALS` in `screening_reduce.py`): Optuna samples on a 0.01 grid (`step=0.01`, so
no log-uniform sampling) and reduced-space bounds are snapped to it — hence
`prob_enable` and `snn_ttfs_threshold` now start at `0.01` instead of `0.005`/`1e-6`. `early_stop_patience` (int, default `0` = disabled) stops training
once that many generations pass with no new `fitTop` record (running-best elite
fitness) — set it per JSON like any other hyperparameter. Currently only wired into
`src/main_car.cpp`'s generation loop (not the other tasks' `main_*.cpp`).
`snapshot_interval` (int, default `0` = off) is likewise `main_car.cpp`-only: it writes
`log/<prefix>_lineage.csv` every generation and full-population genome snapshots to
`log/<prefix>_snap/` (consumed by `wann_car_neighborhood`); pure bookkeeping, no RNG use.

`snn_window_ms`/`snn_tau_exc`/`snn_tau_inh`/`snn_ttfs_threshold` are SNN-simulator
microparameters (`ms` per env step given to the SNN before decoding an action; AMPA/
GABA conductance decay time constants; TTFS no-spike cutoff) — **currently wired only
into `SnnCarTask` and `SnnBipedalTask`** (`snn_window_ms` replaced what used to be the compile-time
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
- `diagnostico/` — Python diagnostics for stagnation / seed dependence, built on the
  `snapshot_interval` outputs and `wann_car_neighborhood` (lineage + coalescence, early
  prediction, batch launcher, neighbourhood summary). `run_neighborhood.py` only prints
  a plan unless given `--run`. See `BINARIOS.md` §4c.
- `graficos/`, `plots/` — generated figures, organized by task.
- `lib/` — vendored front-end JS assets (`vis-9.1.2`, `tom-select`) unrelated to the
  C++/Python pipeline — not part of the WANN codebase proper.

## Known issues to review (found, not fixed)

Two quirks in the evolution core, confirmed against the real `Wann` code but left
unchanged on purpose: fixing either changes the algorithm, so past runs would become a
different experimental condition. Decide deliberately, ideally behind a hyperparameter
that defaults to the current behaviour. If the user says they are looking for bugs or
errors in the code, remind them of these and pick this up again.

1. **`mutAddConn` never wires the deepest hidden layer to an output** (`src/Wann.cpp`,
   `lastLayer`). Outputs get the same layer as the deepest hidden layer
   (`lastLayer = max(hLay0)+1`, where `hLay0` is 0-based and hidden nodes use `hLay0+1`),
   and destinations must be in a *strictly higher* layer. So with one hidden layer a
   hidden node can never be connected to another output by `addConn` (only `addNode`
   can wire it to an output). Probably an off-by-one from the initial literal port
   (present since the first commit; the original Python was not available to compare —
   from memory it used `max(hLay)+1` with `hLay` already shifted). Size on the saved
   `car` elites: ~1 blocked pair per net (median) out of ~530 valid `addConn` pairs, but
   source-first sampling makes that class ~3 % of `addConn` draws (median, max 24 %).
   Unlikely to explain stagnation by itself. A fix would be outputs one layer deeper;
   `src/Neighborhood.cpp` mirrors this rule and must follow. Open question: whether the
   blocked moves are better than typical ones (a partial evaluation was started and
   interrupted, not analysed).
2. **Disabled hidden→hidden connections come back enabled in the expressed network.**
   `getNodeOrder` (`src/Ind.cpp`) binarises the hidden block with `copysign`, and
   `copysign(1, NaN) = +1`, so a disabled hidden→hidden gene shows up as `+1` in
   `wMat`/`wVec`, in `*_best.out`, and in everything built from it (eval, replay,
   `eval_p3_weights.py`, bootstrap, ODIN export). Training is unaffected: every task's
   `evalPop` uses `buildNetwork(const Ind&)`, which reads the genes. Disabled
   input→hidden and hidden→output genes are kept as `nan`, so only the hidden block is
   hit. On the `car_ttfs_*` elites (geometry: 9 inputs, 2 outputs), 45 of 66 nets had at
   least one such revived edge (lower bound: only detected as a "pure relay bypass"), but
   correcting them moved fitness by a median 0.04 seed-sd (none > 1 sd, symmetric), so
   past results look safe. It matters before ODIN deployment (extra synapses) and for
   consistency. Fix in `Ind::express`, not in `getNodeOrder` (`mutAddConn`'s layering
   reads that block, so changing it alters the operator and can create cycles). Beware
   `nConn` (drives the 1/nConn objective in `probMoo`): correcting the matrix lowers it
   and slightly changes selection. `genomeFromNetFile` (`GenomeIO.h`) inherits the same
   limitation.

## Stagnation diagnosis and candidate interventions (analysed, not implemented)

`diagnostico/RESUMEN_DIAGNOSTICO.md` documents a full diagnosis (linaje/coalescencia,
predicción temprana, control positivo, neutralidad) of why `wann_car` training stagnates
and depends so heavily on the seed, on 30 runs of the `car_ttfs_first_spike`/`rank02`
config. Headline finding: by the late generations ~89% of single-mutation neighbours
(N1, via `wann_car_neighborhood`) don't change behaviour at all, yet the genome keeps
growing regardless — inert structural accumulation, not a search or detection problem
(verified with a positive control: the same tool *does* find real improving neighbours
in early generations, where the run's outcome is actually still being decided).

A first candidate fix (Option A: disable, at a configurable interval, every enabled
connection with no structural path from an input to an output — pure graph reachability,
no simulation) was implemented, verified correct and fully toggleable, then **reverted**
(2026-09-28, not in the repo) after testing against ~87,000 real evolved connections
(4 population snapshots up to gen 1023) and fresh training runs up to 300 generations:
it never found anything to prune. Reasoned from the operators themselves and confirmed
empirically: `mutAddNode` always inserts its new node into an already-live path, and no
operator disables a previously-live edge except that same self-preserving split — so a
connection that's enabled but structurally disconnected from the input-output flow
appears to be unreachable as a genome state under `Wann`'s current 5 operators. The
~89% neutrality measured is real but *behavioural* (spiking dynamics / decoder
saturation / timing), not structural, and needs simulation to detect — which is exactly
what Options B and C below would do differently.

One further candidate intervention was designed but **not implemented yet** — pick this
up deliberately, behind a hyperparameter that defaults to current behaviour, same
convention as `early_stop_patience`/`snapshot_interval`:

- **Option B — DISCARDED (2026-09-29), not just "more invasive": proven a no-op.**
  The idea was to change what `probMoo`'s `1/nConn` objective (`src/Wann.cpp`) counts —
  only connections on some input-to-output path, instead of every enabled connection —
  to push selection toward less structural bloat. It relies on the exact same
  reachability definition as Option A. Since Option A's empirical test already showed
  that under `Wann`'s 5 operators an enabled connection is *never* structurally
  unreachable (0 of ~87,000 real evolved connections), "connections on an input-to-output
  path" and "every enabled connection" are the same set for any genome this codebase can
  produce — so the two objectives would compute the exact same `nConn` value, always.
  Redefining it this way cannot change a single selection decision Wann ever makes; it is
  not a smaller/weaker version of Option A, it is mathematically identical to not
  implementing it at all. Reviving this would need a genuinely different notion of "live"
  (e.g. weighted by behavioural redundancy — see Option C), not a structural one.
- **Option C — prune behaviourally-neutral connections.** Periodically (e.g. every N
  generations), for each individual, test whether disabling a connection changes fitness
  under simulation (paired seed, like `wann_car_neighborhood`'s N1 screening), and disable
  ones that don't. Targets the *actual* neutrality measured (behavioural, not structural),
  unlike A. Costs extra simulation per generation — cost/benefit not yet estimated.

A third idea (Option D, from the same discussion, distinct from B/C): instead of trying
to reduce stagnation, detect it cheaply and reliably. `early_stop_patience` currently
stops training after a fixed K generations with no new `fitTop` record, with no evidence
that the plateau is a genuine local optimum rather than bad luck — real median stagnation
length was ~440 generations (of 1023) across the 30 diagnosed runs. On triggering the
existing patience condition, run a cheap subset-of-N1 check (`wann_car_neighborhood
--max-per-op N --confirm ...`, e.g. ~30 neighbours of the elite, confirmed) instead of
stopping outright: if nothing confirms, stop with actual evidence it's a true local
optimum; if something does, keep training. Estimated overhead: ~0.2% of total training
compute (a ~1,000-episode check every ~50 generations against a ~17,000-episode/generation
run). Doesn't create new fitness — it only saves the compute otherwise spent waiting out
a confirmed-empty plateau.

Update (2026-09-29): checked the gap (generations 150/200/250/300/350, `log/
diag_neighborhood_mid/`) that was missing between the dense-signal window (25-100) and
the empty one (400/1023). It's not a clean cutoff, it's a regime change: dense,
strongly-significant confirmations through ~gen 100, then from 150 on confirmations
become rare (5 of 42 run×generation combinations in the alto group had any) and, where
they exist, small (0.34%-2.8% of the parent's fitness) — never absent, never large.
A stopping-check at, say, every 50 generations past gen ~100-150 would correctly find
"nothing" most of the time and occasionally a real small candidate — see
`diagnostico/RESUMEN_DIAGNOSTICO.md` §4 for the numbers (includes a concrete case:
`seed22`, which only escapes its plateau around gen ~680, already had confirmed
~0.35%-effect neighbours at gen 150 that evolution took hundreds of generations to
exploit).

- **Option E — random immigrants, IMPLEMENTED (2026-09-29), not yet trained/tested.**
  Distinct from B/C/D: doesn't rescue an already-stuck lineage (a fresh random
  individual competing against an already-evolved network with 50+ hidden nodes would
  essentially never win a tournament — pointless once a run has already stagnated),
  it targets a different local optimum: which of the 480 starting genomes wins the
  premature-convergence race. Measured: a single generation-0 founder dominates the
  whole population by a median of generation 14 (`diagnostico/lineage_analysis.py`),
  and that founder's own rank in generation 0 does NOT predict the final result (ρ ≈
  0.03, not significant) — consistent with an early, largely arbitrary bottleneck,
  not early merit. This is Grefenstette's "random immigrants" (1992), a known
  countermeasure for premature convergence (see also ALPS, Hornby 2006); the
  founder-count trigger (adaptive, not a fixed generation cutoff) is specific to this
  project, reusing the same founder-counting logic `diagnostico/diag_lib.py`'s
  `founders()` already computed post-hoc, now live in `Wann.cpp`.

  Mechanism: `immigrant_fraction` (double) and `immigrant_min_founders` (int), both 0
  by default. While the population's current distinct-founder count is below
  `immigrant_min_founders`, each generation replaces `immigrant_fraction` of the
  non-elite offspring with brand-new random individuals (`Wann::randomBaseIndividual`,
  the same construction as generation 0) instead of tournament+mutation children —
  recorded in the lineage with `op == -2`, `parent == parentB == -1` (no parent).
  Lives in `Wann.cpp`/`Wann.h` itself (`recombine`/`evolvePop`, plus the extracted
  `baseGenome`/`randomBaseIndividual`), not in `main_car.cpp`, so — unlike
  `early_stop_patience`/`snapshot_interval`/the reverted Option A — it applies to
  every `wann_<task>` binary, not just `wann_car`. Essentially free in compute:
  immigrants are evaluated exactly like any other individual in that generation's
  `evalPop()`, so this never adds simulated episodes, only changes which genomes
  occupy some population slots.

  Verified: compiles clean across every target; with both hyperparameters at 0 (or
  omitted), the RNG draw deciding "is this child an immigrant" and the founder-count
  bookkeeping are both skipped entirely (never evaluated-and-discarded), so `stats.out`
  for a fixed seed is byte-for-byte identical to before this existed. `diag_lib.py`'s
  `founders()` was updated to recognise `op == -2` as a fresh founder rather than
  mis-indexing `parent == -1` as "the last individual of the previous generation"
  (verified against a hand-built synthetic lineage with an immigrant) — `mrca_gen()`
  has the same latent bug, **not yet fixed**, don't trust it on a run with immigrants
  enabled until it is (immigrant-free runs, i.e. everything diagnosed so far, are
  unaffected either way).

  Not verified: whether it actually changes training outcomes. That needs a real
  multi-seed training comparison (with vs. without), which — per the project's agent
  policy — the user runs, not the agent. Two things left unresolved even before that:
  sensible defaults for `immigrant_fraction`/`immigrant_min_founders` were discussed
  (~0.05-0.1 / ~20-50) but not settled or tested; and there's a live, unresolved
  alternative hypothesis this design does not itself distinguish from "premature
  convergence wastes a good starting bet" — that early lineages are roughly
  fungible and the real bottleneck is something that happens *after* arriving at the
  productive window (gen ~25-100) regardless of which lineage gets there, in which
  case giving the population more early candidates to choose from wouldn't move the
  outcome much. Only the real experiment adjudicates between these.

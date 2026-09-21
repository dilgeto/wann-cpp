#!/usr/bin/env bash
# Same as run.sh, but with lexicographic_parsimony_epsilon=1.0 instead of the
# strict exact-tie default (0.0) — the strict version turned out to be all but
# inert past ~gen 100 in this continuous-reward task (exact float ties on
# meanFit become vanishingly rare once the population is mostly functional).
# Also produces log/<prefix>_lpp_ties.csv (gen,tieCount,popSize) per run, so
# you can check whether widening epsilon actually keeps the tie-break firing
# later in evolution, not just assume it from the complexity curves.
#
# Usage (from the repo root):
#   ./experiments/car_lpp_repro/run_eps1.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

BASE_CFG="p/car_snn.json"
OVERRIDE="experiments/car_lpp_repro/override_lpp_eps1.json"
SEEDS=(0 100 200)   # matches screening_full.py's rank*10000 + si*100 for rank=0

for seed in "${SEEDS[@]}"; do
    prefix="repro_car_best_lpp_eps1_s${seed}"
    echo "=== seed ${seed} -> log/${prefix}_* ==="
    ./build/wann_car -d "$BASE_CFG" -p "$OVERRIDE" -o "$prefix" -s "$seed"
done

#!/usr/bin/env bash
# Re-runs the car_ttfs_first_spike screening's best config (rank00 from
# screening_full/car_ttfs_first_spike/p2_results.csv) with
# lexicographic_parsimony=true added on top, 3 times with different seeds.
#
# Usage (from the repo root):
#   ./experiments/car_lpp_repro/run.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

BASE_CFG="p/car_snn.json"
OVERRIDE="experiments/car_lpp_repro/override_lpp.json"
SEEDS=(0 100 200)   # matches screening_full.py's rank*10000 + si*100 for rank=0

for seed in "${SEEDS[@]}"; do
    prefix="repro_car_best_lpp_s${seed}"
    echo "=== seed ${seed} -> log/${prefix}_* ==="
    ./build/wann_car -d "$BASE_CFG" -p "$OVERRIDE" -o "$prefix" -s "$seed"
done

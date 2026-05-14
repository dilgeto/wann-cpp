#!/bin/bash
# screening.sh  –  Local end-to-end Bayesian screening (requires Python + compiled exes).
#
# For cluster use (no Python on cluster), use the round-based workflow:
#   python screening.py --task TASK --mode suggest --n 20
#   rsync / run on cluster / rsync back
#   python screening.py --task TASK --mode observe
#   (repeat rounds, then:)
#   python screening.py --task TASK --mode analyse
#
# Usage:
#   bash screening.sh [--task TASK] [--n-trials N] [--jobs J] [--omp M] [--seed S]

set -euo pipefail

JOBS=8
OMP=""
N_TRIALS=60
SEED=0
TASKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)     JOBS="$2";     shift 2 ;;
        --omp)      OMP="$2";      shift 2 ;;
        --n-trials) N_TRIALS="$2"; shift 2 ;;
        --seed)     SEED="$2";     shift 2 ;;
        --task)     TASKS+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ ${#TASKS[@]} -eq 0 ]] && TASKS=(mountain_car acrobot car)

echo "==> Compiling..."
cmake --build build -j"$(nproc)"
echo ""

OMP_FLAG=""
[[ -n "$OMP" ]] && OMP_FLAG="--omp $OMP"

for task in "${TASKS[@]}"; do
    echo "==> Task: $task   n-trials=${N_TRIALS}   jobs=${JOBS}"
    python screening.py \
        --task "$task" \
        --mode full \
        --n-trials "$N_TRIALS" \
        --jobs "$JOBS" \
        $OMP_FLAG \
        --seed "$SEED"
done

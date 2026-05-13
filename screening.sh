#!/bin/bash
# screening.sh  –  Local end-to-end screening (requires Python + compiled exes).
#
# For cluster use (no Python), run instead:
#   python screening.py --task TASK --mode generate-only --n 64 --seed 42
#   # transfer configs, run screening_run.sh on cluster, transfer peaks.csv back
#   python screening.py --task TASK --mode analyse-only
#
# Usage:
#   bash screening.sh [--task TASK] [--n N] [--jobs J] [--omp M] [--seed S]
#
# Defaults: all tasks, 64 configs, 16 parallel jobs, OMP = cpu_count/jobs

set -euo pipefail

JOBS=16
OMP=12
N=64
SEED=42
TASKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --omp)  OMP="$2";  shift 2 ;;
        --n)    N="$2";    shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --task) TASKS+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ ${#TASKS[@]} -eq 0 ]] && TASKS=(mountain_car acrobot car)

echo "==> Compiling..."
cmake --build build -j"$(nproc)"
echo ""

TASK_FLAGS=()
for t in "${TASKS[@]}"; do TASK_FLAGS+=(--task "$t"); done

echo "==> Screening: tasks=${TASKS[*]}  n=${N}  jobs=${JOBS}  omp/run=${OMP}"
python screening.py \
    "${TASK_FLAGS[@]}" \
    --mode full \
    --n    "$N"    \
    --jobs "$JOBS" \
    --omp  "$OMP"  \
    --seed "$SEED"

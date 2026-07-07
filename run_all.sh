#!/bin/bash
# Ejecuta el screening completo (Phase 1 reduce → Phase 2+3 full) para todas
# las combinaciones tarea/encoder/decoder.
#
# Variables de entorno configurables:
#   JOBS_REDUCE   hilos paralelos en Phase 1  (default 8)
#   OMP_REDUCE    threads OMP por proceso en Phase 1 (default 64)
#   JOBS_FULL     hilos paralelos en Phase 2+3 (default 3)
#   OMP_FULL      threads OMP por proceso en Phase 2+3 (default 190)
#
# Uso:
#   bash run_all.sh
#   JOBS_REDUCE=4 OMP_REDUCE=128 bash run_all.sh

set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}

run_combo() {
    local task=$1 enc=$2 dec=$3
    echo "=== $task | $enc | $dec ==="
    python screening_reduce.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --rounds 4 --n 30 \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"
    python screening_full.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --mode both --n 20 --top 3 --seeds 11 \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
}

# ── Acrobot (acciones discretas: first_spike, wta) ────────────────────────────
for enc in ttfs small; do
    for dec in first_spike wta; do
        run_combo acrobot "$enc" "$dec"
    done
done

# ── Mountain Car discreto (acciones discretas: first_spike, wta) ──────────────
for enc in ttfs small; do
    for dec in first_spike wta; do
        run_combo disc_mc "$enc" "$dec"
    done
done

# ── Racing Car (acción continua: first_spike, rate) ───────────────────────────
for enc in ttfs small; do
    for dec in first_spike rate; do
        run_combo car "$enc" "$dec"
    done
done

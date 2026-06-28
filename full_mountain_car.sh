#!/bin/bash
set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}

# Fase 1: reducción del espacio (fidelidad baja: maxGen=64, popSize=64)
for enc in ttfs small; do
    for dec in first_spike rate; do
        python screening_reduce.py --task mountain_car \
            --encoder "$enc" --decoder "$dec" \
            --rounds 4 --n 30 \
            --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"
    done
done

# Fases 2+3: búsqueda completa y validación (fidelidad del JSON base)
for enc in ttfs small; do
    for dec in first_spike rate; do
        python screening_full.py --task mountain_car \
            --encoder "$enc" --decoder "$dec" \
            --mode both --n 20 --top 3 --seeds 11 \
            --jobs "$JOBS_FULL" --omp "$OMP_FULL"
    done
done

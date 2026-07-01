#!/bin/bash
set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}

# Fase 1: reducción del espacio (fidelidad baja: maxGen=128, popSize=96)
for enc in ttfs small; do
    python screening_reduce.py --task acrobot \
        --encoder "$enc" --decoder first_spike \
        --rounds 4 --n 30 \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"
done

# Fases 2+3: búsqueda completa y validación (fidelidad del JSON base)
for enc in ttfs small; do
    python screening_full.py --task acrobot \
        --encoder "$enc" --decoder first_spike \
        --mode both --n 20 --top 3 --seeds 11 \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
done

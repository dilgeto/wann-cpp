#!/bin/bash
# Búsqueda de hiperparámetros de Acrobot (Fase 1 reduce -> Fase 2+3 full) con el
# NUEVO rango de pesos compartidos {0.5, 1, 2, 3, 5, 8} (igual que Mountain Car y
# Racing Car; antes Acrobot usaba {0.5, 1, 1.5, 2, 2.5, 3}, ver SnnAcrobotTask.cpp).
# Derivado de run_optuna_discrete.sh (que queda intacto).
#
# Las 4 combinaciones por defecto: encoder {ttfs, small} x decoder {first_spike, rate_argmax}.
#
#   - mismo espacio de búsqueda que car_ttfs_first_spike_discrete: INITIAL_SPACE de
#     screening_reduce.py, grilla de 2 decimales; no se pasa --space-override
#   - mismos presupuestos: Fase 1 = 4 rondas x 30 trials; Fase 2 = 40 trials;
#     Fase 3 = top 3 x 11 seeds
#   - --tag w8 -> run_key `acrobot_<encoder>_<decoder>_w8`, así no pisa ni reanuda
#     los runs anteriores (acrobot_<enc>_<dec> y acrobot_<enc>_<dec>_discrete),
#     que se entrenaron con la tabla de pesos vieja
#
# Microparámetros SNN (snn_window_ms / snn_tau_exc / snn_tau_inh / snn_ttfs_threshold):
# SnnAcrobotTask los lee solo con snn_micro_active=true, que screening_reduce/full
# activan solos para acrobot, y SÍ se optimizan. snn_ttfs_threshold solo lo lee el
# TTFSEncoder, así que con encoder small se saca del espacio automáticamente
# (space_for_task en screening_reduce.py): 14 dimensiones con ttfs, 13 con small.
#
# Al terminar la Fase 3, screening_full.py corre solo la evaluación de pesos +
# bootstrap vs ANN/PPO (bootstrap_compare_acrobot_auto.py). Eso exige que
# eval_results/eval_p3_weights.py tenga la misma tabla de pesos que el binario
# (ya actualizada).
#
# IMPORTANTE: los binarios deben estar recompilados con la tabla nueva
# (./build.sh) antes de lanzar; si no, entrenarían con el rango viejo.
#
# Variables de entorno configurables (mismos nombres/defaults que run_all.sh):
#   JOBS_REDUCE   procesos paralelos en Fase 1        (default 8)
#   OMP_REDUCE    threads OMP por proceso en Fase 1   (default 64)
#   JOBS_FULL     procesos paralelos en Fase 2+3      (default 3)
#   OMP_FULL      threads OMP por proceso en Fase 2+3 (default 190)
#   ENCODERS      encoders SNN, separados por espacio (default "ttfs small")
#   DECODERS      decoders SNN, separados por espacio (default "first_spike rate_argmax")
#   TAG           sufijo del run_key                  (default w8)
#
# Uso:
#   bash run_optuna_acrobot_w8.sh
#   ENCODERS=small DECODERS=first_spike bash run_optuna_acrobot_w8.sh
#   JOBS_REDUCE=4 OMP_REDUCE=128 bash run_optuna_acrobot_w8.sh
#
# Es reanudable: screening_reduce.py retoma results.csv/space_log.jsonl del
# run_key si existen; volver a lanzar el script continúa donde quedó.

set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}
ENCODERS=${ENCODERS:-"ttfs small"}
DECODERS=${DECODERS:-"first_spike rate_argmax"}
TAG=${TAG:-w8}
TASK=acrobot

# Seguro contra olvidar recompilar: la tabla de pesos nueva termina en 8.0.
if ! grep -q "0.5, 1.0, 2.0, 3.0, 5.0, 8.0" src/SnnAcrobotTask.cpp; then
    echo "ERROR: src/SnnAcrobotTask.cpp no tiene la tabla de pesos nueva." >&2
    exit 1
fi
if [ build/wann_acrobot -ot src/SnnAcrobotTask.cpp ]; then
    echo "ERROR: build/wann_acrobot es más viejo que src/SnnAcrobotTask.cpp; corre ./build.sh antes." >&2
    exit 1
fi

run_combo() {
    local enc=$1 dec=$2
    echo "=== $TASK | $enc | $dec | tag=$TAG ==="

    # Phase 1: reducción del espacio (4 rondas x 30 trials)
    python screening_reduce.py --task "$TASK" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --rounds 4 --n 30 \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"

    # Phase 2+3: 40 trials, top 3 configs, 11 seeds cada una
    python screening_full.py --task "$TASK" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --mode both --n 40 --top 3 --seeds 11 \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
}

for enc in $ENCODERS; do
    for dec in $DECODERS; do
        run_combo "$enc" "$dec"
    done
done

#!/bin/bash
# Runs de Optuna (Fase 1 reduce -> Fase 2+3 full) para Acrobot y Mountain Car
# discreto, replicando la configuración de car_ttfs_first_spike_discrete:
#
#   - mismo espacio de búsqueda: INITIAL_SPACE de screening_reduce.py (14
#     hiperparámetros, grilla de 2 decimales) — no se pasa --space-override
#   - mismos presupuestos: Fase 1 = 4 rondas x 30 trials; Fase 2 = 40 trials;
#     Fase 3 = top 3 x 11 seeds
#   - encoder ttfs; decoders first_spike y rate_argmax (ambas tareas)
#   - --tag discrete -> run_key `<task>_<encoder>_<decoder>_discrete`, así no
#     pisa ni reanuda los runs anteriores (screening_reduce/<task>_<enc>_<dec>)
#
# Al terminar la Fase 3, screening_full.py corre solo la evaluación de pesos +
# bootstrap vs ANN/PPO (bootstrap_compare_{acrobot,mountain_car}_auto.py).
#
# OJO: snn_window_ms / snn_tau_exc / snn_tau_inh / snn_ttfs_threshold forman
# parte del espacio (para ser idénticos a car), pero hoy solo SnnCarTask y
# SnnBipedalTask los leen; Acrobot y Mountain Car discreto usan un
# SIM_WINDOW_MS fijo en compilación, así que esas 4 dimensiones no afectan el
# fitness en estas tareas. Tampoco aplica early_stop_patience (solo main_car).
#
# Variables de entorno configurables (mismos nombres/defaults que run_all.sh):
#   JOBS_REDUCE   procesos paralelos en Fase 1        (default 8)
#   OMP_REDUCE    threads OMP por proceso en Fase 1   (default 64)
#   JOBS_FULL     procesos paralelos en Fase 2+3      (default 3)
#   OMP_FULL      threads OMP por proceso en Fase 2+3 (default 190)
#   ENCODERS      encoders SNN, separados por espacio (default "ttfs")
#   DECODERS      decoders SNN, separados por espacio (default "first_spike rate_argmax")
#   TAG           sufijo del run_key                  (default discrete)
#   TASKS         tareas a correr, separadas por espacio (default "acrobot disc_mc")
#
# Uso:
#   bash run_optuna_discrete.sh
#   TASKS=acrobot bash run_optuna_discrete.sh
#   DECODERS=rate_argmax bash run_optuna_discrete.sh
#   JOBS_REDUCE=4 OMP_REDUCE=128 bash run_optuna_discrete.sh
#
# Es reanudable: screening_reduce.py retoma results.csv/space_log.jsonl del
# run_key si existen; volver a lanzar el script continúa donde quedó.

set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}
ENCODERS=${ENCODERS:-"ttfs"}
DECODERS=${DECODERS:-"first_spike rate_argmax"}
TAG=${TAG:-discrete}
TASKS=${TASKS:-"acrobot disc_mc"}

run_combo() {
    local task=$1 enc=$2 dec=$3
    echo "=== $task | $enc | $dec | tag=$TAG ==="

    # Phase 1: reducción del espacio (4 rondas x 30 trials)
    python screening_reduce.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --rounds 4 --n 30 \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"

    # Phase 2+3: 40 trials, top 3 configs, 11 seeds cada una
    python screening_full.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --mode both --n 40 --top 3 --seeds 11 \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
}

for task in $TASKS; do
    for enc in $ENCODERS; do
        for dec in $DECODERS; do
            run_combo "$task" "$enc" "$dec"
        done
    done
done

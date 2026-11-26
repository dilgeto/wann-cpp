#!/bin/bash
# Búsqueda de hiperparámetros (Fase 1 reduce -> Fase 2+3 full) para Mountain Car
# (discreto, disc_mc) y Racing Car con el rango REDUCIDO de pesos compartidos
# {0.5, 1, 1.5, 2, 2.5, 3}: la tabla que usaba Acrobot antes del 2026-11-23, en vez
# de la tabla compilada {0.5, 1, 2, 3, 5, 8} de estas dos tareas.
#
# Combinaciones (una por tarea):
#   disc_mc  small / rate_argmax   -> run_key disc_mc_small_rate_argmax_w3
#   car      ttfs  / first_spike   -> run_key car_ttfs_first_spike_w3
#
# Cómo se aplica el rango: --weight-vals pasa alg_weight_vals como override fijo
# (ver Hyperparams.h). Queda grabado en cada JSON de trial y en cada
# screening_full/<run_key>/p3_configs/*.json, y wann_eval_weights_<task> lo vuelve a
# leer de ahí (-p), así que la post-evaluación + bootstrap que screening_full.py
# corre al final de la Fase 3 usa la misma tabla. Los .wi de estos runs indexan
# en {0.5..3}, no en {0.5..8}.
#
# Igual que run_optuna_acrobot_w8.sh / run_optuna_discrete.sh:
#   - espacio de búsqueda INITIAL_SPACE de screening_reduce.py (sin --space-override)
#   - Fase 1 = 4 rondas x 30 trials; Fase 2 = 40 trials; Fase 3 = top 3 x 11 seeds
#   - --tag w3 -> run_key propio, no pisa ni reanuda los runs con la tabla hasta 8
#
# IMPORTANTE: los binarios deben estar compilados con soporte de alg_weight_vals
# (./build.sh); el script lo verifica antes de lanzar.
#
# Variables de entorno configurables (mismos nombres/defaults que run_all.sh):
#   JOBS_REDUCE   procesos paralelos en Fase 1        (default 8)
#   OMP_REDUCE    threads OMP por proceso en Fase 1   (default 64)
#   JOBS_FULL     procesos paralelos en Fase 2+3      (default 3)
#   OMP_FULL      threads OMP por proceso en Fase 2+3 (default 190)
#   WEIGHT_VALS   tabla de 6 pesos compartidos        (default 0.5,1,1.5,2,2.5,3)
#   TAG           sufijo del run_key                  (default w3)
#   COMBOS        combinaciones task:encoder:decoder  (default
#                 "disc_mc:small:rate_argmax car:ttfs:first_spike")
#
# Uso:
#   bash run_optuna_w3.sh
#   COMBOS="car:ttfs:first_spike" bash run_optuna_w3.sh      # solo Racing Car
#   JOBS_REDUCE=4 OMP_REDUCE=128 bash run_optuna_w3.sh
#
# Es reanudable: screening_reduce.py retoma results.csv/space_log.jsonl del
# run_key si existen; volver a lanzar el script continúa donde quedó.

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}
WEIGHT_VALS=${WEIGHT_VALS:-"0.5,1,1.5,2,2.5,3"}
TAG=${TAG:-w3}
COMBOS=${COMBOS:-"disc_mc:small:rate_argmax car:ttfs:first_spike"}

# Seguro contra binarios viejos: sin soporte de alg_weight_vals entrenarían
# en silencio con la tabla compilada hasta 8.0.
for task in disc_mc car; do
    for bin in "build/wann_${task}" "build/wann_eval_weights_${task}"; do
        if [ ! -x "$bin" ] || ! grep -qa "alg_weight_vals" "$bin"; then
            echo "ERROR: $bin no existe o no soporta alg_weight_vals; corre ./build.sh antes." >&2
            exit 1
        fi
    done
done

run_combo() {
    local task=$1 enc=$2 dec=$3
    echo "=== $task | $enc | $dec | tag=$TAG | pesos={$WEIGHT_VALS} ==="

    # Phase 1: reducción del espacio (4 rondas x 30 trials)
    python screening_reduce.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --weight-vals "$WEIGHT_VALS" --rounds 4 --n 30 \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"

    # Phase 2+3: 40 trials, top 3 configs, 11 seeds cada una (+ eval/bootstrap)
    python screening_full.py --task "$task" --encoder "$enc" --decoder "$dec" \
        --tag "$TAG" --weight-vals "$WEIGHT_VALS" --mode both --n 40 --top 3 --seeds 11 \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
}

for combo in $COMBOS; do
    IFS=: read -r task enc dec <<< "$combo"
    run_combo "$task" "$enc" "$dec"
done

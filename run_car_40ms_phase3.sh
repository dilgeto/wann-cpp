#!/bin/bash
# Corre SOLO la Fase 3 (validación, top-K configs × N seeds) de la
# combinación oficial Racing Car ttfs+first_spike, usando wann_car (ventana
# de simulación SNN = WANN_CAR_SIM_WINDOW_MS en include/wann/SnnCarTask.h,
# ahora 40.0 ms).
#
# Reutiliza el MISMO resultado de Fase 2 que la corrida oficial
# (screening_full/car_ttfs_first_spike/p2_results.csv — mismas
# configuraciones de hiperparámetros ganadoras). Escribe en un directorio de
# salida distinto (--tag 40ms → screening_full/car_ttfs_first_spike_40ms/,
# log/full_p3_car_ttfs_first_spike_40ms/) para no pisar la Fase 3 oficial
# (log/full_p3_car_ttfs_first_spike/, corrida con SIM_WINDOW_MS=20.0).
#
# Uso:
#   bash run_car_40ms_phase3.sh
#   JOBS_FULL=4 OMP_FULL=4 bash run_car_40ms_phase3.sh   # ajustar a los cores disponibles
#
# Requiere un Python con optuna/pandas/scipy/scikit-learn en PATH. En el
# cluster: `.venv/` (activado automáticamente abajo si existe). Localmente,
# si no tienes ese venv, actívalo tú antes de correr el script, ej.:
#   source /home/dilget/Tesis/cluster_results/venv/bin/activate
#   bash run_car_40ms_phase3.sh
#
# Variables de entorno configurables:
#   JOBS_FULL   hilos paralelos (default 3)
#   OMP_FULL    threads OMP por proceso (default 190 — bájalo si no estás
#               en el cluster; en una máquina de N cores, jobs × omp ≈ N)

set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}

TASK=car
ENCODER=ttfs
DECODER=first_spike
TAG=40ms

OFFICIAL_P2="screening_full/${TASK}_${ENCODER}_${DECODER}/p2_results.csv"
TAGGED_DIR="screening_full/${TASK}_${ENCODER}_${DECODER}_${TAG}"

if [ ! -f "$OFFICIAL_P2" ]; then
    echo "ERROR: no existe $OFFICIAL_P2 — la Fase 2 oficial de $TASK/$ENCODER/$DECODER debe existir primero." >&2
    exit 1
fi

echo "=== Compilando wann_car (SIM_WINDOW_MS actual del .hpp) ==="
cmake --build build --target wann_car

echo "=== Sembrando Fase 2 (misma que la oficial) en $TAGGED_DIR ==="
mkdir -p "$TAGGED_DIR"
cp "$OFFICIAL_P2" "$TAGGED_DIR/p2_results.csv"

echo "=== Fase 3 (40ms) — $TASK | $ENCODER | $DECODER | tag=$TAG ==="
python3 screening_full.py \
    --task "$TASK" --encoder "$ENCODER" --decoder "$DECODER" \
    --tag "$TAG" \
    --mode phase3 --top 3 --seeds 11 \
    --jobs "$JOBS_FULL" --omp "$OMP_FULL"

echo "=== Listo — resultados en ${TAGGED_DIR}/ (log/full_p3_${TASK}_${ENCODER}_${DECODER}_${TAG}/) ==="

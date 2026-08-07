#!/bin/bash
# Corre el proceso de optimización Optuna "wide" para Racing Car
# (ttfs + first_spike): rangos de hiperparámetros ampliados
# (p/car_wide_space.json), 512 generaciones en Fase 2/3
# (p/car_snn_512gen.json), 40 trials en Fase 2.
#
# Usa --tag wide en ambas fases para escribir en
# screening_reduce/car_ttfs_first_spike_wide/ y
# screening_full/car_ttfs_first_spike_wide/ — no toca ni hace resume
# sobre las corridas oficiales (car_ttfs_first_spike, sin sufijo).
#
# Correr en el cluster, dentro de wann-cpp/ (requiere .venv activable y
# ./build/wann_car ya compilado; subir antes con rsync):
#   rsync -av screening_reduce.py screening_full.py \
#       p/car_wide_space.json p/car_snn_512gen.json run_car_wide.sh \
#       ctorresu@mullo.diinf.usach.cl:/home/DIINF/ctorresu/wann-cpp/
#
# Uso:
#   nohup bash run_car_wide.sh > run_car_wide.log 2>&1 &
#   disown
#
# Variables de entorno configurables:
#   JOBS_REDUCE   hilos paralelos en Fase 1  (default 8)
#   OMP_REDUCE    threads OMP por proceso en Fase 1 (default 64)
#   JOBS_FULL     hilos paralelos en Fase 2+3 (default 3)
#   OMP_FULL      threads OMP por proceso en Fase 2+3 (default 190)
#
#   JOBS_REDUCE=16 OMP_REDUCE=8 JOBS_FULL=4 OMP_FULL=32 bash run_car_wide.sh

set -euo pipefail

source .venv/bin/activate

JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}

TASK=car
ENCODER=ttfs
DECODER=first_spike
TAG=wide

echo "=== Fase 1 (reduce) — $TASK | $ENCODER | $DECODER | tag=$TAG ==="
python screening_reduce.py \
    --task "$TASK" --encoder "$ENCODER" --decoder "$DECODER" \
    --tag "$TAG" --space-override p/car_wide_space.json \
    --rounds 4 --n 30 \
    --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"

echo "=== Fase 2+3 (full) — $TASK | $ENCODER | $DECODER | tag=$TAG ==="
python screening_full.py \
    --task "$TASK" --encoder "$ENCODER" --decoder "$DECODER" \
    --tag "$TAG" --base p/car_snn_512gen.json \
    --mode both --n 40 --top 3 --seeds 11 \
    --jobs "$JOBS_FULL" --omp "$OMP_FULL"

echo "=== Listo — resultados en screening_reduce/${TASK}_${ENCODER}_${DECODER}_${TAG}/ y screening_full/${TASK}_${ENCODER}_${DECODER}_${TAG}/ ==="

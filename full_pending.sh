#!/bin/bash

set -euo pipefail

source .venv/bin/activate

# ── Ya completados (no re-correr) ─────────────────────────────────────────────
# acrobot_small_first_spike  p2=20/20  p3=✓
# acrobot_small_wta          p2=20/20  p3=✓
# mountain_car_ttfs_rate     p2=15/20  p3=✓  (aceptable)

# ── Re-correr: Phase 2 tuvo timeout parcial → muestra sesgada ─────────────────
# (sobreescribe p2_results.csv y p3 con resultados correctos)

python screening_full.py --task acrobot --encoder ttfs --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task acrobot --encoder ttfs --decoder wta \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task mountain_car --encoder ttfs --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

# ── Re-correr: Phase 2 tuvo 0 exitosos (todos timeout) ───────────────────────
python screening_full.py --task mountain_car --encoder small --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

# ── Correr por primera vez ────────────────────────────────────────────────────
python screening_full.py --task mountain_car --encoder small --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task car --encoder ttfs --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task car --encoder ttfs --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task car --encoder small --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

python screening_full.py --task car --encoder small --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 4 --omp 60

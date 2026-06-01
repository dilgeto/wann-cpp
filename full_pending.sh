#!/bin/bash

set -euo pipefail

source .venv/bin/activate

# ── Ya completados (acrobot_*, mountain_car_ttfs_*) ───────────────────────────
# acrobot_small_first_spike  p2=20/20  p3=✓
# acrobot_small_wta          p2=20/20  p3=✓
# acrobot_ttfs_first_spike   p2= 8/20  p3=✓
# acrobot_ttfs_wta           p2= 5/20  p3=✓
# mountain_car_ttfs_first_spike  p2= 6/20  p3=✓
# mountain_car_ttfs_rate         p2=15/20  p3=✓

# ── Mountain Car — small (re-correr: phase2 tuvo 0/20 exitosos) ───────────────
python screening_full.py --task mountain_car --encoder small --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

python screening_full.py --task mountain_car --encoder small --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

# ── Racing Car ────────────────────────────────────────────────────────────────
python screening_full.py --task car --encoder ttfs --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

python screening_full.py --task car --encoder ttfs --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

python screening_full.py --task car --encoder small --decoder first_spike \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

python screening_full.py --task car --encoder small --decoder rate \
    --mode both --n 20 --top 3 --seeds 5 --jobs 8 --omp 24

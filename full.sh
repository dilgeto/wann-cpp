#!/bin/bash

set -euo pipefail

source .venv/bin/activate

# Acrobot
python screening_full.py --task acrobot --encoder ttfs --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task acrobot --encoder ttfs --decoder wta --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task acrobot --encoder small --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task acrobot --encoder small --decoder wta --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10

# Mountain Car
python screening_full.py --task mountain_car --encoder ttfs --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task mountain_car --encoder ttfs --decoder rate --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task mountain_car --encoder small --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task mountain_car --encoder small --decoder rate --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10

# Racing Car
python screening_full.py --task car --encoder ttfs --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task car --encoder ttfs --decoder rate --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task car --encoder small --decoder first_spike --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
python screening_full.py --task car --encoder small --decoder rate --mode both --n 20 --top 3 --seeds 5 --jobs 20 --omp 10
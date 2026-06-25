#!/bin/bash

set -euo pipefail

source .venv/bin/activate

# Acrobot
python screening_reduce.py --task acrobot --encoder ttfs --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task acrobot --encoder ttfs --decoder wta --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task acrobot --encoder small --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task acrobot --encoder small --decoder wta --rounds 4 --n 30 --jobs 20 --omp 10

# Mountain Car
python screening_reduce.py --task mountain_car --encoder ttfs --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task mountain_car --encoder ttfs --decoder rate --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task mountain_car --encoder small --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task mountain_car --encoder small --decoder rate --rounds 4 --n 30 --jobs 20 --omp 10

# Racing Car
python screening_reduce.py --task car --encoder ttfs --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task car --encoder ttfs --decoder rate --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task car --encoder small --decoder first_spike --rounds 4 --n 30 --jobs 20 --omp 10
python screening_reduce.py --task car --encoder small --decoder rate --rounds 4 --n 30 --jobs 20 --omp 10
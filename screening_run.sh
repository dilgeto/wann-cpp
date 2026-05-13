#!/bin/bash
# screening_run.sh  –  Run pre-generated screening configs on the cluster.
#
# No Python required. Uses only bash + awk.
#
# Prerequisites
# -------------
#   1. Compile:  cmake --build build -j$(nproc)
#   2. Transfer configs from local machine:
#        rsync -av screening/<task>/configs/ cluster:wann-cpp/screening/<task>/configs/
#
# Usage
# -----
#   bash screening_run.sh --task mountain_car [--jobs 16] [--omp 12] [--seed 42]
#   bash screening_run.sh --task acrobot      --jobs 32 --omp 6
#
# Multiple tasks (run sequentially):
#   for task in mountain_car acrobot car; do
#       bash screening_run.sh --task "$task" --jobs 16 --omp 12
#   done
#
# Output
# ------
#   screening/<task>/peaks.csv   – two columns (no header): idx, peak_fitness
#   Transfer this file back to your local machine for analysis.

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────────
TASK=""
JOBS=16
OMP=12
SEED=42

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --omp)  OMP="$2";  shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep -v '^set'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "Usage: bash screening_run.sh --task TASK [--jobs N] [--omp M] [--seed S]" >&2
    echo "Tasks: mountain_car  acrobot  car  pendulum" >&2
    exit 1
fi

# ── Task → executable and base config ─────────────────────────────────────────
case "$TASK" in
    mountain_car) EXE="./build/wann_mountain_car"; BASE="p/mountain_car_snn.json" ;;
    acrobot)      EXE="./build/wann_acrobot";      BASE="p/acrobot_snn.json"      ;;
    car)          EXE="./build/wann_car";           BASE="p/car_snn.json"          ;;
    pendulum)     EXE="./build/wann_snn";           BASE="p/pendulum_snn.json"     ;;
    *)
        echo "Unknown task: $TASK" >&2
        echo "Valid tasks: mountain_car  acrobot  car  pendulum" >&2
        exit 1 ;;
esac

CFG_DIR="screening/$TASK/configs"
PKS_DIR="screening/$TASK/peaks"
LOG_DIR="log/scr_$TASK"

if [[ ! -d "$CFG_DIR" ]]; then
    echo "ERROR: $CFG_DIR not found." >&2
    echo "Generate configs locally first:  python screening.py --task $TASK --mode generate-only" >&2
    exit 1
fi

if [[ ! -f "$EXE" ]]; then
    echo "ERROR: $EXE not found. Run:  cmake --build build -j\$(nproc)" >&2
    exit 1
fi

mkdir -p "$PKS_DIR" "$LOG_DIR"

# ── Count configs ──────────────────────────────────────────────────────────────
# Collect into array to handle zero-match case safely
mapfile -t cfg_files < <(find "$CFG_DIR" -maxdepth 1 -name '*.json' | sort)
N_CONFIGS=${#cfg_files[@]}

if [[ $N_CONFIGS -eq 0 ]]; then
    echo "ERROR: no JSON files found in $CFG_DIR" >&2
    exit 1
fi

echo "============================================================"
echo "  Task    : $TASK"
echo "  Configs : $N_CONFIGS"
echo "  Parallel: $JOBS  |  OMP/run: $OMP"
echo "  Exe     : $EXE"
echo "  Base    : $BASE"
echo "  Logs    : $LOG_DIR/"
echo "============================================================"
echo ""

# ── Worker function ────────────────────────────────────────────────────────────
# Each invocation runs one config and writes its peak fitness to a per-job file.
run_one() {
    local cfg_file="$1"
    local idx
    idx=$(basename "$cfg_file" .json)

    local prefix="scr_${TASK}/${idx}"
    local pks_file="${PKS_DIR}/${idx}.txt"
    local run_seed=$(( SEED * 100000 + 10#$idx ))

    local t_start=$SECONDS

    export OMP_NUM_THREADS="$OMP"

    if "$EXE" -d "$BASE" -p "$cfg_file" -o "$prefix" -s "$run_seed" \
              >/dev/null 2>&1; then
        local stats_file="log/${prefix}_stats.out"
        local peak="nan"
        if [[ -f "$stats_file" ]]; then
            # Column 5 (1-indexed) is fitPeak; find its maximum over all rows.
            # awk handles scientific notation natively.
            peak=$(awk -F',' '
                BEGIN { m = -1e300 }
                {
                    v = $5 + 0
                    if (v > m) m = v
                }
                END { print m }
            ' "$stats_file")
        fi
        echo "$peak" > "$pks_file"
        printf "  OK   #%s  peak=%-12s  (%ds)\n" \
            "$idx" "$peak" $(( SECONDS - t_start ))
    else
        echo "nan" > "$pks_file"
        printf "  FAIL #%s\n" "$idx"
    fi
}

export -f run_one
export EXE BASE TASK PKS_DIR LOG_DIR SEED OMP

# ── Parallel execution (bash semaphore) ───────────────────────────────────────
T_WALL=$SECONDS

for cfg_file in "${cfg_files[@]}"; do
    # Wait until a slot is free
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        sleep 0.3
    done
    run_one "$cfg_file" &
done

# Wait for all background jobs
wait

ELAPSED=$(( SECONDS - T_WALL ))
echo ""
echo "All runs complete. Wall time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo ""

# ── Collect peaks into CSV ─────────────────────────────────────────────────────
PEAKS_CSV="screening/$TASK/peaks.csv"

{
    for pks_file in "$PKS_DIR"/*.txt; do
        idx=$(basename "$pks_file" .txt)
        peak=$(cat "$pks_file")
        echo "${idx},${peak}"
    done
} | sort -t, -k1,1n > "$PEAKS_CSV"

N_OK=$(awk -F',' '$2 != "nan" && $2 != "" {c++} END {print c+0}' "$PEAKS_CSV")
echo "Results: ${N_OK}/${N_CONFIGS} successful runs"
echo "Peaks CSV → $PEAKS_CSV"
echo ""
echo "Transfer to your local machine:"
echo "  rsync -av cluster:$(pwd)/screening/$TASK/peaks.csv screening/$TASK/"
echo ""
echo "Then analyse locally:"
echo "  python screening.py --task $TASK --mode analyse-only"

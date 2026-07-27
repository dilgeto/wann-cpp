#!/usr/bin/env bash
# generate_graphs.sh — genera las imágenes de graph.py (curvas de
# entrenamiento, frente de Pareto, topología de la mejor red, población
# final) para cada modelo listado en eval_p3_weights/{task}_best.csv (uno
# por combinación encoder/decoder), y las organiza en graficos/<Tarea>/.
#
# nInput/nOutput/encoder/decoder de cada modelo se leen de
# screening_full/<run_key>/p3_configs/rank<NN>_seed<NN>.json; si un
# hiperparámetro no está ahí (no fue sobreescrito), se usa el valor por
# defecto de p/*.json.
#
# Uso:
#   bash generate_graphs.sh                  # las 3 tareas
#   bash generate_graphs.sh acrobot car       # solo estas tareas
#   VENV_PY=/otra/ruta/python3 bash generate_graphs.sh
#
# Requiere: eval_p3_weights.py ya corrido (para tener *_best.csv) y un
# intérprete con matplotlib + graphviz (python) + pandas/numpy instalados.

set -euo pipefail

VENV_PY="${VENV_PY:-/home/dilget/Tesis/cluster_results/venv/bin/python3}"
BEST_DIR="eval_p3_weights"
OUT_ROOT="graficos"

# task_key:display_name:base_config
ALL_TASKS=(
    "acrobot:Acrobot:p/acrobot_snn.json"
    "mountain_car:Mountain Car:p/disc_mc_snn.json"
    "car:Racing Car:p/car_snn.json"
)

if [[ $# -gt 0 ]]; then
    selected=("$@")
else
    selected=(acrobot mountain_car car)
fi

is_selected() {
    local key="$1"
    for s in "${selected[@]}"; do
        [[ "$s" == "$key" ]] && return 0
    done
    return 1
}

for entry in "${ALL_TASKS[@]}"; do
    IFS=":" read -r task_key display_name base_config <<< "$entry"
    is_selected "$task_key" || continue

    best_csv="${BEST_DIR}/${task_key}_best.csv"
    if [[ ! -f "$best_csv" ]]; then
        echo "[$task_key] no existe $best_csv, se omite."
        continue
    fi

    out_dir="${OUT_ROOT}/${display_name}"
    mkdir -p "$out_dir"
    echo "=== $task_key ($display_name) -> $out_dir ==="

    while IFS=$'\t' read -r run_key rank seed_idx; do
        rank_p=$(printf "%02d" "$rank")
        seed_p=$(printf "%02d" "$seed_idx")
        prefix="log/full_p3_${run_key}/rank${rank_p}_seed${seed_p}"

        if [[ ! -f "${prefix}_stats.out" ]]; then
            echo "  [SKIP] no existe ${prefix}_stats.out"
            continue
        fi

        cfg_json="screening_full/${run_key}/p3_configs/rank${rank_p}_seed${seed_p}.json"

        IFS=$'\t' read -r n_input n_output title <<< "$("$VENV_PY" - "$base_config" "$cfg_json" "$display_name" <<'PY'
import json, sys

base_path, cfg_path, display_name = sys.argv[1], sys.argv[2], sys.argv[3]
base = json.load(open(base_path))
try:
    cfg = json.load(open(cfg_path))
except FileNotFoundError:
    cfg = {}

def get(key):
    return cfg.get(key, base.get(key))

def cap(s):
    s = str(s)
    return s[0].upper() + s[1:] if s else s

n_input  = get("ann_nInput")
n_output = get("ann_nOutput")
encoder  = get("snn_encoder")
decoder  = get("snn_decoder")
title = f"{cap(display_name)} {cap(encoder)}+{cap(decoder)}"
print(f"{n_input}\t{n_output}\t{title}")
PY
)"

        echo "  [$run_key rank=$rank_p seed=$seed_p] nInput=$n_input nOutput=$n_output title=\"$title\""

        "$VENV_PY" graph.py --prefix "$prefix" \
            --nInput "$n_input" --nOutput "$n_output" \
            --save --title "$title"

        base_name="${run_key}"
        for suffix in training pareto_evolution network final_pop; do
            src="${prefix}_${suffix}.png"
            if [[ -f "$src" ]]; then
                mv "$src" "${out_dir}/${base_name}_${suffix}.png"
            fi
        done
    done < <("$VENV_PY" - "$best_csv" <<'PY'
import csv, sys

with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        print(f"{row['run_key']}\t{row['rank']}\t{row['seed_idx']}")
PY
)
done

echo
echo "Listo. Imágenes en ${OUT_ROOT}/"

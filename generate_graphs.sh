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
# Para cada tarea procesada, además genera graficos/<Tarea>/phase3_summary_best.png
# — comparación de las combinaciones encoder/decoder de esa tarea (mismo
# estilo que plots/phase3_summary.png pero usando los valores de *_best.csv).
# {task}_best.csv tiene una fila por (run_key, seed_idx) — hasta 11 filas por
# combinación encoder/decoder. Por cada run_key se toma como "campeón" la fila
# con mayor "reward", y su rank/seed_idx se usan para ubicar
# log/full_p3_<run_key>/rank<NN>_seed<NN>_stats.out (curvas de entrenamiento,
# Pareto, topología, población final). Cada caja de phase3_summary_best.png
# son las 11 medias por semilla ("reward" de cada una de las filas de ese
# run_key), no episodios individuales.
# phase3_summary_best.png siempre incluye además una caja de referencia PPO
# (episodios individuales del agente ANN nativo/PPO de cada tarea, leídos de
# bootstrap_*/rewards.csv — ver PPO_REWARDS_CSV más abajo), a la izquierda de
# las combinaciones SNN, sin participar del orden ni de la comparación de
# "mejor config". Con --ann se puede agregar además un boxplot de referencia
# DQN (valores tipeados a mano), que se muestra a la izquierda de todo.
#
# Uso:
#   bash generate_graphs.sh                        # las 3 tareas
#   bash generate_graphs.sh acrobot car             # solo estas tareas
#   bash generate_graphs.sh --ann acrobot:-90.1,-85.3,-97.2,-88.0,-91.5,-86.7,-93.4,-89.9,-84.2,-90.8
#   VENV_PY=/otra/ruta/python3 bash generate_graphs.sh
#
# Requiere: eval_p3_weights.py ya corrido (para tener *_best.csv) y un
# intérprete con matplotlib + graphviz (python) + pandas/numpy/scipy instalados.

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

# task_key -> rewards.csv de bootstrap_compare_*.py con la referencia PPO
# (columna "agent"=="ann", 100 episodios). Se agrega automáticamente como
# caja de referencia en phase3_summary_best.png de cada tarea.
declare -A PPO_REWARDS_CSV=(
    [acrobot]="bootstrap_acrobot_ppo_experiment/rewards.csv"
    [mountain_car]="bootstrap_mountain_car/rewards.csv"
    [car]="bootstrap_car/rewards.csv"
)

declare -A ANN_VALUES
selected=()

VALID_TASKS=(acrobot mountain_car car)

is_valid_task() {
    local key="$1"
    for t in "${VALID_TASKS[@]}"; do
        [[ "$t" == "$key" ]] && return 0
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ann)
            IFS=":" read -r ann_task ann_values <<< "$2"
            ANN_VALUES["$ann_task"]="$ann_values"
            shift 2
            ;;
        -*)
            echo "ERROR: flag desconocida '$1' (¿quisiste decir --ann?)" >&2
            exit 1
            ;;
        *)
            if ! is_valid_task "$1"; then
                echo "ERROR: tarea desconocida '$1' (válidas: ${VALID_TASKS[*]})" >&2
                exit 1
            fi
            selected+=("$1")
            shift
            ;;
    esac
done

if [[ ${#selected[@]} -eq 0 ]]; then
    selected=("${VALID_TASKS[@]}")
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

    task_accum="$(mktemp)"

    # Por cada run_key en {task}_best.csv (hasta 11 filas, una por seed_idx),
    # se toma como "campeón" la fila de mayor "reward" y se emite su
    # rank/seed_idx junto con las 11 medias por semilla (columna "reward" de
    # todas sus filas), para usar como datos de la caja en phase3_summary_best.png.
    while IFS=$'\t' read -r run_key rank seed_idx seed_means; do
        rank_p=$(printf "%02d" "$rank")
        seed_p=$(printf "%02d" "$seed_idx")
        prefix="log/full_p3_${run_key}/rank${rank_p}_seed${seed_p}"

        if [[ ! -f "${prefix}_stats.out" ]]; then
            echo "  [SKIP] no existe ${prefix}_stats.out"
            continue
        fi

        cfg_json="screening_full/${run_key}/p3_configs/rank${rank_p}_seed${seed_p}.json"

        IFS=$'\t' read -r n_input n_output title enc_dec <<< "$("$VENV_PY" - "$base_config" "$cfg_json" "$display_name" <<'PY'
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
if encoder == "small":
    encoder = "signed"
title   = f"{cap(display_name)} {cap(encoder)}+{cap(decoder)}"
enc_dec = f"{encoder}_{decoder}"
print(f"{n_input}\t{n_output}\t{title}\t{enc_dec}")
PY
)"

        echo "  [$run_key rank=$rank_p seed=$seed_p (campeón)] nInput=$n_input nOutput=$n_output title=\"$title\""

        "$VENV_PY" graph.py --prefix "$prefix" \
            --nInput "$n_input" --nOutput "$n_output" \
            --save --title "$title"

        base_name="${run_key/small/signed}"
        for suffix in training pareto_evolution network final_pop; do
            src="${prefix}_${suffix}.png"
            if [[ -f "$src" ]]; then
                mv "$src" "${out_dir}/${base_name}_${suffix}.png"
            fi
        done

        # Curvas de entrenamiento promediadas sobre las 11 semillas de esta
        # misma configuración ganadora (mismo run_key + rank del campeón).
        "$VENV_PY" graph_mean_training.py --run-key "$run_key" --rank "$rank" \
            --title "$title" --out "${out_dir}/${base_name}_mean_training.png"

        printf '%s\t%s\n' "$enc_dec" "$seed_means" >> "$task_accum"
    done < <("$VENV_PY" - "$best_csv" <<'PY'
import csv, sys
from collections import defaultdict

groups = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        groups[row["run_key"]].append(row)

for run_key, rows in groups.items():
    champion = max(rows, key=lambda r: float(r["reward"]))
    means = ",".join(r["reward"] for r in rows)
    print(f"{run_key}\t{champion['rank']}\t{champion['seed_idx']}\t{means}")
PY
)

    if [[ -s "$task_accum" ]]; then
        summary_path="${out_dir}/phase3_summary_best.png"
        ppo_csv="${PPO_REWARDS_CSV[$task_key]:-}"
        if [[ -n "$ppo_csv" && ! -f "$ppo_csv" ]]; then
            echo "  [aviso] no existe $ppo_csv, se omite caja PPO"
            ppo_csv=""
        fi
        "$VENV_PY" - "$task_accum" "$summary_path" "$display_name" \
            "${ANN_VALUES[$task_key]:-}" "$ppo_csv" <<'PY'
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

accum_path, out_path, display_name, ann_values_arg, ppo_csv_path = sys.argv[1:6]

records = []
with open(accum_path) as f:
    for line in f:
        enc_dec, values_csv = line.rstrip("\n").split("\t")
        values = [float(v) for v in values_csv.split(",") if v != ""]
        records.append((enc_dec, values))
records.sort(key=lambda r: np.mean(r[1]), reverse=True)

labels = [r[0] for r in records]
data   = [r[1] for r in records]
colors = ["#d62728" if i == 0 else "#4C72B0" for i in range(len(records))]

# Cajas de referencia (ANN convencional), siempre a la izquierda de las
# combinaciones SNN, en el orden DQN (manual, --ann) y luego PPO (automático,
# leído de bootstrap_*/rewards.csv).
ref_labels, ref_data, ref_colors = [], [], []
if ann_values_arg != "":
    ref_labels.append("DQN")
    ref_data.append([float(v) for v in ann_values_arg.split(",") if v != ""])
    ref_colors.append("#808080")
if ppo_csv_path != "":
    with open(ppo_csv_path) as f:
        ppo_values = [float(row["reward"]) for row in csv.DictReader(f) if row["agent"] == "ann"]
    if ppo_values:
        ref_labels.append("PPO")
        ref_data.append(ppo_values)
        ref_colors.append("#2ca02c")

labels = ref_labels + labels
data   = ref_data + data
colors = ref_colors + colors

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.5), 5))

bp = ax.boxplot(data, positions=x, patch_artist=True, widths=0.6,
                medianprops=dict(color="black"))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.85)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_title(f"{display_name} — mejor config validada por combinación", fontweight="bold")
ax.set_ylabel("Reward Gymnasiun (mejor rank, Phase 3)", fontsize=13)

ylim  = ax.get_ylim()
rng   = ylim[1] - ylim[0]
label_off = rng * 0.02
# Reservar espacio arriba para que las etiquetas de texto no queden
# cortadas por el borde superior del gráfico.
top_needed = max(max(vals) for vals in data) + label_off + rng * 0.08
ax.set_ylim(ylim[0], max(ylim[1], top_needed))

for pos, vals in zip(x, data):
    ax.text(pos, max(vals) + label_off, f"{np.mean(vals):.2f}",
             ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  → {out_path}")
PY
    fi

    rm -f "$task_accum"
done

echo
echo "Listo. Imágenes en ${OUT_ROOT}/"

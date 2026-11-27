#!/usr/bin/env bash
# Ruido de memristores de las redes WANN `_w3` de Racing Car y Mountain Car (búsqueda `run_optuna_w3.sh`, pesos
# compartidos {0.5, 1, 1.5, 2, 2.5, 3}) para la línea naranja "WANN, weights 0.5–3" de figura_04D / figura_05D
# (plot_paper_noise_occ_single.py), igual que la de Acrobot (acrobot_replicas_pre_w8). SOLO EVALÚA (no entrena).
# Correrlo en el clúster, desde la raíz del repo, donde están log/full_p3_<run_key>, screening_full/<run_key> y
# ./build/wann_{car,disc_mc}_noise. Solo WANN (--no-sg): las curvas SG de las figuras no cambian.
#
# Mismo protocolo que las redes 0.5–8 de las figuras 04C/05C/04D/05D: las 11 semillas del rank ganador de la
# revalidación; nivel ODIN entero 1..7 de cada red elegido SIN ruido (semilla 999); ese nivel guardado a cada ocupación
# (w_max = nivel / ocupación); PCM y ReRAM-CMO rango aihwkit (88.2 / 9.0 uS) con el mapeo `single` (y los de MAPPINGS).
# El ruido usa --weight <nivel> explícito, así que la tabla de pesos del binario no interviene.
#
# Qué hace (reanudable: relanzarlo continúa donde quedó)
#   1. Revalidación de cada run_key -> eval_results/<run_key>/<car|mountain_car>_best.csv, SOLO si no existe (la deja
#      la post-evaluación de screening_full.py; si la copiaste de otro lado, se usa esa).
#   2. Celda sin ruido (niveles ODIN 1..7) -> eval_results/w3_noisefree/<tarea>_replicas_w3: elige el nivel de cada red.
#   3. PCM a 5 ocupaciones -> eval_results/paper_noise_occ/<tarea>_replicas_w3
#   4. ReRAM-CMO por mapeo -> eval_results/paper_noise_occ_<mapeo>_aihwkit/<tarea>_replicas_w3
#   Carpetas propias (sufijo _w3, --rep-suffix): los model_id de WANN no llevan el run_key y run_noise_compare.py salta
#   las partes que ya existen, así que en <tarea>_replicas reutilizaría en silencio los resultados de las redes 0.5–8.
#   No toca ninguna carpeta existente.
#
# Estimación (--dry-run, cota superior, por tarea): sin ruido ~1 h CPU, PCM ~4 h, CMO ~4.4 h por mapeo -> ~19 h CPU las
# dos tareas con MAPPINGS=single; con 176 hebras, del orden de 10-20 min de pared en total.
#
# Uso:
#   bash eval_results/run_paper_noise_w3.sh                        # JOBS=176 MAPPINGS=single, las dos tareas
#   TASKS=car bash eval_results/run_paper_noise_w3.sh              # solo Racing Car
#   MAPPINGS="single pair" bash eval_results/run_paper_noise_w3.sh # también el mapeo pair (figuras B)
#   DRY_RUN=1 bash eval_results/run_paper_noise_w3.sh              # solo imprime plan y estimación (pasos 2-4)
# Traer al notebook: eval_results/<run_key>/ (revalidación), eval_results/w3_noisefree/,
# eval_results/paper_noise_occ/<tarea>_replicas_w3/ y eval_results/paper_noise_occ_single_aihwkit/<tarea>_replicas_w3/.
# Figuras (en el notebook):
#   python eval_results/plot_paper_noise_occ_single.py --journal --zoom \
#       --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix D
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-176}"
MAPPINGS="${MAPPINGS:-single}"
TASKS="${TASKS:-car disc_mc}"
SUFFIX=_w3
OCCS=(0.0625 0.125 0.25 0.5 1)
LEVELS=eval_results/w3_noisefree
DRY=()
[[ -n "${DRY_RUN:-}" ]] && DRY=(--dry-run)

declare -A RUN_KEY=([car]=car_ttfs_first_spike_w3 [disc_mc]=disc_mc_small_rate_argmax_w3)
declare -A EVAL_TASK=([car]=car [disc_mc]=mountain_car)          # nombre de la tarea en eval_p3_weights.py

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -d .venv ]] && source .venv/bin/activate

# ── 0. Comprobaciones ─────────────────────────────────────────────────────────────────────────────────────────────
grep -q -- "--rep-suffix" eval_results/run_noise_compare.py \
    || die "eval_results/run_noise_compare.py no tiene --rep-suffix/--no-sg/--wann-eval-csv; sincronizá el repo."
grep -q "weight_vals_for" eval_results/eval_p3_weights.py \
    || die "eval_results/eval_p3_weights.py no lee alg_weight_vals de los p3_configs; sincronizá el repo."
for task in $TASKS; do
    rk=${RUN_KEY[$task]:-}; [[ -n "$rk" ]] || die "tarea desconocida: $task (car | disc_mc)"
    [[ -d "log/full_p3_$rk" && -d "screening_full/$rk/p3_configs" ]] || die "faltan log/full_p3_$rk o screening_full/$rk"
    grep -q '"alg_weight_vals"' "screening_full/$rk"/p3_configs/rank00_seed00.json \
        || die "screening_full/$rk/p3_configs no tiene alg_weight_vals: ¿es de verdad la búsqueda _w3?"
    for b in "build/wann_${task}_noise" "build/wann_eval_weights_${task}"; do
        [[ -x "$b" ]] || die "$b no existe: corré ./build.sh"
    done
    grep -qa alg_weight_vals "build/wann_eval_weights_${task}" \
        || die "build/wann_eval_weights_${task} no soporta alg_weight_vals (revalidaría con los pesos 0.5..8): ./build.sh"
done

for task in $TASKS; do
    rk=${RUN_KEY[$task]}
    eval_csv="eval_results/$rk/${EVAL_TASK[$task]}_best.csv"

    # ── 1. Revalidación (solo si falta) ─────────────────────────────────────────────────────────────────────────────
    if [[ ! -f "$eval_csv" ]]; then
        echo "=== 1. [$task] Revalidación de $rk -> $eval_csv ==="
        python eval_results/eval_p3_weights.py --task "${EVAL_TASK[$task]}" --run-key "$rk" \
            --seeds 11 --nreps 11 --jobs 11 --omp $(( JOBS / 11 > 0 ? JOBS / 11 : 1 )) --out-dir "eval_results/$rk"
    fi
    python - "$eval_csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
w = d.sort_values("reward", ascending=False).iloc[0]
print(f"WANN {w.run_key} rank{int(w['rank']):02d}: {len(d)} redes, reward medio {d.reward.mean():.2f} "
      f"(mejor seed{int(w.seed_idx):02d}: {w.reward:.2f}); pesos ganadores {sorted(d.weight_value.unique())}")
EOF
    COMMON=(--replicas --task "$task" --no-sg --wann-eval-csv "$eval_csv" --rep-suffix "$SUFFIX"
            --jobs "$JOBS" --skip-plots "${DRY[@]}")

    # ── 2. Celda sin ruido (elige el nivel ODIN de cada red) ───────────────────────────────────────────────────────
    echo "=== 2. [$task] Celda sin ruido -> $LEVELS/${task}_replicas$SUFFIX ==="
    python eval_results/run_noise_compare.py "${COMMON[@]}" --wann-mappings odin --devices simple --out-dir "$LEVELS"
    [[ -n "${DRY_RUN:-}" ]] && { echo "(dry-run: los pasos 3-4 necesitan los niveles del paso 2; se omiten)"; continue; }

    # ── 3. PCM a 5 ocupaciones ──────────────────────────────────────────────────────────────────────────────────────
    echo "=== 3. [$task] PCM a ${OCCS[*]} -> eval_results/paper_noise_occ/${task}_replicas$SUFFIX ==="
    python eval_results/run_noise_compare.py "${COMMON[@]}" --out-dir eval_results/paper_noise_occ \
        --levels-from "$LEVELS" --occupancies "${OCCS[@]}" --devices pcm

    # ── 4. ReRAM-CMO, rango aihwkit, por mapeo ──────────────────────────────────────────────────────────────────────
    for m in $MAPPINGS; do
        echo "=== 4. [$task] ReRAM-CMO mapeo $m -> eval_results/paper_noise_occ_${m}_aihwkit/${task}_replicas$SUFFIX ==="
        python eval_results/run_noise_compare.py "${COMMON[@]}" --out-dir "eval_results/paper_noise_occ_${m}_aihwkit" \
            --levels-from "$LEVELS" --occupancies "${OCCS[@]}" \
            --devices reramcmo --reramcmo-g-max 88.199997 --reramcmo-g-min 9.0 --reramcmo-mapping "$m"
    done
done

echo
echo "Listo. Traer al notebook:"
for task in $TASKS; do
    echo "  eval_results/${RUN_KEY[$task]}/  $LEVELS/${task}_replicas$SUFFIX/  eval_results/paper_noise_occ/${task}_replicas$SUFFIX/"
    for m in $MAPPINGS; do echo "  eval_results/paper_noise_occ_${m}_aihwkit/${task}_replicas$SUFFIX/"; done
done
echo "Figuras: python eval_results/plot_paper_noise_occ_single.py --journal --zoom \\"
echo "    --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix D"

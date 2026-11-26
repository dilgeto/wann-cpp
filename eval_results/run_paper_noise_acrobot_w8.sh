#!/usr/bin/env bash
# Ruido de memristores de las redes NUEVAS de Acrobot (búsqueda `_w8`, pesos {0.5,1,2,3,5,8}) para figura_04C / figura_05C.
# SOLO EVALÚA (no entrena). Correrlo en el clúster, desde la raíz del repo, donde están log/full_p3_acrobot_*_w8,
# ./build/wann_acrobot_noise y el venv + checkpoints de ../surrogate-gradient-rl (con apply_sgrl_cmo_mapping.py ya aplicado).
#
# Qué hace (reanudable: relanzarlo continúa donde quedó)
#   1. Revalidación de los 4 run_keys acrobot_*_w8 -> eval_results/acrobot_w8/acrobot_best.csv, SOLO si no existe (lo
#      mejor es copiarlo del notebook, donde ya se hizo, para que el ganador sea exactamente el mismo).
#   2. Celda sin ruido de WANN (niveles ODIN 1..7 por red, semillas 0 y 999) -> eval_results/acrobot_w8_noisefree/: de ahí
#      sale el nivel que cada red elige SIN ruido (--levels-from). `--devices simple` deja solo la celda `none` (el
#      protocolo reducido de --replicas no tiene celdas `simple`).
#   3. Mueve los directorios Acrobot viejos a acrobot_replicas_pre_w8 y copia a los nuevos SOLO las partes SG (sgrl_*):
#      las redes SG no cambiaron y son deterministas, así que no se re-simulan. Esto es OBLIGATORIO: los model_id de WANN
#      (p. ej. wann_occ25_s00_L2) no incluyen el run_key, y run_noise_compare.py salta toda parte que ya exista, así que
#      sobre el directorio viejo reutilizaría en silencio los resultados de las redes VIEJAS.
#   4. PCM (+ ReRAM-CMO legacy 8-90 uS, para que el directorio quede igual que el de las otras tareas) a 5 ocupaciones
#      -> eval_results/paper_noise_occ/acrobot_replicas  (mismos flags que run_paper_noise_occupancy.sh)
#   5. ReRAM-CMO rango aihwkit, mapeo `single` (y los de MAPPINGS) -> eval_results/paper_noise_occ_<mapeo>_aihwkit/
#      acrobot_replicas  (mismos flags que run_paper_noise_cmo_mapping.sh)
#   eval_results/acrobot_replicas (SG al 100 % en PCM) NO se toca: solo se leen de ahí las filas SG, que no cambian.
#
# Estimación (--dry-run en el notebook, cota superior): paso 4 = 1210 trabajos, ~8.1 h CPU de WANN; paso 5 (single) =
# 660 trabajos, ~4.4 h CPU; SG = 0 h (todo reutilizado). Con 176 hebras: unos minutos de pared por paso.
#
# Uso:
#   bash eval_results/run_paper_noise_acrobot_w8.sh                       # JOBS=176 JOBS_SGRL=60 MAPPINGS=single
#   MAPPINGS="single pair" bash eval_results/run_paper_noise_acrobot_w8.sh   # también figura_04B/05B (mapeo pair)
# Después (en cualquier máquina con los CSV de las 3 tareas):
#   python eval_results/plot_paper_noise_occ_macro.py --journal --zoom \
#       --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix C
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-176}"
JOBS_SGRL="${JOBS_SGRL:-60}"
MAPPINGS="${MAPPINGS:-single}"
OCCS=(0.0625 0.125 0.25 0.5 1)
EVAL=eval_results/acrobot_w8/acrobot_best.csv
LEVELS=eval_results/acrobot_w8_noisefree
RUN_KEYS=(acrobot_small_first_spike_w8 acrobot_small_rate_argmax_w8 acrobot_ttfs_first_spike_w8 acrobot_ttfs_rate_argmax_w8)

die() { echo "ERROR: $*" >&2; exit 1; }

# ── 0. Comprobaciones ─────────────────────────────────────────────────────────────────────────────────────────────
grep -q "eval_results/acrobot_w8/acrobot_best.csv" eval_results/run_noise_compare.py \
    || die "eval_results/run_noise_compare.py todavía elige la red WANN de Acrobot de la revalidación vieja; sincronizá el repo."
grep -q "0.5, 1.0, 2.0, 3.0, 5.0, 8.0" src/SnnAcrobotTask.cpp || die "src/SnnAcrobotTask.cpp no tiene la tabla de pesos nueva."
for b in build/wann_acrobot_noise build/wann_eval_weights_acrobot; do
    [[ "$b" -nt src/SnnAcrobotTask.cpp ]] || die "$b falta o es más viejo que src/SnnAcrobotTask.cpp: corré ./build.sh"
done
for rk in "${RUN_KEYS[@]}"; do
    [[ -d "log/full_p3_$rk" && -d "screening_full/$rk/p3_configs" ]] || die "faltan log/full_p3_$rk o screening_full/$rk"
done

# ── 1. Revalidación (solo si falta) ───────────────────────────────────────────────────────────────────────────────
if [[ ! -f "$EVAL" ]]; then
    echo "=== 1. Revalidación de ${RUN_KEYS[*]} ==="
    python eval_results/eval_p3_weights.py --task acrobot $(printf -- "--run-key %s " "${RUN_KEYS[@]}") \
        --seeds 11 --nreps 11 --jobs 16 --omp $(( JOBS / 16 > 0 ? JOBS / 16 : 1 )) --out-dir eval_results/acrobot_w8
fi
python - "$EVAL" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
w = d.sort_values("reward", ascending=False).iloc[0]
print(f"Ganador WANN Acrobot: {w.run_key} rank{int(w['rank']):02d} (mejor red seed{int(w.seed_idx):02d}, reward {w.reward:.2f})")
EOF

# ── 2. Celda sin ruido de WANN (elige el nivel ODIN de cada red) ─────────────────────────────────────────────────────
echo "=== 2. Celda sin ruido de WANN -> $LEVELS ==="
python eval_results/run_noise_compare.py --replicas --task acrobot --only-methods wann --wann-mappings odin \
    --devices simple --out-dir "$LEVELS" --jobs "$JOBS" --skip-plots

# ── 3. Directorios nuevos (los viejos quedan en acrobot_replicas_pre_w8; se copian solo las partes SG) ───────────────
fresh_dir() {
    local d="$1/acrobot_replicas" bak="$1/acrobot_replicas_pre_w8"
    if [[ -d "$bak" ]]; then
        echo "  $d: ya preparado antes (viejo en $bak)"
    elif [[ -d "$d" ]]; then
        mv "$d" "$bak"
        mkdir -p "$d/parts"
        cp -a "$bak"/parts/sgrl_* "$d/parts/"
        echo "  $d: viejo -> $bak; copiadas $(ls -d "$d"/parts/sgrl_* | wc -l) carpetas de partes SG"
    fi
}
echo "=== 3. Preparando directorios ==="
fresh_dir eval_results/paper_noise_occ
for m in $MAPPINGS; do fresh_dir "eval_results/paper_noise_occ_${m}_aihwkit"; done

# ── 4. PCM (+ CMO legacy 8-90 uS) a 5 ocupaciones ──────────────────────────────────────────────────────────────────
echo "=== 4. PCM a ${OCCS[*]} -> eval_results/paper_noise_occ ==="
python eval_results/run_noise_compare.py \
    --replicas --task acrobot \
    --out-dir eval_results/paper_noise_occ --levels-from "$LEVELS" \
    --occupancies "${OCCS[@]}" \
    --devices pcm reramcmo --reramcmo-g-max 90 --reramcmo-g-min 8 \
    --sg-variants con neu 64 --sgrl-full-occupancy 64 \
    --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" --skip-plots

# ── 5. ReRAM-CMO, rango aihwkit, por mapeo ───────────────────────────────────────────────────────────────────────────
for m in $MAPPINGS; do
    echo "=== 5. ReRAM-CMO mapeo $m -> eval_results/paper_noise_occ_${m}_aihwkit ==="
    python eval_results/run_noise_compare.py \
        --replicas --task acrobot \
        --out-dir "eval_results/paper_noise_occ_${m}_aihwkit" --levels-from "$LEVELS" \
        --occupancies "${OCCS[@]}" \
        --devices reramcmo --reramcmo-g-max 88.199997 --reramcmo-g-min 9.0 --reramcmo-mapping "$m" \
        --sg-variants con neu 64 --sgrl-full-occupancy con neu 64 \
        --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" --skip-plots
done

echo
echo "Listo. Traer al notebook: eval_results/acrobot_w8_noisefree/, eval_results/paper_noise_occ/acrobot_replicas/ y"
for m in $MAPPINGS; do echo "  eval_results/paper_noise_occ_${m}_aihwkit/acrobot_replicas/"; done
echo "Figuras: python eval_results/plot_paper_noise_occ_macro.py --journal --zoom \\"
echo "    --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix C"

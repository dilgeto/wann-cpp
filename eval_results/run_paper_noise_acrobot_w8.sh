#!/usr/bin/env bash
# Ruido de memristores de Acrobot `_w8` para figura_04C / figura_05C: las redes WANN NUEVAS (búsqueda `_w8`, pesos
# {0.5,1,2,3,5,8}) y las redes SG NUEVAS "mismas conexiones" / "mismas neuronas", reentrenadas al tamaño de la red WANN
# nueva (SG 64x64 no cambia y se reutiliza). SOLO EVALÚA (no entrena). Correrlo en el clúster, desde la raíz del repo,
# donde están log/full_p3_acrobot_*_w8, ./build/wann_acrobot_noise y el venv + checkpoints de ../surrogate-gradient-rl
# (con apply_sgrl_cmo_mapping.py ya aplicado).
#
# Redes SG nuevas: por defecto en ../surrogate-gradient-rl/{runs_w8,screening_w8} (mismas conexiones) y
# {runs_neuronas_w8,screening_neuronas_w8} (mismas neuronas); cambiar con SG_CON_RUNS / SG_CON_SCREENING / SG_NEU_RUNS /
# SG_NEU_SCREENING. De cada variante se toma, como siempre, el rank de mayor media en <screening>/acrobot/p3_summary.json.
#
# Qué hace (reanudable: relanzarlo continúa donde quedó)
#   1. Revalidación de los 4 run_keys acrobot_*_w8 -> eval_results/acrobot_w8/acrobot_best.csv, SOLO si no existe (lo
#      mejor es copiarlo del notebook, donde ya se hizo, para que el ganador sea exactamente el mismo).
#   2. Celda sin ruido de WANN (niveles ODIN 1..7 por red, semillas 0 y 999) -> eval_results/acrobot_w8_noisefree/: de ahí
#      sale el nivel que cada red elige SIN ruido (--levels-from); su models.csv lista además las redes SG nuevas, que es
#      de donde las lee la figura 03. `--devices simple` deja solo la celda `none`.
#   3. Directorios nuevos: los viejos pasan a acrobot_replicas_pre_w8 y a los nuevos se copian SOLO las partes de SG 64x64
#      (sgrl_64_*; red sin cambios y determinista). OBLIGATORIO: los model_id no distinguen redes viejas de nuevas
#      (WANN: wann_occ25_s00_L2; SG: sgrl_con_occ25_acrobot-p3-rank0-s101, igual en runs/ que en runs_w8/) y
#      run_noise_compare.py salta toda parte que ya exista. Si el directorio ya se preparó con la versión anterior de este
#      script (que copiaba también sgrl_con_* / sgrl_neu_* de las redes SG viejas), esas copias se borran una vez
#      (marca .sg_w8); los originales siguen en acrobot_replicas_pre_w8.
#   4. PCM (+ ReRAM-CMO legacy 8-90 uS, para que el directorio quede igual que el de las otras tareas) a 5 ocupaciones
#      -> eval_results/paper_noise_occ/acrobot_replicas. A diferencia de run_paper_noise_occupancy.sh, SG con/neu se
#      simulan también al 100 % (--sgrl-full-occupancy con neu 64): el 100 % de las otras tareas sale de
#      eval_results/<tarea>_replicas, que aquí tiene las redes SG viejas. plot_paper_noise_occupancy.py usa los datos al
#      100 % de este barrido cuando existen.
#   5. ReRAM-CMO rango aihwkit, mapeo `single` (y los de MAPPINGS) -> eval_results/paper_noise_occ_<mapeo>_aihwkit/
#      acrobot_replicas  (mismos flags que run_paper_noise_cmo_mapping.sh)
#   eval_results/acrobot_replicas NO se toca (para Acrobot ya no se lee de ahí en las figuras 04C/05C).
#
# Estimación (--dry-run en el notebook, cota superior): WANN ~8.1 h CPU (paso 4) + ~4.4 h (paso 5, single); SG con/neu
# ~8.1 h + ~4.4 h (estimado con las redes SG viejas); SG 64x64 = 0 h (reutilizado). Con 176 hebras: unos minutos de pared
# por paso. Memoria: ~0.5 GB por proceso SG (JOBS_SGRL=60 -> ~30 GB).
#
# Uso:
#   bash eval_results/run_paper_noise_acrobot_w8.sh                          # JOBS=176 JOBS_SGRL=60 MAPPINGS=single
#   MAPPINGS="single pair" bash eval_results/run_paper_noise_acrobot_w8.sh   # también figura_04B/05B (mapeo pair)
#   SG_CON_RUNS=otro SG_CON_SCREENING=otro_scr bash eval_results/run_paper_noise_acrobot_w8.sh
# Para la figura 03 hace falta además la cuantización a 3 bits de las redes SG nuevas (en ../surrogate-gradient-rl):
#   python scripts/quantize_eval.py --runs-dir runs_w8 runs_neuronas_w8 --task acrobot --jobs 32
# Después (en cualquier máquina con los CSV de las 3 tareas):
#   python eval_results/plot_paper_noise_occ_macro.py --journal --zoom \
#       --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix C
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-176}"
JOBS_SGRL="${JOBS_SGRL:-60}"
MAPPINGS="${MAPPINGS:-single}"
SG_REPO="${SG_REPO:-../surrogate-gradient-rl}"
SG_CON_RUNS="${SG_CON_RUNS:-runs_w8}"
SG_CON_SCREENING="${SG_CON_SCREENING:-screening_w8}"
SG_NEU_RUNS="${SG_NEU_RUNS:-runs_neuronas_w8}"
SG_NEU_SCREENING="${SG_NEU_SCREENING:-screening_neuronas_w8}"
OCCS=(0.0625 0.125 0.25 0.5 1)
EVAL=eval_results/acrobot_w8/acrobot_best.csv
LEVELS=eval_results/acrobot_w8_noisefree
RUN_KEYS=(acrobot_small_first_spike_w8 acrobot_small_rate_argmax_w8 acrobot_ttfs_first_spike_w8 acrobot_ttfs_rate_argmax_w8)
SG=(--sg-repo "$SG_REPO" --sg-con-runs "$SG_CON_RUNS" --sg-con-screening "$SG_CON_SCREENING"
    --sg-neu-runs "$SG_NEU_RUNS" --sg-neu-screening "$SG_NEU_SCREENING")

die() { echo "ERROR: $*" >&2; exit 1; }

# ── 0. Comprobaciones ─────────────────────────────────────────────────────────────────────────────────────────────
grep -q "eval_results/acrobot_w8/acrobot_best.csv" eval_results/run_noise_compare.py \
    || die "eval_results/run_noise_compare.py todavía elige la red WANN de Acrobot de la revalidación vieja; sincronizá el repo."
grep -q "full_here" eval_results/plot_paper_noise_occupancy.py \
    || die "eval_results/plot_paper_noise_occupancy.py no lee SG al 100 % de este barrido; sincronizá el repo."
grep -q "0.5, 1.0, 2.0, 3.0, 5.0, 8.0" src/SnnAcrobotTask.cpp || die "src/SnnAcrobotTask.cpp no tiene la tabla de pesos nueva."
for b in build/wann_acrobot_noise build/wann_eval_weights_acrobot; do
    [[ "$b" -nt src/SnnAcrobotTask.cpp ]] || die "$b falta o es más viejo que src/SnnAcrobotTask.cpp: corré ./build.sh"
done
for rk in "${RUN_KEYS[@]}"; do
    [[ -d "log/full_p3_$rk" && -d "screening_full/$rk/p3_configs" ]] || die "faltan log/full_p3_$rk o screening_full/$rk"
done
for pair in "$SG_CON_SCREENING:$SG_CON_RUNS" "$SG_NEU_SCREENING:$SG_NEU_RUNS"; do
    scr=${pair%%:*}; runs=${pair#*:}
    [[ -f "$SG_REPO/$scr/acrobot/p3_summary.json" ]] \
        || die "no existe $SG_REPO/$scr/acrobot/p3_summary.json: falta entrenar (screening) las redes SG nuevas"
    ls "$SG_REPO/$runs"/acrobot-p3-rank*-s*/final.pt >/dev/null 2>&1 || die "no hay $SG_REPO/$runs/acrobot-p3-rank*-s*/final.pt"
    [[ "$runs" != runs && "$runs" != runs_neuronas ]] || die "$runs son las redes SG VIEJAS (tamaño de la red WANN vieja)"
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
    --devices simple --out-dir "$LEVELS" "${SG[@]}" --jobs "$JOBS" --skip-plots

# ── 3. Directorios nuevos (los viejos quedan en acrobot_replicas_pre_w8; se copian solo las partes de SG 64x64) ─────
fresh_dir() {
    local d="$1/acrobot_replicas" bak="$1/acrobot_replicas_pre_w8"
    if [[ ! -d "$bak" && -d "$d" ]]; then
        mv "$d" "$bak"
        mkdir -p "$d/parts"
        cp -a "$bak"/parts/sgrl_64_* "$d/parts/"
        echo "  $d: viejo -> $bak; copiadas $(ls -d "$d"/parts/sgrl_64_* | wc -l) carpetas de partes SG 64x64"
    elif [[ -d "$d" && ! -f "$d/.sg_w8" ]]; then     # preparado por la versión anterior: tenía copias de SG con/neu viejas
        rm -rf "$d"/parts/sgrl_con_* "$d"/parts/sgrl_neu_*
        echo "  $d: borradas las copias de SG con/neu viejas (originales en $bak)"
    else
        echo "  $d: ya preparado"
    fi
    mkdir -p "$d"
    touch "$d/.sg_w8"
}
echo "=== 3. Preparando directorios ==="
fresh_dir eval_results/paper_noise_occ
for m in $MAPPINGS; do fresh_dir "eval_results/paper_noise_occ_${m}_aihwkit"; done

# ── 4. PCM (+ CMO legacy 8-90 uS) a 5 ocupaciones; SG con/neu también al 100 % ──────────────────────────────────────
echo "=== 4. PCM a ${OCCS[*]} -> eval_results/paper_noise_occ ==="
python eval_results/run_noise_compare.py \
    --replicas --task acrobot \
    --out-dir eval_results/paper_noise_occ --levels-from "$LEVELS" \
    --occupancies "${OCCS[@]}" \
    --devices pcm reramcmo --reramcmo-g-max 90 --reramcmo-g-min 8 \
    --sg-variants con neu 64 --sgrl-full-occupancy con neu 64 "${SG[@]}" \
    --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" --skip-plots

# ── 5. ReRAM-CMO, rango aihwkit, por mapeo ───────────────────────────────────────────────────────────────────────────
for m in $MAPPINGS; do
    echo "=== 5. ReRAM-CMO mapeo $m -> eval_results/paper_noise_occ_${m}_aihwkit ==="
    python eval_results/run_noise_compare.py \
        --replicas --task acrobot \
        --out-dir "eval_results/paper_noise_occ_${m}_aihwkit" --levels-from "$LEVELS" \
        --occupancies "${OCCS[@]}" \
        --devices reramcmo --reramcmo-g-max 88.199997 --reramcmo-g-min 9.0 --reramcmo-mapping "$m" \
        --sg-variants con neu 64 --sgrl-full-occupancy con neu 64 "${SG[@]}" \
        --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" --skip-plots
done

echo
echo "Listo. Traer al notebook: eval_results/acrobot_w8_noisefree/, eval_results/paper_noise_occ/acrobot_replicas/ y"
for m in $MAPPINGS; do echo "  eval_results/paper_noise_occ_${m}_aihwkit/acrobot_replicas/"; done
echo "y de $SG_REPO: runs_quant/odin/quantization.csv (tras quantize_eval.py, para la figura 03)."
echo "Figuras: python eval_results/plot_paper_noise_occ_macro.py --journal --zoom \\"
echo "    --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix C"

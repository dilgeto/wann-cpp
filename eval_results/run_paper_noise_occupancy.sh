#!/usr/bin/env bash
# Barrido de ruido de memristores a distintas OCUPACIONES de g_max (6.25, 12.5, 25, 50, 100 %: las mismas del estudio
# de ocupación) para las figuras del paper. PCM (g_max 25 uS) y ReRAM-CMO con el rango de Falcone (8-90 uS); WANN,
# SGRL con las mismas conexiones, SGRL con las mismas neuronas y SGRL 64x64 (metodología nueva); float32, sin cuantización.
# SOLO EVALÚA (no entrena). Correrlo donde estén ./build/wann_*_noise, el venv de ../surrogate-gradient-rl y los
# checkpoints (p. ej. el clúster).
#
# Requisitos
#   1. Parche de SGRL (agrega --occupancy a noise_eval.py; por defecto 1 = comportamiento anterior):
#        (cd ../surrogate-gradient-rl && git apply ../wann-cpp/eval_results/sgrl_occupancy.patch)
#      run_noise_compare.py lo comprueba y se detiene con ese mensaje si falta.
#   2. eval_results/<tarea>_replicas/{sweep_all.csv,models.csv}: de ahí sale el nivel ODIN que cada red WANN eligió SIN
#      ruido (el barrido solo simula ese nivel por red, no los 7).
#
# Variante 64x64 (SGRL 64x64 con la metodología de screening nueva: ../surrogate-gradient-rl/runs_64x64 y screening_64x64;
# NO las de runs_default/): se simula a las 5 ocupaciones, incluido el 100 % (no hay datos previos). Si ya corriste el
# barrido sin la 64, relanzar este mismo script solo calcula las partes que faltan (las demás se saltan).
#
# Qué simula (protocolo de las réplicas: 30 chips x 1 lectura, semilla 0, 22 celdas por modelo y ocupación)
#   * WANN: SU nivel ODIN elegido, guardado a cada ocupación (w_max = nivel / ocupación), las 5 ocupaciones.
#   * SGRL con/neu: el mayor |w| de cada capa a esa fracción de g_max, ocupaciones 6.25-50 %. El 100 % NO se simula:
#     es el comportamiento por defecto y ya está en eval_results/<tarea>_replicas (PCM) y eval_results/paper_noise (CMO).
#   * Dispositivos pcm y reramcmo (8-90 uS); ReRAM Wan2022 no.
#   * Salida: eval_results/paper_noise_occ/<tarea>_replicas/{parts/,sweep_all.csv,models.csv}. Reanudable (relanzar el
#     mismo comando salta lo hecho; fallos en failed_jobs.txt).
#
# Estimación (--dry-run, cota superior): 9438 trabajos de 30 filas; ~24 h CPU de WANN + ~135 h CPU de SGRL (casi todo
# CarTrack); con JOBS=300 y JOBS_SGRL=100 ≈ 81 min de pared; hasta 0.5 GB de RAM por proceso de SGRL.
#
# Uso (desde la raíz del repo):
#   bash eval_results/run_paper_noise_occupancy.sh                    # JOBS=300 JOBS_SGRL=100
#   JOBS=190 JOBS_SGRL=60 bash eval_results/run_paper_noise_occupancy.sh
#   DRY=1 bash eval_results/run_paper_noise_occupancy.sh              # solo el plan, no simula
#   TASKS="acrobot disc_mc" bash eval_results/run_paper_noise_occupancy.sh   # solo algunas tareas
# Luego, en cualquier máquina con los CSV:   python eval_results/plot_paper_noise_occupancy.py
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-300}"
JOBS_SGRL="${JOBS_SGRL:-100}"
TASKS="${TASKS:-car disc_mc acrobot}"
EXTRA=()
[[ "${DRY:-0}" == "1" ]] && EXTRA+=(--dry-run)

python eval_results/run_noise_compare.py \
    --replicas --task $TASKS \
    --out-dir eval_results/paper_noise_occ --levels-from eval_results \
    --occupancies 0.0625 0.125 0.25 0.5 1 \
    --devices pcm reramcmo --reramcmo-g-max 90 --reramcmo-g-min 8 \
    --sg-variants con neu 64 --sgrl-full-occupancy 64 \
    --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" \
    --skip-plots "${EXTRA[@]}"

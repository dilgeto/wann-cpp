#!/usr/bin/env bash
# Barrido de ruido de memristores para las figuras del paper: ReRAM-CMO con el rango de Falcone et al. (8-90 uS),
# 11 redes por método (WANN, SGRL con las mismas conexiones, SGRL con las mismas neuronas), SIN cuantización y SIN
# las redes SGRL 64x64. SOLO EVALÚA (no entrena). Hay que correrlo donde estén ./build/wann_*_noise, el venv de
# ../surrogate-gradient-rl y los checkpoints (p. ej. el clúster).
#
# Qué calcula y qué reutiliza
#   * Simula únicamente el dispositivo ReRAM-CMO (g_max 90, g_min 8 uS) + la celda sin ruido (`none`, barata), con el
#     MISMO protocolo de las réplicas anteriores (30 chips x 1 lectura, semilla 0, 12 celdas por red).
#   * PCM NO se recalcula: no depende del rango de CMO y ya está en eval_results/<tarea>_replicas/sweep_all.csv.
#     plot_paper_noise.py une ambas fuentes.
#   * Salida: eval_results/paper_noise/<tarea>_replicas/{parts/,sweep_all.csv,models.csv}. No toca eval_results/
#     <tarea>_replicas/ (CMO 1-25 uS, supuesto anterior).
#   * Reanudable: relanzar el mismo comando salta lo ya hecho; los fallos quedan en failed_jobs.txt.
#
# Estimación (--dry-run, cota superior): 3795 trabajos de 30 filas; ~20 h CPU de WANN + ~18 h CPU de SGRL;
# con 300 procesos ≈ 8 min de pared; hasta 0.5 GB de RAM por proceso de SGRL (JOBS_SGRL).
#
# Uso (desde la raíz del repo):
#   bash eval_results/run_paper_noise_falcone.sh                    # JOBS=300 JOBS_SGRL=100
#   JOBS=190 JOBS_SGRL=60 bash eval_results/run_paper_noise_falcone.sh
#   DRY=1 bash eval_results/run_paper_noise_falcone.sh              # solo el plan, no simula
# Luego, en cualquier máquina con los CSV:   python eval_results/plot_paper_noise.py
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-300}"
JOBS_SGRL="${JOBS_SGRL:-100}"
EXTRA=()
[[ "${DRY:-0}" == "1" ]] && EXTRA+=(--dry-run)

python eval_results/run_noise_compare.py \
    --replicas --task car disc_mc acrobot \
    --out-dir eval_results/paper_noise \
    --devices reramcmo --reramcmo-g-max 90 --reramcmo-g-min 8 \
    --wann-mappings odin --only-methods wann sgrl_con sgrl_neu --sg-variants con neu \
    --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" \
    --skip-plots "${EXTRA[@]}"

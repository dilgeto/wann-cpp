#!/usr/bin/env bash
# Barrido ReRAM-CMO (Falcone 8-90 uS) con un mapeo peso->conductancia que NO deforma los pesos a ruido 0, a las 5 ocupaciones,
# para WANN y SG (same connections / same neurons / 64x64). Reemplaza los datos CMO de paper_noise_occ y paper_noise (mapeo
# "legacy"), donde el recorte a g_min anulaba/achicaba los pesos incluso con ruido 0 (sobre todo en SG).
# SOLO EVALÚA (no entrena). Mismo protocolo que run_paper_noise_occupancy.sh (30 chips x 1 lectura, semilla 0). Solo CMO:
# PCM no tiene piso y sus datos no cambian.
#
#   MAPPING=pair    par diferencial con g_min en ambas ramas (como SinglePairConductanceConverter de aihwkit). A ruido 0 el
#                   peso leído es w. Ocupación = fracción del rango (g_max - g_min): a 100 % el mayor |w| queda en g_max.
#   MAPPING=single  un dispositivo por peso, afín en [min W, max W] de cada matriz (WANN: de toda la red): min -> g_min,
#                   max -> g_max (SingleDeviceConductanceConverter: el que aihwkit usa para ReRamCMONoiseModel). Ocupación
#                   (extensión nuestra) = fracción del rango, centrada: a 100 % es el mapeo literal de aihwkit.
#
# Requisitos (en el cluster, además de los de run_paper_noise_occupancy.sh):
#   1. Recompilar los binarios de ruido (wann_*_noise tienen --reramcmo-mapping):  ./build.sh
#   2. Parche de SGRL, DESPUÉS del de ocupación:
#        (cd ../surrogate-gradient-rl && git apply ../wann-cpp/eval_results/sgrl_cmo_mapping.patch)
#   3. eval_results/<tarea>_replicas/{sweep_all.csv,models.csv} (nivel ODIN por red, elegido sin ruido; no cambia).
#
# SGRL se simula también al 100 % (no se puede reutilizar eval_results/paper_noise: es el mapeo legacy).
# Salida: eval_results/paper_noise_occ_<MAPPING>/<tarea>_replicas/{parts/,sweep_all.csv,models.csv}. Reanudable.
#
# Uso (desde la raíz del repo):
#   MAPPING=pair   DRY=1 bash eval_results/run_paper_noise_cmo_mapping.sh   # solo el plan, no simula
#   MAPPING=pair   bash eval_results/run_paper_noise_cmo_mapping.sh         # JOBS=300 JOBS_SGRL=100 (~83 min de pared)
#   MAPPING=single bash eval_results/run_paper_noise_cmo_mapping.sh
#   MAPPING=pair TASKS="acrobot" JOBS=190 bash eval_results/run_paper_noise_cmo_mapping.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MAPPING="${MAPPING:?definí MAPPING=pair o MAPPING=single}"
case "$MAPPING" in pair|single) ;; *) echo "MAPPING debe ser pair o single" >&2; exit 1;; esac
JOBS="${JOBS:-300}"
JOBS_SGRL="${JOBS_SGRL:-100}"
TASKS="${TASKS:-car disc_mc acrobot}"
EXTRA=()
[[ "${DRY:-0}" == "1" ]] && EXTRA+=(--dry-run)

python eval_results/run_noise_compare.py \
    --replicas --task $TASKS \
    --out-dir "eval_results/paper_noise_occ_${MAPPING}" --levels-from eval_results \
    --occupancies 0.0625 0.125 0.25 0.5 1 \
    --devices reramcmo --reramcmo-g-max 90 --reramcmo-g-min 8 --reramcmo-mapping "$MAPPING" \
    --sg-variants con neu 64 --sgrl-full-occupancy con neu 64 \
    --jobs "$JOBS" --jobs-sgrl "$JOBS_SGRL" \
    --skip-plots "${EXTRA[@]}"

#!/bin/bash
# run_car_trials_budget.sh — Prueba aislada de la hipótesis de "presupuesto
# de búsqueda" para Racing Car TTFS+First spike (car_ttfs_first_spike):
# ¿mejora el resultado si Fase 2 corre más trials de Optuna, manteniendo
# todo lo demás idéntico a la corrida oficial?
#
# A diferencia de car_ttfs_first_spike_wide (run_car_wide.sh), que cambiaba
# 3 cosas a la vez (rango de hiperparámetros ampliado, más trials Y menos
# generaciones por trial — 512 en vez de 2048), acá se cambia SOLO el
# número de trials de Fase 2, dejando fijo:
#   - el mismo espacio reducido de Fase 1 (copiado tal cual de
#     screening_reduce/car_ttfs_first_spike/space_log.jsonl, sin re-correr
#     Fase 1 ni ampliar el rango)
#   - el mismo presupuesto de entrenamiento completo que documenta la Tabla
#     tab:presupuesto_completo (maxGen=2048, popSize=480 — el p/car_snn.json
#     ACTUAL en el repo quedó en maxGen=512 por una corrida posterior, así
#     que acá se reconstruye el original tal como estaba en el commit
#     765ca92, que es el que de verdad se usó para el campeón oficial)
#   - la misma ventana de simulación de 20ms (build_car_win20/, NO el
#     build/ compartido, que hoy compila con SIM_WINDOW_MS=40 por defecto)
#   - los mismos 3 configs validados en Fase 3 (--top 3) y las mismas 11
#     semillas por config (--seeds 11)
#
# El único eje que cambia es --n (trials de Fase 2): TRIALS por defecto,
# el doble de los 20 oficiales, para que el resultado sea comparable en
# escala con car_ttfs_first_spike_wide.
#
# ADVERTENCIA DE COSTO: a diferencia de wide (que compensó los trials extra
# recortando maxGen a 512), acá Fase 2 corre TRIALS trials al presupuesto
# COMPLETO (2048 generaciones cada uno) — el costo de Fase 2 por sí solo ya
# es comparable al de la corrida oficial completa, multiplicado por
# TRIALS/20. Fase 3 (3 configs × 11 semillas × 2048 generaciones) tiene el
# mismo costo que la Fase 3 oficial.
#
# Correr en el cluster, dentro de wann-cpp/ (requiere .venv activable y
# cmake; subir antes con rsync si hace falta):
#   nohup bash run_car_trials_budget.sh > run_car_trials_budget.log 2>&1 &
#   disown
#
# Variables de entorno configurables:
#   TRIALS  trials de Fase 2 (default 40, el doble de los 20 oficiales)
#   JOBS    corridas en paralelo (default 3, igual que JOBS_FULL de run_car_wide.sh)
#   OMP     threads OMP por corrida (default 190, igual que OMP_FULL de run_car_wide.sh)
#
#   TRIALS=60 JOBS=4 OMP=64 bash run_car_trials_budget.sh
#
# Al terminar, revisar eval_p3_weights_trials${TRIALS}/car_best.csv para
# identificar rank/seed_idx/weight_index del modelo campeón de esta corrida,
# y escribir a partir de ahí un bootstrap_compare_car_trials${TRIALS}.py
# siguiendo el mismo patrón que bootstrap_compare_car_wide.py.

set -euo pipefail

TRIALS=${TRIALS:-40}
JOBS=${JOBS:-3}
OMP=${OMP:-190}

RUN_KEY="car_ttfs_first_spike_trials${TRIALS}"

# ── 1. Compilar wann_car (entrenamiento) y wann_eval_weights_car
#      (revalidación) a 20ms, sin tocar build/ ─────────────────────────────
echo "=== Compilando binarios a SIM_WINDOW_MS=20 (build_car_win20/) ==="
cmake -S . -B build_car_win20 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DWANN_CAR_SIM_WINDOW_MS=20.0" >/dev/null
cmake --build build_car_win20 --target wann_car wann_eval_weights_car -j "$(nproc)"

# ── 2. Config base = presupuesto completo original (maxGen=2048,
#      popSize=480, commit 765ca92) ─────────────────────────────────────────
mkdir -p p
cat > p/car_trials_budget_2048gen.json <<'JSON'
{
    "task":                     "snn_car",
    "maxGen":                   2048,
    "popSize":                  480,
    "alg_nVals":                6,
    "alg_nReps":                6,
    "alg_probMoo":              0.3,
    "ann_nInput":               9,
    "ann_nOutput":              2,
    "ann_initAct":              1,
    "ann_actRange":             [1, 2, 3, 4, 5, 6],
    "ann_absWCap":              20.0,
    "prob_addConn":             0.25,
    "prob_addNode":             0.20,
    "prob_mutAct":              0.50,
    "prob_enable":              0.05,
    "prob_initEnable":          0.5,
    "prob_crossover":           0.0,
    "prob_toggleExcitatory":    0.10,
    "select_cullRatio":         0.20,
    "select_eliteRatio":        0.20,
    "select_tournSize":         8,
    "save_mod":                 10,
    "bestReps":                 20,
    "snn_encoder":              "ttfs",
    "snn_decoder":              "first_spike",
    "snn_reset_between_steps":  true,
    "snn_neurons_per_var":      5
}
JSON

# ── 3. Copiar el espacio reducido oficial de Fase 1 bajo el tag nuevo, sin
#      re-correr Fase 1 ni ampliar el rango (screening_full.py exige que
#      --tag coincida con un screening_reduce/<run_key>_<tag>/ existente) ──
mkdir -p "screening_reduce/${RUN_KEY}"
cp screening_reduce/car_ttfs_first_spike/space_log.jsonl \
   "screening_reduce/${RUN_KEY}/space_log.jsonl"

# ── 4. Fase 2 (TRIALS trials) + Fase 3 (top 3 × 11 semillas), presupuesto
#      completo, mismo rango, misma ventana de simulación ──────────────────
echo "=== Fase 2+3 — car | ttfs | first_spike | tag=trials${TRIALS} (${TRIALS} trials) ==="
python screening_full.py \
    --task car --encoder ttfs --decoder first_spike \
    --tag "trials${TRIALS}" --mode both \
    --n "${TRIALS}" --top 3 --seeds 11 \
    --base p/car_trials_budget_2048gen.json \
    --exe build_car_win20/wann_car \
    --jobs "$JOBS" --omp "$OMP"

# ── 5. Revalidación (eval_p3_weights.py), usando el binario de 20ms y
#      restringido a este run_key para no arrastrar otras corridas car_* ───
echo "=== Revalidando ${RUN_KEY} (eval_p3_weights, build_car_win20/wann_eval_weights_car) ==="
python3 - <<PYEOF
import eval_p3_weights as epw

epw.TASKS["car"]["executable"] = "build_car_win20/wann_eval_weights_car"
epw.run_task(
    task="car",
    seeds=list(range(11)),
    nreps=11,
    jobs=${JOBS},
    omp=${OMP},
    timeout=None,
    out_dir=epw.Path("eval_p3_weights_trials${TRIALS}"),
    run_keys={"${RUN_KEY}"},
)
PYEOF

echo "=== Listo — screening_full/${RUN_KEY}/, log/full_p3_${RUN_KEY}/, eval_p3_weights_trials${TRIALS}/car_best.csv ==="
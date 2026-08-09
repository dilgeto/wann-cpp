#!/bin/bash
# run_car_ablation_seeds.sh — Re-entrena con las 11 semillas de Fase 3 cada
# una de 3 variantes puntuales ya reportadas en la Discusión (Tabla
# tab:disc_car_ablacion) sobre la configuración ganadora de Racing Car
# (car_ttfs_first_spike, rank=2): addConn=0.05, mutAct=0.70, y las 3
# combinadas (addConn=0.05 + mutAct=0.70 + ventana de simulación x2, 40ms).
# La variante de solo ventana de simulación (sin addConn/mutAct) no se
# incluye acá porque ya se hizo por separado.
#
# A diferencia de esas corridas puntuales (bootstrap_car_addConn_05, etc,
# 1 sola semilla cada una), acá se corren las mismas 11 semillas que usó
# Fase 3 para la configuración ganadora, para poder calcular media y
# desviación estándar comparables con el resto de la tesis y así correr
# después eval_p3_weights.py + bootstrap_compare sobre el resultado.
#
# Semillas: run_seed = rank*10000 + seed_idx*100, con rank=2 (el rank de la
# configuración ganadora car_ttfs_first_spike en Fase 3, ver
# bootstrap_compare_car.py) y seed_idx=0..10 → 20000, 20100, ..., 21000
# (misma fórmula que screening_full.py:run_phase3).
#
# Presupuesto (maxGen/popSize): el mismo que ya se usó en las corridas
# puntuales de un solo seed (Tabla tab:disc_car_ablacion) — 512 generaciones
# para addConn, 1024 para mutAct/combinada, popSize=480 en todos los casos
# (igual que Fase 2/3 oficial). NO es el presupuesto completo de 2048
# generaciones.
#
# La ventana de simulación (SIM_WINDOW_MS) es una macro de compilación
# (WANN_CAR_SIM_WINDOW_MS en include/wann/SnnCarTask.h), así que este script
# compila 2 binarios aparte (build_car_win20/ para addConn/mutAct,
# build_car_win40/ para la combinada) en vez de depender del valor por
# defecto actual del header, y sin tocar el build/ que usa el resto del
# proyecto (p.ej. eval_p3_weights.py).
#
# Correr en el cluster, dentro de wann-cpp/ (requiere cmake y el mismo
# entorno usado para compilar el resto del proyecto; subir antes con rsync
# si hace falta):
#   nohup bash run_car_ablation_seeds.sh > run_car_ablation_seeds.log 2>&1 &
#   disown
#
# Variables de entorno configurables:
#   JOBS   corridas en paralelo (default 3, igual que JOBS_FULL de run_car_wide.sh)
#   OMP    threads OMP por corrida (default 190, igual que OMP_FULL de run_car_wide.sh)
#
#   JOBS=4 OMP=64 bash run_car_ablation_seeds.sh

set -euo pipefail

JOBS=${JOBS:-2}
OMP=${OMP:-288}

# ── 1. Compilar los 2 binarios (ventana 20 ms y 40 ms) ──────────────────────
echo "=== Compilando wann_car con SIM_WINDOW_MS=20 (build_car_win20/) ==="
cmake -S . -B build_car_win20 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DWANN_CAR_SIM_WINDOW_MS=20.0" >/dev/null
cmake --build build_car_win20 --target wann_car -j "$(nproc)"

echo "=== Compilando wann_car con SIM_WINDOW_MS=40 (build_car_win40/) ==="
cmake -S . -B build_car_win40 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DWANN_CAR_SIM_WINDOW_MS=40.0" >/dev/null
cmake --build build_car_win40 --target wann_car -j "$(nproc)"

# ── 2. Configs base (maxGen 512 / 1024, resto igual a p/car_snn.json
#      oficial — commit 765ca92, maxGen=2048 en el original) ────────────────
mkdir -p p
cat > p/car_ablation_512gen.json <<'JSON'
{
    "task":                     "snn_car",
    "maxGen":                   512,
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
sed 's/"maxGen":                   512,/"maxGen":                   1024,/' \
    p/car_ablation_512gen.json > p/car_ablation_1024gen.json

# ── 3. Overrides de hiperparámetros (config ganadora car_ttfs_first_spike
#      rank=2 — screening_full/car_ttfs_first_spike/p3_configs/rank02_seed00.json
#      — con la modificación puntual de cada variante) ─────────────────────
mkdir -p p/car_ablation_overrides

cat > p/car_ablation_overrides/addConn_05.json <<'JSON'
{
    "alg_probMoo": 0.33164673819607576,
    "prob_addConn": 0.05,
    "prob_addNode": 0.2331213139132494,
    "prob_enable": 0.010686415433666023,
    "prob_mutAct": 0.6573697219176592,
    "prob_toggleExcitatory": 0.15991387504790106,
    "prob_initEnable": 0.6145387859562532,
    "select_cullRatio": 0.3741281694372055,
    "select_eliteRatio": 0.22133860484521845,
    "select_tournSize": 13.0,
    "save_mod": 10,
    "snn_encoder": "ttfs",
    "snn_decoder": "first_spike"
}
JSON

cat > p/car_ablation_overrides/mutAct_070.json <<'JSON'
{
    "alg_probMoo": 0.33164673819607576,
    "prob_addConn": 0.054659070489093436,
    "prob_addNode": 0.2331213139132494,
    "prob_enable": 0.010686415433666023,
    "prob_mutAct": 0.70,
    "prob_toggleExcitatory": 0.15991387504790106,
    "prob_initEnable": 0.6145387859562532,
    "select_cullRatio": 0.3741281694372055,
    "select_eliteRatio": 0.22133860484521845,
    "select_tournSize": 13.0,
    "save_mod": 10,
    "snn_encoder": "ttfs",
    "snn_decoder": "first_spike"
}
JSON

cat > p/car_ablation_overrides/combined.json <<'JSON'
{
    "alg_probMoo": 0.33164673819607576,
    "prob_addConn": 0.05,
    "prob_addNode": 0.2331213139132494,
    "prob_enable": 0.010686415433666023,
    "prob_mutAct": 0.70,
    "prob_toggleExcitatory": 0.15991387504790106,
    "prob_initEnable": 0.6145387859562532,
    "select_cullRatio": 0.3741281694372055,
    "select_eliteRatio": 0.22133860484521845,
    "select_tournSize": 13.0,
    "save_mod": 10,
    "snn_encoder": "ttfs",
    "snn_decoder": "first_spike"
}
JSON

# ── 4. Entrenar las 3 variantes x 11 semillas ───────────────────────────────
# Mismas semillas que Fase 3 usó para car_ttfs_first_spike rank=2.
SEEDS=(20000 20100 20200 20300 20400 20500 20600 20700 20800 20900 21000)

# run_key : binario : config base : override
VARIANTS=(
    "car_ttfs_first_spike_addConn05_seeds:build_car_win20/wann_car:p/car_ablation_512gen.json:p/car_ablation_overrides/addConn_05.json"
    "car_ttfs_first_spike_mutAct070_seeds:build_car_win20/wann_car:p/car_ablation_1024gen.json:p/car_ablation_overrides/mutAct_070.json"
    "car_ttfs_first_spike_combined_seeds:build_car_win40/wann_car:p/car_ablation_1024gen.json:p/car_ablation_overrides/combined.json"
)

for entry in "${VARIANTS[@]}"; do
    IFS=":" read -r run_key exe base_cfg override <<< "$entry"
    echo "=== $run_key (bin=$exe base=$base_cfg override=$override) ==="

    cfg_dir="screening_full/${run_key}/p3_configs"
    mkdir -p "$cfg_dir"
    mkdir -p "log/full_p3_${run_key}"

    for si in "${!SEEDS[@]}"; do
        seed="${SEEDS[$si]}"
        seed_p=$(printf "%02d" "$si")
        cfg_path="${cfg_dir}/rank00_seed${seed_p}.json"
        cp "$override" "$cfg_path"
        log_prefix="full_p3_${run_key}/rank00_seed${seed_p}"

        (
            OMP_NUM_THREADS="$OMP" "./$exe" \
                -d "$base_cfg" -p "$cfg_path" -o "$log_prefix" -s "$seed"
        ) &

        while (( $(jobs -r -p | wc -l) >= JOBS )); do wait -n; done
    done
    wait
done

echo
echo "=== Listo. Resultados en log/full_p3_<run_key>/rank00_seed*_* ==="
echo "Run_keys: car_ttfs_first_spike_addConn05_seeds, _mutAct070_seeds, _combined_seeds"
echo "screening_full/<run_key>/p3_configs/ ya queda listo para eval_p3_weights.py"

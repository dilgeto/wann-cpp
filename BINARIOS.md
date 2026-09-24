# Cómo correr cada binario

Todos los binarios quedan en `build/` (`./build.sh` los compila; requiere
`../snn-simulator` ya compilado). Los comandos asumen que se ejecutan **desde la
raíz del repo**, porque los defaults (`p/*.json`, `log/...`) son rutas relativas.

Convenciones comunes:

- `-d` = config base (`p/<tarea>_snn.json`), `-p` = overrides encima de la base.
  `-p` acepta una ruta a un JSON **o** un JSON inline: `-p '{"maxGen":1}'`.
- Los binarios de entrenamiento escriben todo bajo `log/<prefix>_*` (`-o prefix`).
- El paralelismo sobre individuos usa OpenMP: `OMP_NUM_THREADS=N ./wann_car ...`.
- "Peso" = uno de los `alg_nVals` valores del peso compartido
  (`WEIGHT_VALS = {0.5, 1, 2, 3, 5, 8}`). `.wi` guarda el índice ganador.

Terminología: "Mountain Car" a secas = discreto (`disc_mc`, `MountainCar-v0`).
El continuo es `mountain_car` (siempre "continuous Mountain Car").

---

## 1. Entrenamiento (evolución WANN)

Binarios: `wann_car`, `wann_acrobot`, `wann_disc_mc`, `wann_mountain_car`
(continuous), `wann_l2f`, `wann_snn` (Pendulum) y `wann_train` (sin SNN, stub).

```bash
./build/wann_<tarea> [-d base.json] [-p overrides.json|'{"k":v}'] [-o prefix] [-s seed] [-v]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-d` | Config base | `p/<tarea>_snn.json` (`wann_train`: `p/default_wan.json`) |
| `-p` | Overrides (archivo o JSON inline) | ninguno |
| `-o` | Prefijo de salida → `log/<prefix>_*` | `snn_<tarea>` (`wann_train`: `test`) |
| `-s` | Semilla | `42` |
| `-v` | Escribe `log/<prefix>_debug.log` (estado SNN del mejor cada `save_mod`). No existe en `wann_snn` ni `wann_train` | off |

`early_stop_patience` (en el JSON) solo funciona en `wann_car`.

### Reproducir un entrenamiento del screening (car, ttfs, first_spike)

Los hiperparámetros ganadores del screening están en
`screening_full/car_ttfs_first_spike/p3_configs/rank<RR>_seed<SS>.json`
(solo los 10 hiperparámetros afinados + `save_mod`, `snn_encoder`, `snn_decoder`).
El screening los aplica sobre `p/car_snn.json` con la semilla
`rank*10000 + seed_idx*100`:

```bash
OMP_NUM_THREADS=64 ./build/wann_car \
    -d p/car_snn.json \
    -p screening_full/car_ttfs_first_spike/p3_configs/rank00_seed00.json \
    -o mi_corrida -s 0
```

(`rank00_seed00` → semilla 0; `rank02_seed03` → semilla `20000 + 300 = 20300`.)

`screening_full.py` pasa además `--early-stop-patience 100` por defecto; esos
JSON no lo traen. Para replicarlo, mergéalo en un override:

```bash
python3 -c "import json;c=json.load(open('screening_full/car_ttfs_first_spike/p3_configs/rank00_seed00.json'));c['early_stop_patience']=100;json.dump(c,open('p/mi_override.json','w'),indent=2)"
./build/wann_car -d p/car_snn.json -p p/mi_override.json -o mi_corrida -s 0
```

### Archivos que produce (`log/`)

| Archivo | Contenido |
|---------|-----------|
| `<prefix>_stats.out` | Por generación: evals acumuladas, fitMed, fitMax, fitTop, fitPeak, nodeMed, connMed, fitTopOrig |
| `<prefix>_best.out` / `.wi` / `.gen` | Mejor red, índice de peso ganador, generación del guardado |
| `<prefix>_best/NNNN.out` | Snapshot del mejor por generación guardada |
| `<prefix>_pareto/NNNN.out` | Población (fitness, fitMax, nConn) para el frente de Pareto |
| `<prefix>_replay/gen_NNNN.csv` | Trayectoria del élite (solo `wann_car`) |
| `<prefix>_time.log` | Tiempo total / por generación (solo `wann_car`) |
| `<prefix>_mutstats.csv` | Por generación: qué mutación se eligió/aplicó y qué objetivo usó `probMoo` (solo `wann_car`) |
| `<prefix>_mutstats_summary.csv` | Totales y porcentajes de lo anterior (solo `wann_car`) |

---

## 2. Evaluación de la mejor red — `wann_<tarea>_eval`

Binarios: `wann_car_eval`, `wann_acrobot_eval`, `wann_disc_mc_eval`,
`wann_mountain_car_eval`, `wann_l2f_eval`.

```bash
./build/wann_car_eval [-f red.out] [-d config.json] [-w peso|best] [-n episodios] \
                      [-s seed] [-t steps] [-S best|N] [-o salida.csv] [-i nInput]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada (`.out`; el `.wi` se busca junto a ella) | `log/snn_<tarea>_best.out` |
| `-d` | Config | `p/<tarea>_snn.json` |
| `-w` | `best` = lee el `.wi` (o prueba todos los pesos si no existe); un número = peso fijo (se usa el índice más cercano) | `best` |
| `-n` | Nº de episodios | `10` |
| `-s` | Seed base | `0` |
| `-t` | Pasos máx. por episodio (**solo car**) | default de la tarea |
| `-S` | Guardar la trayectoria de un episodio: `best` (mejor shaped) o índice `N` | no guarda |
| `-o` | CSV de la trayectoria de `-S` | `<red>_ep<N>_replay.csv` |
| `-i` | Sobrescribe `ann_nInput` del config (**solo car, acrobot, disc_mc**) | del config |

Imprime cada episodio (shaped y original) y un resumen media/desv/min/max.

```bash
./build/wann_car_eval -f log/mi_corrida_best.out -d p/car_snn.json -n 30 -s 0 -S best
```

---

## 3. Replay de un episodio — `wann_car_replay`, `wann_l2f_replay`

Corre **un** episodio y exporta la trayectoria a CSV para visualizar.

```bash
./build/wann_car_replay [-f best.out] [-d config.json] [-w peso|best] [-s seed] [-o salida.csv]
./build/wann_l2f_replay [-f best.out] [-d config.json] [-w idx|best]  [-s seed] [-o salida.csv]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada | `log/snn_<car\|l2f>_best.out` |
| `-d` | Config | `p/<car\|l2f>_snn.json` |
| `-w` | `best` = evalúa todos los pesos y elige el mejor. En **car** un valor de peso (p.ej. `5.0`); en **l2f** un índice de `WEIGHT_VALS` | `best` |
| `-s` | Seed | `0` |
| `-o` | CSV de salida | `<red>_replay.csv` |

```bash
./build/wann_car_replay -f log/mi_corrida_best.out -d p/car_snn.json -w best -s 0
```

---

## 4. Barrido de pesos compartidos — `wann_eval_weights_<tarea>`

Binarios: `wann_eval_weights_acrobot`, `wann_eval_weights_car`,
`wann_eval_weights_disc_mc` (la tarea va fija en compilación). Imprime a stdout
un CSV con una fila por seed. Lo orquesta `eval_results/eval_p3_weights.py`.

```bash
./build/wann_eval_weights_car -f red.out -d p/car_snn.json --seeds 0,1,2,3 \
    [-p overrides.json] [--reward shaped|original] \
    [--episode-detail --weight-index N] [--nreps N]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada (**obligatorio**) | — |
| `-d` | Config (**obligatorio**) | — |
| `--seeds` | Lista de seeds separadas por coma (**obligatorio**) | — |
| `-p` | Overrides del config (archivo) | ninguno |
| `--reward` | `shaped` (como en entrenamiento) u `original` (sin shaping) | `shaped` |
| `--episode-detail` | En vez del promedio por (seed, peso), imprime cada episodio individual (`seed,episode,reward`). Requiere `--weight-index` | off |
| `--weight-index` | Índice de peso (≥ 0) para `--episode-detail` | — |
| `--nreps` | Sobrescribe `alg_nReps` solo en memoria | del config |

---

## 5. Exportar a ODIN — `wann_car_odin_export`

Convierte una red car ya evolucionada a un JSON de configuración para el core
ODIN (ver `ODIN_DEPLOYMENT.md`). Corre en x86, no necesita hardware.

```bash
./build/wann_car_odin_export [-f red.out] [-d config.json] [-w weight_index] \
                             [-k run_key] [-o salida.json] [--thr 0-7]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada | `log/snn_car_best.out` |
| `-d` | Config | `p/car_snn.json` |
| `-w` | Índice de peso compartido; si falta se lee el `.wi` junto a la red (error si no existe) | `.wi` |
| `-k` | `run_key` (metadato) | vacío |
| `-o` | JSON de salida | `odin_config.json` |
| `--thr` | Fuerza el umbral IZH (0–7, 3 bits) de todas las neuronas | valores de `izhParamsFor()` |

Verificar antes de exportar una red con decoder `population_vector` (no está
comprobado).

## 6. Evaluar sobre hardware ODIN — `wann_car_odin_eval`

Solo existe si se compiló con `-DWANN_ODIN_HW=ON` (ARM/Linux, ZCU104/PYNQ). En
cada power cycle hay que correr antes `python3 odin_bootstrap.py /ruta/odin.bit`.

```bash
./build/wann_car_odin_eval [-c odin_config.json] [-d config.json] [-n episodios] \
                           [-s seed] [-m window_ms] [--csv] [--profile] [--steps N]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-c` | Config exportada por `wann_car_odin_export` | `odin_config.json` |
| `-d` | Config de hiperparámetros | `p/car_snn.json` |
| `-n` | Nº de episodios | `10` |
| `-s` | Seed | `0` |
| `-m`, `--window-ms` | Ventana SNN en ms; **debe coincidir con `snn_window_ms` del entrenamiento** (avisa si no se pasa) | `40` |
| `--csv` | stdout = solo `episode,reward` (el resto a stderr) | off |
| `--profile` | Desglose de tiempo entorno vs. camino ODIN | off |
| `--steps` | Pasos por episodio | `1000` |

---

## 7. Pipeline de screening (Python, orquesta los binarios)

```bash
python screening_reduce.py --task car --encoder ttfs --decoder first_spike --rounds 4 --n 30 --jobs 8 --omp 64
python screening_full.py   --task car --encoder ttfs --decoder first_spike --mode both --n 20 --top 3 --seeds 11 --jobs 3 --omp 190
python eval_results/eval_p3_weights.py --task car --seeds 11
bash generate_graphs.sh car
python bootstrap_results/bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike
```

`--jobs` × `--omp` deben ajustarse al número de cores (se multiplican).

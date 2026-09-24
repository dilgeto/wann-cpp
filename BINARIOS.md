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

`snapshot_interval` (en el JSON, default `0` = apagado, solo `wann_car`): si es
`> 0`, escribe `log/<prefix>_lineage.csv` (cada generación: padre, operador y
fitness de cada individuo) y, cada `snapshot_interval` generaciones más la
última, `log/<prefix>_snap/gen_XXXXX.json` con los genomas completos de la
población y sus recompensas. Es solo registro: no consume números aleatorios,
así que la misma semilla da la misma corrida con o sin él. Lo consume
`wann_car_neighborhood` (sección 4b).

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

## 4b. Análisis de vecindad — `wann_car_neighborhood`

**Solo evalúa, no entrena.** Dado un individuo, enumera todos los genomas a una
mutación de distancia (N1: cada `addConn`, `addNode`, `enable`, `mutAct` y
`toggleExcitatory` posible), los evalúa con las **mismas semillas que el padre**
y reporta cuántos mejoran, son neutros o empeoran, por operador. También da la
probabilidad exacta por mutación de caer en un vecino que mejora. Sirve para
distinguir "óptimo local estricto" de "hay vecinos mejores que la evolución no
encuentra".

```bash
./build/wann_car_neighborhood -i <snapshot.json | red_best.out> [-d config.json] [-p overrides]
    [-o prefijo] [--who elite|top:K|idx:N,N|all] [--seed S] [--max-per-op N]
    [--noise-seeds R] [--eps E] [--confirm K] [--confirm-seeds V]
    [--n2-mids M] [--n2-per-mid S] [--n2-if-stuck] [--climb K] [--rng-seed S] [--dry-run]
```

`-d`/`-p` deben ser **los mismos que en el entrenamiento** (probabilidades de
operadores, `ann_actRange`, parámetros SNN).

| Flag | Significado | Default |
|------|-------------|---------|
| `-i` | `*.json`: snapshot de `snapshot_interval` (genomas completos, semillas de entrenamiento). Otro: `*_best.out` (genoma **reconstruido**, aproximado; ver `GenomeIO.h`) | obligatorio |
| `--who` | Individuos del snapshot a analizar (`elite` = mayor fitness medio) | `elite` |
| `--seed` | Fuerza la semilla base de evaluación (individuo i: `S*10000+i`) | la del snapshot; `0` con `.out` |
| `--max-per-op` | Submuestrea N vecinos por operador (`0` = todos). `weight` en el CSV reescala las probabilidades | `0` |
| `--noise-seeds` | Re-evalúa el padre con R semillas para medir el ruido; sin `--eps`, `eps = 2·sd` | `4` |
| `--eps` | Umbral fijo: `|d_mean| <= E` es neutro | `2·sd` del ruido |
| `--confirm` | Confirmación en dos etapas: el cribado de N1 usa **una** semilla por vecino, y elegir los mejores de ~1000 evaluaciones ruidosas sobreestima su ventaja (maldición del ganador). Re-evalúa los K vecinos con mayor `d` que superen `eps`, junto al padre, con semillas nuevas; se confirman si la ventaja media supera 2 errores estándar. `0` = apagada. Con ella activa, `--climb` y `--n2-if-stuck` usan solo vecinos confirmados. Si hay más de K candidatos, `P(mejora confirmada)` es cota inferior | `20` |
| `--confirm-seeds` | Semillas nuevas por candidato (≥ 2, hace falta una varianza). Costo extra ≈ `(K+1) × V × 6 × alg_nReps` episodios | `5` |
| `--n2-mids`, `--n2-per-mid` | Muestrea N2: M intermedios (ponderados por su probabilidad de aparecer) y S vecinos de cada uno. Separa mejoras alcanzables por un intermedio neutro de las que exigen cruzar un valle | `0`, `50` |
| `--n2-if-stuck` | Solo corre N2 si N1 no tiene ningún vecino que mejore | off |
| `--climb` | Ascenso voraz: hasta K pasos al mejor vecino que supere `eps`, con semilla nueva en cada paso | `0` |
| `--dry-run` | Solo cuenta vecinos por operador; no simula | off |

Salidas (`-o`, default `log/neighbors_<archivo>`): `_neighbors.csv` (una fila por
individuo evaluado, con la recompensa por peso), `_parents.csv` (resumen por
padre/paso), `_ops.csv` (desglose por operador) y `_confirm.csv` (candidatos
re-evaluados: `d` del cribado vs `d` validado ± error estándar). Costo ≈ `nº vecinos × 6 pesos ×
alg_nReps` episodios por padre; el binario lo imprime antes de simular.
`OMP_NUM_THREADS` aplica igual que en el entrenamiento.

```bash
# vecindad del élite de una generación (N1 completa) y N2 si está atascado
./build/wann_car_neighborhood -i log/mi_corrida_snap/gen_00400.json \
    -d p/car_snn.json -p mi_override.json --n2-mids 20 --n2-if-stuck
```

---

## 4c. Diagnóstico de estancamiento y dependencia de la semilla — `diagnostico/`

Scripts de Python (solo numpy/pandas; matplotlib opcional) que trabajan sobre lo que
escribe `wann_car` con `snapshot_interval > 0` y sobre las salidas de
`wann_car_neighborhood`. Ninguno entrena. Todos anclan el cwd a la raíz del repo y
escriben por defecto en `diag_results/`.

```bash
# 1) tras el entrenamiento (log/diag_rank00/seed*_lineage.csv y seed*_snap/)
python diagnostico/lineage_analysis.py --glob 'log/diag_rank00/seed*_lineage.csv' --out diag_results/lineage
python diagnostico/early_prediction.py --glob 'log/diag_rank00/seed*_lineage.csv' --out diag_results/prediction

# 2) vecindad sobre los snapshots: primero costear (no simula), luego lanzar
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank00/seed*_snap' \
    --override diag/rank00.json --gens 25,100,250,last --estimate
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank00/seed*_snap' \
    --override diag/rank00.json --gens 25,100,250,last --run --jobs 3 --omp 60 \
    --extra "--n2-mids 20 --n2-if-stuck"

# 3) resumen, agrupando semillas en terciles de nivel final
python diagnostico/neighborhood_summary.py --dir log/diag_neighborhood \
    --runs-csv diag_results/lineage_runs.csv
```

| Script | Qué responde |
|--------|--------------|
| `lineage_analysis.py` | Cuándo se "decide" una corrida: nº de ancestros de la gen 0 vivos por generación, cuándo queda uno solo, ancestro común más reciente de la población final, percentil del ancestro del élite final, largo del estancamiento y tasa de hijos que superan al élite por operador (fase temprana vs tardía). Salidas `_runs/_curves/_ops.csv` |
| `early_prediction.py` | Cuánto predice lo medido en la generación g (fitness del élite, récord, top 5 %, nº de conexiones) el nivel final, entre corridas: Spearman con IC 95 % bootstrap. Salidas `_prediction.csv/.png` |
| `run_neighborhood.py` | Elige el snapshot más cercano a cada generación pedida y lanza `wann_car_neighborhood`. **Por defecto solo imprime el plan**; `--estimate` cuenta episodios sin simular; `--run` ejecuta. Salta lo ya hecho (`--force` rehace) |
| `neighborhood_summary.py` | Clasifica cada padre (óptimo local en N1 / mejoras no confirmadas / mejoras confirmadas) y, con N2, si la salida pasa por un intermedio neutro o por un valle. Agrupa por terciles con `--runs-csv` |

Notas: el fitness de cada generación se evalúa con otra semilla y los élites se
re-evalúan, así que el récord acumulado incluye evaluaciones con suerte; por eso el
"nivel final" es la media del élite en las últimas 20 generaciones. Con menos de ~8
corridas los intervalos de correlación son muy anchos (los scripts avisan).

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

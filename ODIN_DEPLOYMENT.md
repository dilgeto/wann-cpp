# Desplegar un modelo Car en el chip ODIN (ZCU104/PYNQ)

Guía para correr un modelo car ya evolucionado (WANN + SNN) sobre el core
neuromórfico ODIN real, en vez del simulador de software. Todo lo de acá es
**solo evaluación** — nunca se entrena nada en el chip.

Componentes nuevos en este repo:
- `include/wann/OdinExport.h` / `src/OdinExport.cpp` — convierte un `.out`
  (genoma ya entrenado) a una configuración ODIN.
- `include/wann/hw/` / `src/hw/` — driver real de ODIN (`OdinDriver`,
  `OdinNetwork`), puerto de tu `odin.py` a C++ hablando directo por
  `/dev/mem`.
- `src/main_odin_export.cpp` → binario `wann_car_odin_export`.
- `src/main_car_odin_eval.cpp` / `src/SnnCarOdinTask.cpp` → binario
  `wann_car_odin_eval`.
- `odin_bootstrap.py` — carga el overlay + configura los GPIO (Python/PYNQ).
- `bootstrap_results/bootstrap_compare_car_odin_auto.py` /
  `bootstrap_compare_car_odin_manual.py` — comparación estadística ODIN vs
  ANN, reusando `bootstrap_compare_lib.py`/`bootstrap_auto_lib.py`.

Ruta actual de `odin.bit` en el board (las 16 copias que hay repartidas en
`jupyter_notebooks/` son el mismo bitstream, md5sum idéntico — esta es la
que usamos en todos los pasos de acá):
```
/home/xilinx/jupyter_notebooks/neat_comun_ranc_odin/odin.bit
```

## ⚠️ Antes que nada: el riesgo real

El PL (fabric FPGA) de un Zynq **no retiene el bitstream entre reinicios o
power cycles** — a diferencia del PS, que arranca de flash/SD cada vez. Si
`wann_car_odin_eval` intenta escribir por `/dev/mem` a un periférico
AXI-Lite que no está clockeado (porque el overlay no se cargó en este boot),
el core ARM puede quedar **completamente trabado** — no solo el proceso, todo
el sistema, incluido SSH. No hay forma de recuperarlo por software; la única
salida es un power cycle físico del board.

**Regla fija: correr el Paso 1 de abajo (`odin_bootstrap.py`) en CADA power
cycle / reinicio, antes de tocar cualquier binario que hable con ODIN.**

### Segundo incidente conocido (2026-09-22): cuelgue con `-n 4` tras un `-n 2` exitoso

Con el overlay ya cargado y un `-n 2` que había corrido bien, un `-n 4`
posterior volvió a trabar el sistema entero (SSH y Jupyter caídos, solo
recuperable con power cycle). A diferencia del primer incidente, acá el PL
sí estaba programado — la causa parece estar en el propio protocolo AER
entre episodios, no en el PL sin cargar. Dos huecos reales que se
corrigieron:

1. `OdinDriver` nunca dejaba el chip en estado seguro (`GATE_ACTIVITY=1`) al
   salir de un proceso — el siguiente proceso arrancaba su `init()` sin
   saber en qué estado había quedado el chip. Ahora el destructor llama
   `stop()` siempre (con `try/catch`, un destructor no puede tirar).
2. `OdinNetwork::applyInputSpikes()` ignoraba el código de retorno de
   `sendVirtual()` — si el handshake AER fallaba (bus desincronizado), el
   código seguía mandando eventos a un bus ya roto en vez de frenar. Ahora
   tira una excepción clara ante el primer fallo.
3. `OdinNetwork::fastReset()` ahora llama `driver_.aerBusReset()` al
   principio de cada episodio, para limpiar cualquier resto de handshake del
   episodio anterior antes de mandar eventos nuevos.

**Esto reduce el riesgo pero no lo elimina.** Si el cuelgue es a nivel del
bus AXI (una transacción de `/dev/mem` que nunca vuelve, no solo un timeout
lógico del protocolo AER), ningún código a nivel de aplicación puede
prevenirlo — eso depende de cómo esté armado el diseño de Vivado (si tiene
un monitor de timeout AXI o no). Por eso, hasta confirmar que el fix
alcanza, seguí el protocolo de reintento conservador de abajo en vez de

### Causa raíz encontrada (mismo día): el chip nunca se arrancaba

Con el fix de arriba puesto, el primer intento posterior falló limpio (sin
colgar el sistema — la excepción hizo su trabajo) con
`aer_in handshake failed (rc=-2)`. La causa: `OdinDriver::loadConfig()` deja
el chip parado a propósito (`GATE_ACTIVITY=1`, necesario para programar
neuronas/sinapsis con seguridad) y el comentario decía "el que llama debe
hacer `start()`" — pero **nada en el código real llamaba a `start()`**.
`OdinNetwork` nunca arrancaba el chip después de cargarle la config. Todas
las corridas de hardware hasta ahora (incluidas las que "terminaron" sin
tirar error, antes de agregar la revisión del código de retorno) corrieron
contra un chip parado: cada `sendVirtual` esperaba en silencio el timeout
completo de 100ms sin que nada respondiera del otro lado — lo cual también
explica buena parte de la demora que se venía observando, más allá del
timeout fijo de `drainSpikes`.

**Corregido**: `OdinNetwork` ahora llama `driver_.start()` justo después de
`loadConfig()`. No hay garantía de que esto sea la única causa del cuelgue
con `-n 4` (pudo haber contribuido: mantener `REQ` reintentando contra un
chip parado durante más tiempo/episodios es plausible que empeore las cosas
a nivel de bus), así que seguí igual el protocolo de reintento escalonado de
abajo en vez de asumir que ya está resuelto del todo.
volver directo a `-n 4`.

**Protocolo de reintento después de un power cycle:**
1. `odin_bootstrap.py` (Paso 1) + verificar `fpga_manager` en `operating`.
2. Recompilar (`cmake --build build --target wann_car_odin_eval -j$(nproc)`)
   para traer el fix.
3. Un solo episodio: `sudo ./build/wann_car_odin_eval ... -n 1 -m <ventana>`.
   Confirmá que termina limpio antes de seguir.
4. Un segundo proceso separado con `-n 1` de nuevo (invocación nueva, no la
   misma) — para ver si el problema es "entre procesos" y no "entre
   episodios dentro de un mismo proceso".
5. Recién si 3 y 4 salen bien, subir a `-n 2`, después `-n 4` — de a un
   escalón, no saltando directo al número que rompió antes.

## Paso 1 — Cargar el overlay (una vez por power cycle)

```bash
cd /home/xilinx/jupyter_notebooks/WANN/wann-cpp   # o donde tengas el repo en el board
sudo -E python3 odin_bootstrap.py /home/xilinx/jupyter_notebooks/neat_comun_ranc_odin/odin.bit
```

`-E` es importante: preserva `XILINX_XRT` y el resto del entorno de tu shell
(sin eso, `pyxrt` falla con `RuntimeError: No Devices Found` o
`ModuleNotFoundError: No module named 'pynq'` si `sudo` te resetea el `PATH`
del venv). Si `sudo -E` no alcanza, la alternativa robusta es apuntar
directo al intérprete del venv:
```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 odin_bootstrap.py <ruta>/odin.bit
```

El script imprime las 5 direcciones físicas de los IP cores (`axi_gpio_0..3`,
`axi_quad_spi_0`) — ya están cargadas en `include/wann/hw/OdinRegisters.h`
(`ODIN_REGISTERS_CONFIGURED = true`), así que solo hace falta volver a
correrlo si cambia el overlay/bitstream.

**Verificación antes de seguir** (confirma que el PL quedó realmente
programado):
```bash
cat /sys/class/fpga_manager/fpga0/state
# tiene que decir: operating
```
Si no dice `operating`, **no sigas** — algo falló en el paso anterior.

## Paso 2 — Compilar snn-simulator

El board no tiene Catch2 v3 (solo v2 por apt), así que hay que desactivar
los tests para no depender de él:
```bash
cd ../snn-simulator
git pull   # asegurate de tener el commit con Network::setTimeConstants
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j$(nproc)
```

## Paso 3 — Compilar wann-cpp

```bash
cd ../wann-cpp
git pull
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWANN_ODIN_HW=ON
cmake --build build -j$(nproc) --target \
    wann_car_odin_export wann_car_odin_eval wann_eval_weights_car
```

`wann_eval_weights_car` no es parte del camino ODIN en sí, pero lo necesita
`bootstrap_compare_car_odin_auto.py` (el flujo con `--run-key`, ver Paso 5)
para la revalidación/selección del ganador por software.

Si el linker se queja de un símbolo indefinido tipo
`Network::setTimeConstants` u otro método de `snn-simulator` que no
existía antes: el `snn-simulator` del board está desactualizado respecto al
`wann-cpp` que estás compilando — volvé al Paso 2 y hacé `git pull` ahí.

## Paso 4 — Exportar un modelo a configuración ODIN

```bash
./build/wann_car_odin_export -f <ruta al .out> -d p/car_snn.json -o odin_config.json
```

- `-w <índice>` para forzar un peso compartido específico (0-5, ver
  `SnnCarTask::WEIGHT_VALS`); si no se pasa, lee el `.wi` que debería estar
  al lado del `.out` (mismo nombre, extensión `.wi`).
- Qué hace además de convertir pesos/neuronas: si algún nodo tiene salidas
  con signo mixto (excitatorio e inhibitorio a la vez — común, porque
  `mutToggleExcitatory` no respeta la ley de Dale), lo separa en dos
  direcciones físicas ODIN que reciben las mismas entradas y disparan en
  lockstep (real hardware solo permite un signo por neurona pre-sináptica).
  Esto aumenta la cuenta de neuronas usadas; si supera 256, falla con un
  error explícito — no hay forma de desplegar esa red tal cual.
- Las neuronas se mapean a modo Izhikevich real del chip (`neuron_core.v`,
  bit 0 de la palabra de 128 bits = 0), **solo para los 6 comportamientos
  clásicos** que `car_snn.json` evoluciona (`REGULAR_SPIKING`,
  `FAST_SPIKING`, `CHATTERING`, `LOW_THRESHOLD_SPIKING`,
  `INTRINSICALLY_BURSTING`, `RESONATOR` — `ann_actRange=[1..6]`). Ver el
  caveat importante más abajo.

## Paso 5 — Correr episodios en hardware real

```bash
sudo ./build/wann_car_odin_eval -c odin_config.json -d p/car_snn.json \
    -n 10 -s 0 -m <window_ms>
```

- `-m/--window-ms` **tiene que coincidir** con el `snn_window_ms` con el que
  se entrenó ese modelo específico (campo de `Hyperparams`, no siempre el
  default de 40 — un modelo podado puede usar otra ventana, p.ej. 20).
- `--csv`: imprime solo `episode,reward` a stdout (todo lo demás a stderr) —
  para que un script lo capture limpio.
- `--profile`: al final imprime cuánto tiempo se fue en el entorno RL vs.
  encoder vs. envío AER vs. espera de `AER_OUT` vs. decoder — útil para ver
  en qué se va el tiempo real en tu chip (ver "Rendimiento" más abajo).
- `sudo` es necesario porque `OdinDriver` abre `/dev/mem` directo.

El arranque (`init()`, dentro del constructor de `OdinNetwork`) resetea las
65536 sinapsis + 256 neuronas antes de cargar tu red — vas a ver varios
prints (`[1/5]`...`[5/5]`) y puede tardar de menos de un segundo a varios
segundos según qué tan rápido responda el bus SPI real. Es normal, no es un
cuelgue mientras sigan apareciendo líneas nuevas.

## Paso 6 — Comparación estadística contra el ANN

Dos scripts, según de dónde salga el modelo:

**A) Viene de un run de `screening_full.py --mode phase3` (tiene `run_key`,
vive en `log/full_p3_<run_key>/`)** — revalida automáticamente, elige el
(seed, peso) ganador, y compara:
```bash
sudo -E python3 bootstrap_results/bootstrap_compare_car_odin_auto.py \
    --run-key car_ttfs_first_spike_40ms --window-ms 40 --n 30
```

**B) Es un `.out` suelto (p.ej. una red podada a mano, sin `run_key`)** — va
directo al bootstrap, sin revalidación:
```bash
sudo -E python3 bootstrap_results/bootstrap_compare_car_odin_manual.py \
    --model-file <ruta al .out> --window-ms 20 --n 30
```
(`--weight-index N` si no hay `.wi` al lado del `.out`.)

Ambos producen `bootstrap_results/<algo>/{rewards,bootstrap_samples,
summary_stats,ci_results}.csv` + un `.png`, igual formato que
`bootstrap_compare_car_auto.py` (software), pero con la fila `agent=snn`
conteniendo en realidad los episodios de ODIN (así `replot_bootstrap.py`
sigue funcionando sin cambios) y las etiquetas de texto/gráfico dicen
"ODIN" en vez de "SNN".

## Rendimiento: por qué puede tardar

`OdinNetwork::step()` espera un mínimo fijo de 1 ms por cada tick interno de
simulación (`drainSpikes`), sin importar si hubo actividad — es un
placeholder sin calibrar todavía. Con ventana de N ms (`-m N`) hay N ticks
por paso de entorno, y hasta 1000 pasos por episodio → un piso de **N × 1 ms
× hasta 1000 = hasta N segundos por episodio**, solo por esa espera. Usá
`--profile` para medir en tu chip específico si de verdad es esa espera la
que domina, o si es otra cosa (protocolo AER más lento de lo esperado,
entorno RL, etc.) antes de asumir nada.

## Caveats importantes

- **Izhikevich, solo 6 comportamientos**: el chip soporta 20 comportamientos
  Izhikevich, pero ODIN nunca publicó la tabla de qué combinación de bits
  reproduce cada uno — `izhParamsFor()` en `OdinExport.cpp` es una
  interpretación propia de la RTL pública (`izh_neuron.v`,
  `izh_neuron_state.v`), acotada a los 6 que este modelo realmente usa. Los
  valores numéricos (`thr`, `rfr`, `spk_ref`, etc.) son punto de partida, no
  calibrados — si el chip no dispara o dispara todo el tiempo con un
  comportamiento en particular, ese es el primer lugar para ajustar. Si
  algún día el genoma evoluciona con `ann_actRange` más amplio,
  `izhParamsFor()` lanza un error explícito en vez de aproximar mal.
- **Ley de Dale**: ver Paso 4 — nodos con signo mixto se duplican
  automáticamente; si eso hace que la red supere 256 neuronas, no hay forma
  de desplegarla tal cual (habría que podarla más).
- **Solo ttfs/first_spike**: `SnnCarOdinTask` solo soporta
  `snn_encoder=ttfs`/`snn_decoder=first_spike` — es lo único que se probó y
  lo que usan los modelos actuales. Otra combinación tira un error explícito
  al construir la tarea.
- **`SnnCarOdinTask` es una implementación separada** de `SnnCarTask.cpp`
  (no un template compartido), a propósito, para no arriesgar el código de
  evolución activo. Si `SnnCarTask.cpp` cambia normalización de observaciones
  o constantes (`BOUND`, `VX_MAX`, etc.), hay que replicar el cambio acá
  también — no se actualiza solo.

## Troubleshooting rápido

| Síntoma | Causa | Solución |
|---|---|---|
| `Could not find... Catch2... version "3"` compilando snn-simulator | Board solo tiene Catch2 v2 | `-DBUILD_TESTING=OFF` al configurar |
| `wann_eval_weights_car no existe` | Falta compilar ese target | `cmake --build build --target wann_eval_weights_car` |
| `undefined reference to Network::setTimeConstants` (u otro símbolo) | `snn-simulator` desactualizado | `git pull` en `../snn-simulator` y recompilar |
| `no se encontraron modelos de fase 3 para run_key=...` | El modelo no viene de `screening_full.py --mode phase3` | Usar `bootstrap_compare_car_odin_manual.py`, no el `_auto.py` |
| `ModuleNotFoundError: No module named 'pynq'` bajo `sudo` | `sudo` resetea el `PATH`, pierde el venv | `sudo -E ...` o `sudo /usr/local/share/pynq-venv/bin/python3 ...` |
| `RuntimeError: No Devices Found` / "is the XRT environment sourced?" | Falta `XILINX_XRT` bajo `sudo` | `sudo -E ...` (preservá el entorno, no lo vuelvas a sourcear a mano) |
| `OdinDriver: cannot open /dev/mem` | No corriste con `sudo` | Anteponer `sudo` al binario |
| `OdinRegisters.h still has placeholder...` | Direcciones físicas sin completar | Correr `odin_bootstrap.py`, copiar los `phys_addr` a `OdinRegisters.h` |
| **El sistema entero deja de responder (SSH incluido)** | El overlay no estaba cargado en este boot y `/dev/mem` pegó contra un periférico sin clock | Power cycle físico. Después: SIEMPRE Paso 1 antes de tocar hardware |

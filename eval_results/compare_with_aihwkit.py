#!/usr/bin/env python3
"""
Compara NUESTROS modelos de ruido de memristor con los de aihwkit REAL, peso a peso y por mecanismo.

  Lado aihwkit : ReRamCMONoiseModel (single, su conversor por defecto), el mismo modelo con SinglePairConductanceConverter
                 (pair = opción B) y PCMLikeNoiseModel, usando su API de inferencia (`apply_noise`).
  Lado nuestro : (a) sgrl/noise.py de ../surrogate-gradient-rl (puerto PyTorch, ya con apply_sgrl_cmo_mapping.py aplicado) y
                 (b) build/memnoise_stats (puerto C++ de memristor_noise.hpp, el de wann_*_noise).

Como los generadores aleatorios son distintos no se compara muestra a muestra:
  * casos DETERMINISTAS (las tres escalas de ruido en 0): los pesos leídos deben coincidir (|Δ| < 1e-5·max|w|);
  * casos ALEATORIOS: media y desviación POR PESO de N muestras (programar un chip + leer). Pasa si |Δmedia| < 5 errores
    estándar y el cociente de desviaciones está en 1 ± 5.5·sqrt((curtosis-1)/(2(N-1))) (≈ ±10 % si es gaussiana, N = 3000;
    más ancho donde la deriva aprieta los dispositivos contra g_min/g_max y casi todas las muestras caen en un punto). Se prueba cada mecanismo por separado (solo programación,
    solo lectura, solo deriva y los tres juntos), varios tiempos y matrices tipo SG (densa, con ceros) y tipo WANN (±L).
  * ocupación (--occupancy-check): los barridos usan ocupaciones 6.25-100 % de g_max, una extensión nuestra que aihwkit no tiene
    (w_max = max|w|/occ). Se comprueba haciendo que aihwkit calcule ese mismo mapeo con pesos ficticios que ensanchan el rango
    (augment_for_occupancy) y comparando solo los pesos reales.
  * informativo: `apply_drift_noise` de aihwkit (flujo program_analog_weights -> drift_analog_weights) vuelve a calcular
    el mínimo/máximo del tensor ya perturbado; se mide cuánto se aparta ese camino de `apply_noise`.

Cómo obtener aihwkit (no hace falta compilar nada: los modelos de ruido son PyTorch puro):
  A) instalado:        pip install aihwkit            (si hay rueda para tu Python)
  B) desde el código:  git clone https://github.com/IBM/aihwkit  y pasar --aihwkit-src /ruta/a/aihwkit
     (carga solo inference/{noise,converter} sin importar las extensiones compiladas).
Ejecuta código del repo de aihwkit que indiques: usá uno que hayas clonado vos.

Uso (desde la raíz del repo):
  python eval_results/compare_with_aihwkit.py --aihwkit-src ~/aihwkit [--sgrl-repo ../surrogate-gradient-rl]
        [--memnoise-bin build/memnoise_stats] [--n 3000] [--quick] [--models reramcmo pcm] [--out resultados.csv]
        [--occupancy-check [--occupancies 0.0625 0.125 0.25 0.5] [--skip-base]]
Códigos de salida: 0 = todo coincide, 1 = hay diferencias (se listan), 2 = falta algo (aihwkit, SG o el binario).
Solo evalúa pesos sueltos: no simula redes ni entrena.
"""
import argparse
import csv
import itertools
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

# Valores por defecto de aihwkit para ReRAM-CMO (SingleDeviceConductanceConverter) y PCM (SinglePairConductanceConverter)
CMO_GMAX, CMO_GMIN, PCM_GMAX = 88.199997, 9.0, 25.0
TIMES_ALL = (1.0, 3600.0, 86400.0, 31536000.0)
MECHS = {"none": (0, 0, 0), "prog": (1, 0, 0), "read": (0, 1, 0), "drift": (0, 0, 1), "all": (1, 1, 1)}


# ----------------------------------------------------------------------------------------------------------------------
# Carga de aihwkit
# ----------------------------------------------------------------------------------------------------------------------
def _stub_package(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    m.__package__ = name
    sys.modules[name] = m


def load_aihwkit(src):
    """Devuelve (ReRamCMONoiseModel, PCMLikeNoiseModel, SinglePairConductanceConverter, origen)."""
    if src:
        root = Path(src).expanduser().resolve()
        pkg = root / "src" / "aihwkit" if (root / "src" / "aihwkit").is_dir() else root / "aihwkit"
        if not (pkg / "inference" / "noise" / "reram.py").is_file():
            sys.exit(f"No encuentro inference/noise/reram.py bajo {root} (¿es un clon de IBM/aihwkit?)")
        # Paquetes "vacíos" que apuntan al código fuente: así no se ejecutan los __init__.py (que importan las extensiones
        # compiladas) y se pueden importar solo los módulos de ruido.
        for name, path in (("aihwkit", pkg), ("aihwkit.inference", pkg / "inference"),
                           ("aihwkit.inference.noise", pkg / "inference" / "noise"),
                           ("aihwkit.inference.converter", pkg / "inference" / "converter")):
            _stub_package(name, path)
        try:
            import importlib
            importlib.import_module("aihwkit.exceptions")
        except Exception:                                   # exceptions.py importa algo pesado: basta con la clase
            ex = types.ModuleType("aihwkit.exceptions")
            ex.ArgumentError = type("ArgumentError", (Exception,), {})
            sys.modules["aihwkit.exceptions"] = ex
        origin = f"código fuente en {pkg}"
    else:
        origin = "instalado"
    try:
        from aihwkit.inference.noise.reram import ReRamCMONoiseModel
        from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
        from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
    except Exception as e:  # noqa: BLE001
        sys.exit(f"No pude importar aihwkit ({e}).\n  Instalalo (pip install aihwkit) o pasá --aihwkit-src DIR con un clon "
                 f"de https://github.com/IBM/aihwkit")
    import aihwkit
    ver = getattr(aihwkit, "__version__", "?")
    return ReRamCMONoiseModel, PCMLikeNoiseModel, SinglePairConductanceConverter, f"{origin} (versión {ver})"


def load_ours(sgrl_repo):
    sgrl_repo = Path(sgrl_repo).expanduser().resolve()
    if not (sgrl_repo / "sgrl" / "noise.py").is_file():
        sys.exit(f"No encuentro {sgrl_repo}/sgrl/noise.py (--sgrl-repo)")
    sys.path.insert(0, str(sgrl_repo))
    from sgrl import noise as N
    if not hasattr(N.ReRamCMOModel, "g_offset"):
        sys.exit("sgrl/noise.py no tiene el mapeo CMO. Aplicá:\n  python eval_results/apply_sgrl_cmo_mapping.py "
                 f"{sgrl_repo}")
    if "aihwkit: claves en PORCENTAJE" not in (sgrl_repo / "sgrl" / "noise.py").read_text(encoding="utf-8"):
        sys.exit("sgrl/noise.py tiene el paso 1 pero no el paso 2 (réplica exacta de aihwkit). Corré de nuevo "
                 f"apply_sgrl_cmo_mapping.py {sgrl_repo}")
    return N


# ----------------------------------------------------------------------------------------------------------------------
# Muestreo
# ----------------------------------------------------------------------------------------------------------------------
def test_weights(kind):
    import torch
    if kind == "sg":        # capa densa con pesos de magnitudes dispares, ceros y valores chicos
        g = torch.Generator().manual_seed(0)
        w = torch.cat([torch.randn(1, 44, generator=g) * 0.4, torch.tensor([[1.0, -1.0, 0.0, 0.02]])], 1)
    else:                   # WANN: todos ±L
        w = torch.tensor([[5., -5., 5., 5., -5., -5., 5., -5., 5., 5., -5., 5.]])
    return w.float()


def aihwkit_model(kind, mapping, scales, classes):
    cmo, pcm, pair_conv, _ = classes
    p, r, d = scales
    if kind == "pcm":
        return pcm(g_max=PCM_GMAX, prog_noise_scale=p, read_noise_scale=r, drift_scale=d)
    m = cmo(g_max=CMO_GMAX, g_min=CMO_GMIN, prog_noise_scale=p, read_noise_scale=r, drift_scale=d)
    if mapping == "pair":
        m.g_converter = pair_conv(g_max=CMO_GMAX, g_min=CMO_GMIN)
    return m


def augment_for_occupancy(W, kind, mapping, occ):
    """aihwkit no tiene "ocupación" (siempre mapea el máximo a g_max), pero se puede hacer que ÉL calcule el mapeo de
    ocupación `occ` añadiendo pesos ficticios que ensanchen el rango, y descartar su lectura:
      * pair / PCM : un peso ficticio de magnitud max|w|/occ  -> escala = (g_max-g_min)·occ/max|w|, igual que w_max = max|w|/occ.
      * single     : pesos ficticios en c ± R/occ (c = centro y R = radio de [min W, max W]) -> el rango [c-R/occ, c+R/occ] va a
                     [g_min, g_max], o sea g = g_c + (w-c)·occ/(2R)·(g_max-g_min): el mapeo centrado que usamos (en pesos
                     simétricos, ±max|w|/occ)."""
    import torch
    if occ >= 1.0:
        return W
    if kind == "pcm" or mapping == "pair":
        extra = [float(W.abs().max()) / occ]
    else:
        lo, hi = float(W.min()), float(W.max())
        c, R = 0.5 * (lo + hi), 0.5 * (hi - lo)
        extra = [c - R / occ, c + R / occ]
    return torch.cat([W, torch.tensor([extra], dtype=W.dtype)], 1)


def sample_aihwkit(model, W, t, n, seed, two_step=False, aug=None):
    import torch
    torch.manual_seed(seed)
    Win = W if aug is None else aug
    k = W.numel()
    out = np.empty((n, k))
    try:
        for i in range(n):
            if two_step:
                wp, nu = model.apply_programming_noise(Win.clone())
                res = model.apply_drift_noise(wp, nu, t)
            else:
                res = model.apply_noise(Win.clone(), t)
            out[i] = res.reshape(-1)[:k].double().numpy()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"El propio código de aihwkit falló al muestrear ({type(e).__name__}: {e}).\n"
                 f"  Suele ser incompatibilidad de versiones (aihwkit aplica numpy.log10/sqrt a tensores de torch).\n"
                 f"  torch {torch.__version__}, numpy {np.__version__}: probá otra versión de numpy o de aihwkit.")
    return out


def sample_sgrl(N, kind, mapping, scales, W, t, n, seed, occ=1.0):
    import torch
    p, r, d = scales
    if kind == "pcm":
        m = N.PCMModel(g_max=PCM_GMAX, prog_noise_scale=p, read_noise_scale=r, drift_scale=d)
    else:
        m = N.ReRamCMOModel(g_max=CMO_GMAX, g_min=CMO_GMIN, prog_noise_scale=p, read_noise_scale=r, drift_scale=d,
                            mapping=mapping)
    gen = torch.Generator().manual_seed(seed)
    wmax = float(W.abs().max()) / occ
    out = np.empty((n, W.numel()))
    for i in range(n):
        x = N.Crossbar(W, wmax, m)
        x.program(gen)
        out[i] = x.read(t, gen).reshape(-1).double().numpy()
    return out


def stats_cpp(binary, kind, mapping, scales, W, t, n, seed, occ=1.0):
    wl = ",".join(f"{v:.9g}" for v in W.reshape(-1).tolist())
    cmd = [str(binary), "--model", "pcm" if kind == "pcm" else "reramcmo", "--weights", wl, "--w-max", f"{float(W.abs().max()) / occ:.9g}",
           "--t", repr(t), "--n", str(n), "--prog", str(scales[0]), "--read", str(scales[1]), "--drift", str(scales[2]),
           "--seed", str(seed)]
    if kind != "pcm":
        cmd += ["--mapping", mapping, "--g-max", repr(CMO_GMAX), "--g-min", repr(CMO_GMIN)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"{binary} falló: {res.stderr.strip()}")
    rows = [tuple(map(float, l.split())) for l in res.stdout.strip().splitlines()[1:]]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


# ----------------------------------------------------------------------------------------------------------------------
# Comparación
# ----------------------------------------------------------------------------------------------------------------------
def compare(mean_a, sd_a, mean_b, sd_b, n, scale, deterministic, kurt=None):
    """-> (ok, max|z|, ratio_min, ratio_max, max|Δmedia|/scale)."""
    dm = np.abs(mean_a - mean_b)
    tiny = 1e-5 * scale
    if deterministic:
        ok = bool(dm.max() < tiny and sd_a.max() < tiny and sd_b.max() < tiny)
        return ok, 0.0, 1.0, 1.0, float(dm.max() / scale)
    both_det = (sd_a < tiny) & (sd_b < tiny)
    se = np.sqrt((sd_a ** 2 + sd_b ** 2) / n)
    z = np.where(both_det, np.where(dm < tiny, 0.0, np.inf), dm / np.maximum(se, 1e-300))
    ratio = np.where(both_det, 1.0, sd_a / np.maximum(sd_b, 1e-300))
    # Error estándar del cociente de dos desviaciones: sqrt((kurt-1)/(2n)) (= 1/sqrt(n) si es gaussiana, kurt = 3). Con deriva a 1 año
    # los dispositivos pegados a g_min/g_max quedan contra el recorte y casi todas las muestras caen en un punto: la curtosis es
    # enorme y la desviación depende de unas pocas muestras de la cola, así que se usa la curtosis medida (muestra de aihwkit).
    k = np.full_like(sd_a, 3.0) if kurt is None else np.maximum(kurt, 1.0)
    tol = 5.5 * np.sqrt((k - 1.0) / (2.0 * (n - 1)))
    r_ok = np.where(both_det, True, np.abs(ratio - 1.0) <= tol)
    ok = bool(np.max(np.abs(z)) < 5.0 and r_ok.all())
    return ok, float(np.max(np.abs(z))), float(ratio.min()), float(ratio.max()), float(dm.max() / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihwkit-src", default=None, help="clon de IBM/aihwkit (si no, se usa el aihwkit instalado)")
    ap.add_argument("--sgrl-repo", default=str(_ROOT.parent / "surrogate-gradient-rl"))
    ap.add_argument("--memnoise-bin", default=str(_ROOT / "build" / "memnoise_stats"))
    ap.add_argument("--n", type=int, default=3000, help="muestras por caso y método")
    ap.add_argument("--quick", action="store_true", help="solo t = 1 s y 1 año, N/3: para probar que todo carga")
    ap.add_argument("--models", nargs="+", default=["reramcmo", "pcm"], choices=["reramcmo", "pcm"])
    ap.add_argument("--no-cpp", action="store_true")
    ap.add_argument("--no-sgrl", action="store_true")
    ap.add_argument("--occupancy-check", action="store_true",
                    help="además (o solo, con --skip-base) prueba ocupaciones < 1, la extensión nuestra que usan los barridos: "
                         "aihwkit calcula el mismo mapeo añadiendo pesos ficticios (ver augment_for_occupancy)")
    ap.add_argument("--occupancies", type=float, nargs="+", default=[0.0625, 0.125, 0.25, 0.5])
    ap.add_argument("--skip-base", action="store_true", help="omitir la grilla base (ocupación 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV con todos los casos")
    a = ap.parse_args()

    import torch  # noqa: F401  (falla temprano y con mensaje claro si no está)
    classes = load_aihwkit(a.aihwkit_src)
    print(f"aihwkit: {classes[3]}")
    N = None if a.no_sgrl else load_ours(a.sgrl_repo)
    binary = Path(a.memnoise_bin)
    if not a.no_cpp and not binary.is_file():
        print(f"No existe {binary}. Compilá: cmake --build build --target memnoise_stats  (o usá --no-cpp)")
        sys.exit(2)
    n = max(300, a.n // 3) if a.quick else a.n
    times = (1.0, 31536000.0) if a.quick else TIMES_ALL

    cases = []
    for kind in a.models:
        for mapping in (("single", "pair") if kind == "reramcmo" else ("pair",)):
            if not a.skip_base:
                for wk, mech, t in itertools.product(("sg", "wann"), MECHS, times):
                    cases.append((kind, mapping, wk, mech, t, 1.0))
            if a.occupancy_check:
                for occ, wk, mech, t in itertools.product(a.occupancies, ("sg", "wann"), ("none", "prog", "all"), (1.0, 31536000.0)):
                    cases.append((kind, mapping, wk, mech, t, occ))
    if not cases:
        sys.exit("No hay casos: --skip-base sin --occupancy-check")
    print(f"{len(cases)} casos, {n} muestras por caso y método\n")
    print(f"{'caso':52s} {'vs':6s} {'ok':3s} {'max|z|':>7s} {'sd min..max':>13s} {'max|Δ|/max|w|':>14s}")

    rows, bad = [], []
    for ci, (kind, mapping, wk, mech, t, occ) in enumerate(cases):
        W = test_weights(wk)
        scales = MECHS[mech]
        det = mech == "none"
        model = aihwkit_model(kind, mapping, scales, classes)
        ref = sample_aihwkit(model, W, t, n, a.seed + ci, aug=augment_for_occupancy(W, kind, mapping, occ))
        m_a, s_a = ref.mean(0), ref.std(0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(s_a > 1e-12, ((ref - m_a) ** 4).mean(0) / np.maximum(ref.var(0), 1e-300) ** 2, 3.0)
        scale = float(W.abs().max())
        label = f"{kind}/{mapping}/{wk}/{mech}/t={t:g}" + (f"/occ={occ:g}" if occ < 1 else "")
        results = []
        if N is not None:
            o = sample_sgrl(N, kind, mapping, scales, W, t, n, a.seed + 1000 + ci, occ)
            results.append(("SG", m_a, s_a, o.mean(0), o.std(0, ddof=1)))
        if not a.no_cpp:
            cm, cs = stats_cpp(binary, kind, mapping, scales, W, t, n, a.seed + 2000 + ci, occ)
            results.append(("C++", m_a, s_a, cm, cs))
        for who, ma, sa, mb, sb in results:
            ok, zmax, r0, r1, dmax = compare(mb, sb, ma, sa, n, scale, det, kurt)  # (nuestro) vs (aihwkit)
            print(f"{label:52s} {who:6s} {'ok' if ok else 'NO':3s} {zmax:7.2f} {r0:6.3f}..{r1:5.3f} {dmax:14.2e}")
            rows.append(dict(case=label, vs=who, ok=ok, max_z=zmax, sd_ratio_min=r0, sd_ratio_max=r1, max_dmean_rel=dmax))
            if not ok:
                bad.append((label, who))

    # informativo: camino de dos pasos de aihwkit (program_analog_weights -> drift_analog_weights) frente a apply_noise
    print("\nInformativo: aihwkit en dos pasos (apply_programming_noise + apply_drift_noise) frente a apply_noise "
          "(los dos son aihwkit; mide el efecto de que el segundo paso renormaliza min/max)")
    for kind, mapping, wk in (("reramcmo", "single", "sg"), ("reramcmo", "pair", "sg"), ("pcm", "pair", "sg")):
        if kind not in a.models:
            continue
        W, sc = test_weights(wk), MECHS["all"]
        for t in (1.0, 31536000.0):
            m1 = aihwkit_model(kind, mapping, sc, classes)
            r1_ = sample_aihwkit(m1, W, t, n, a.seed + 5000)
            r2_ = sample_aihwkit(aihwkit_model(kind, mapping, sc, classes), W, t, n, a.seed + 6000, two_step=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                k1 = np.where(r1_.std(0) > 1e-12, ((r1_ - r1_.mean(0)) ** 4).mean(0) / np.maximum(r1_.var(0), 1e-300) ** 2, 3.0)
            ok, zmax, q0, q1, dmax = compare(r2_.mean(0), r2_.std(0, ddof=1), r1_.mean(0), r1_.std(0, ddof=1), n,
                                             float(W.abs().max()), False, k1)
            print(f"  {kind}/{mapping}/{wk}/all/t={t:g}: max|z|={zmax:.2f}  sd {q0:.3f}..{q1:.3f}  max|Δmedia|/max|w|={dmax:.2e}")

    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {a.out}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} comparaciones coinciden con aihwkit.")
    if bad:
        print("DIFERENCIAS en:")
        for label, who in bad:
            print(f"  {label} ({who})")
        sys.exit(1)


if __name__ == "__main__":
    main()

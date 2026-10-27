#!/usr/bin/env python3
"""
Figuras de la comparación WANN vs. gradientes surrogados (SGRL, en varias variantes de
tamaño de red) bajo ruido de memristores, para una tarea (car, acrobot o disc_mc). Lee lo que consolida
run_noise_compare.py (sweep_all.csv + models.csv + protocol.json en --data-dir);
no simula nada, se puede relanzar en segundos (p. ej. en tu máquina, tras copiar
los CSV desde el clúster).

Figuras (PNG en --out-dir, nombre <prefix>_<tipo>.png como el resto de graficos/):
  noise_robustness   retorno y retención vs. escala de ruido, un panel por dispositivo
  time_robustness    retorno y retención vs. tiempo desde la programación (escala fija)
  rmse_robustness    retorno y retención vs. error efectivo de pesos (eje común)
  heatmap_<modelo>   retención escala×tiempo de cada método (fila 1) y su diferencia con WANN (fila 2)
  ablation_<modelo>  un mecanismo activo cada vez (programación / lectura / deriva)
  summary            barras: retención de cada método por dispositivo a escalas de ruido fijas
Además escribe aggregated.csv (todas las celdas × modelos) para tablas.

Métodos (colores de Okabe-Ito, separables con daltonismo; cada uno con su marcador):
  wann      WANN                                 azul      ○   la red ganadora, peso ganador
  sgrl      SGRL, red 64×64 (Optuna original)   naranja   □   la mejor semilla
  sgrl_con  SGRL con las CONEXIONES de WANN      verde     △   su mejor modelo
  sgrl_neu  SGRL con las NEURONAS de WANN        púrpura   ◇   su mejor modelo
--sources restringe cuáles se dibujan (p. ej. `--sources wann sgrl` reproduce las figuras de 2 métodos).

Qué se dibuja
  - Línea gruesa con banda = IC 95 % por bootstrap remuestreando CHIPS (las
    lecturas de un mismo chip no son independientes), del modelo principal de
    cada método: WANN = red ganadora con su peso ganador; SGRL = su mejor semilla/modelo.
  - Líneas finas = el resto: otros pesos compartidos de la MISMA red WANN, y las
    otras semillas SGRL 64×64 (redes distintas); las variantes por conexiones/neuronas
    traen un solo modelo. Muestran cuánto depende el resultado
    de la elección del ganador.
  - Retención = (retorno - piso) / (retorno sin ruido de ESE mismo modelo - piso)
    × 100, con piso = recompensa del peor desenlace (protocol.json: car 0, que
    reduce esto a retorno / retorno sin ruido; Mountain Car discreto -200;
    Acrobot -500, ambos con -1 por paso). Los retornos sin ruido de ambos
    métodos no son iguales, así que la retención es la comparación justa de
    robustez; el retorno absoluto está en la fila de arriba. El denominador se
    trata como constante (la banda no incluye su incertidumbre). Se omite si el
    retorno sin ruido no supera el piso.

Uso:
  python eval_results/plot_noise_compare.py --data-dir eval_results/car_wann_vs_sgrl
  python eval_results/plot_noise_compare.py --data-dir eval_results/acrobot_wann_vs_sgrl --out-dir /tmp/figs --only noise,heatmap
"""
import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR.parent)           # ancla CWD a la raíz del repo
sys.path.insert(0, str(_SCRIPT_DIR))   # para importar plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from plot_style import TIME_TICK_LABELS, setup_publication_style

# Color = método (Okabe-Ito, separable con daltonismo) y marcador distinto por método, para que no
# dependa solo del color. No son los colores por dispositivo de plot_style.MODEL_COLORS: aquí cada
# panel ya es un dispositivo.
METHODS = {
    "wann":     dict(color="#0072B2", marker="o", name="WANN"),
    "sgrl":     dict(color="#E69F00", marker="s", name="SGRL"),
    "sgrl_con": dict(color="#009E73", marker="^", name="SGRL"),
    "sgrl_neu": dict(color="#CC79A7", marker="D", name="SGRL"),
}
COLORS = {k: v["color"] for k, v in METHODS.items()}
SG_WHAT = {"sgrl": "", "sgrl_con": ", same connections as WANN", "sgrl_neu": ", same neurons as WANN"}
SHORT = {"simple": "Simple (placeholder)", "pcm": "PCM", "reram": "ReRAM (Wan 2022)",
         "reramcmo": "ReRAM CMO/HfOx"}
MARKERS = {"simple": "D", "pcm": "o", "reram": "s", "reramcmo": "^"}
INTENSITY_T = {"simple": 0.0, "pcm": 0.0, "reram": 1.0, "reramcmo": 1.0}
MECH_TITLE = {"prog": "Programming noise only", "read": "Read noise only", "drift": "Drift only"}
KEYS = ["source", "model_id", "noise_model", "mechanism", "scale", "t_inference"]
RET_CMAP = LinearSegmentedColormap.from_list("ret", ["#F2F7FB", "#9CC3E0", "#2F6FA8", "#0B2F5B"])
# divergente: naranja (el método supera a WANN) - gris neutro - azul (WANN supera al método)
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", ["#0072B2", "#E8E8E8", "#E69F00"])


# --------------------------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def aggregate(df: pd.DataFrame, n_boot: int, floor: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Una fila por (modelo, celda): media sobre chips, IC bootstrap sobre chips, error de pesos."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, g in df.groupby(KEYS, sort=False, dropna=False):
        chip_means = g.groupby(["seed", "chip"])["ret"].mean().to_numpy()
        lo, hi = bootstrap_ci(chip_means, rng, n_boot)
        rows.append(dict(zip(KEYS, k), mean=chip_means.mean(), lo=lo, hi=hi, n_chips=len(chip_means),
                         rmse=g["weight_rmse"].mean()))
    agg = pd.DataFrame(rows)
    base = agg[agg.noise_model == "none"].set_index("model_id")["mean"]
    b = agg["model_id"].map(base)
    ok = b > floor
    for c in ("mean", "lo", "hi"):
        agg[f"ret_{c}"] = np.where(ok, (agg[c] - floor) / (b - floor) * 100.0, np.nan)
    agg["baseline"] = b
    return agg


def curve(agg, model_id, noise_model, mechanism, xcol, **fixed) -> pd.DataFrame:
    sel = (agg.model_id == model_id) & (agg.noise_model == noise_model) & (agg.mechanism == mechanism)
    for k, v in fixed.items():
        sel &= np.isclose(agg[k].to_numpy(dtype=float), v)
    return agg[sel].sort_values(xcol)


# --------------------------------------------------------------------------------------------
# Piezas de dibujo
# --------------------------------------------------------------------------------------------
class Series:
    """Qué modelos dibujar: uno principal por método y el resto como líneas finas."""

    def __init__(self, models: pd.DataFrame, present: set, sources=None):
        self.models = models[models.model_id.isin(present)]
        if sources:
            self.models = self.models[self.models.source.isin(sources)]
        self.main = {s: g[g.main == 1].model_id.tolist()[0] for s, g in self.models.groupby("source")
                     if (g.main == 1).any()}
        self.others = {s: [m for m in g.model_id if m != self.main.get(s)]
                       for s, g in self.models.groupby("source")}
        self.order = [s for s in METHODS if s in self.main]     # orden fijo: WANN, SGRL, con, neu

    def hidden(self, source: str) -> str:
        row = self.models[self.models.model_id == self.main[source]].iloc[0]
        h = str(row.get("hidden", "")) if "hidden" in row.index else ""
        return h.replace("x", "×") if h and h != "nan" else ""

    def label(self, source: str) -> str:
        row = self.models[self.models.model_id == self.main[source]].iloc[0]
        if source == "wann":
            return f"WANN, winner (shared weight idx {int(row.weight_index)})"
        h = self.hidden(source)
        suffix = ", best seed" if source == "sgrl" else ", best model"
        return f"SGRL {h}{SG_WHAT[source]}{suffix}".replace("SGRL ,", "SGRL,")

    def other_label(self, source: str) -> str | None:
        n = len(self.others.get(source, []))
        if n == 0:
            return None
        return (f"WANN, other shared weights (n={n})" if source == "wann"
                else f"SGRL {self.hidden(source)}, other seeds (n={n})")

    def handles(self):
        out = [Line2D([], [], color=COLORS[s], lw=2.6, marker=METHODS[s]["marker"], ms=6, label=self.label(s))
               for s in self.order]
        out += [Line2D([], [], color=COLORS[s], lw=1.0, alpha=0.6, ls="--", label=self.other_label(s))
                for s in self.order if self.other_label(s)]
        return out


def draw_curves(ax, agg, ser: Series, noise_model, mechanism, xcol, ycol, **fixed):
    """ycol: 'mean' o 'ret_mean'. Dibuja primero las finas y encima las principales con banda."""
    lo, hi = ("lo", "hi") if ycol == "mean" else ("ret_lo", "ret_hi")
    for s in ser.order:
        for mid in ser.others.get(s, []):
            c = curve(agg, mid, noise_model, mechanism, xcol, **fixed)
            if len(c):
                ax.plot(c[xcol], c[ycol], color=COLORS[s], lw=1.0, ls="--", alpha=0.4, zorder=2)
    for s in ser.order:
        c = curve(agg, ser.main[s], noise_model, mechanism, xcol, **fixed)
        if c.empty:
            continue
        ax.fill_between(c[xcol], c[lo], c[hi], color=COLORS[s], alpha=0.14, linewidth=0, zorder=3)
        ax.plot(c[xcol], c[ycol], color=COLORS[s], lw=2.4, marker=METHODS[s]["marker"], ms=6, zorder=4)


def finish_rows(axes, ylabel_abs="Mean episode reward", ylabel_ret="Retained performance (% of noise-free)"):
    for r, lab in enumerate((ylabel_abs, ylabel_ret)):
        axes[r][0].set_ylabel(lab)
        for ax in axes[r]:
            if r == 1:
                ax.axhline(100, color="0.5", lw=1, ls=":", zorder=1)
    for ax in axes[0]:
        lo = ax.get_ylim()[0]
        ax.set_ylim(bottom=min(0.0, lo))
    for ax in axes[1]:
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))


def set_time_axis(ax, ts):
    """Eje simétrico-log; funde ticks a < linthresh (t=0 y t=1 s) para que no se pisen las etiquetas."""
    lin = 60
    ax.set_xscale("symlog", linthresh=lin)
    s = sorted(set(float(t) for t in ts))
    groups, cur = [], [s[0]]
    for t in s[1:]:
        if t - cur[0] < lin:   # contra el primero del grupo: sin encadenar 0, 1 s, 60 s en un solo tick
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    reps = [next((t for t in g if t in TIME_TICK_LABELS), g[0]) for g in groups]
    ax.set_xticks(reps)
    ax.set_xticklabels([TIME_TICK_LABELS.get(int(t), str(t)) for t in reps], rotation=40, ha="right", fontsize=10)
    ax.minorticks_off()


def legend_and_save(fig, ser, out, note):
    hs = ser.handles()
    fig.legend(handles=hs, loc="lower center", ncol=2 if len(hs) <= 4 else 3, frameon=True,
               bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.07 if len(hs) > 4 else -0.05, note, ha="center", va="top", fontsize=10, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------------
def fig_intensity(agg, ser, models_present, out, note):
    ms = [m for m in ("simple", "pcm", "reram", "reramcmo") if m in models_present
          and not curve(agg, ser.main[ser.order[0]], m, "all", "scale", t_inference=INTENSITY_T[m]).empty]
    if not ms:
        return print("Sin datos de intensidad.")
    fig, axes = plt.subplots(2, len(ms), figsize=(5.4 * len(ms), 8.6), sharex=True, squeeze=False)
    for j, m in enumerate(ms):
        for r, y in enumerate(("mean", "ret_mean")):
            draw_curves(axes[r][j], agg, ser, m, "all", "scale", y, t_inference=INTENSITY_T[m])
        axes[0][j].set_title(f"{SHORT[m]}\n(t = {TIME_TICK_LABELS.get(int(INTENSITY_T[m]), INTENSITY_T[m])})", fontsize=14)
        axes[1][j].set_xlabel("Noise scale (1.0 = literature default)")
    finish_rows(axes)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    legend_and_save(fig, ser, out, note)


def fig_time(agg, ser, out, note, time_scale):
    ref = ser.main[ser.order[0]]
    ms = [m for m in ("pcm", "reram", "reramcmo")
          if curve(agg, ref, m, "all", "t_inference", scale=time_scale).shape[0] > 1]
    if not ms:
        return print("Sin datos de tiempo.")
    fig, axes = plt.subplots(2, len(ms), figsize=(5.4 * len(ms), 8.6), squeeze=False)
    for j, m in enumerate(ms):
        for r, y in enumerate(("mean", "ret_mean")):
            draw_curves(axes[r][j], agg, ser, m, "all", "t_inference", y, scale=time_scale)
        c = curve(agg, ref, m, "all", "t_inference", scale=time_scale)
        for r in (0, 1):
            set_time_axis(axes[r][j], c["t_inference"])
        axes[0][j].set_title(f"{SHORT[m]}\n(noise scale = {time_scale:g})", fontsize=14)
        axes[1][j].set_xlabel("Time since programming")
    finish_rows(axes)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    legend_and_save(fig, ser, out, note)


def binned(x, y, n_bins):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < n_bins * 2:
        return pd.DataFrame(columns=["x", "y"])
    d["b"] = pd.qcut(d.x, n_bins, duplicates="drop")
    return d.groupby("b", observed=True).agg(x=("x", "mean"), y=("y", "mean")).sort_values("x")


def fig_rmse(agg, ser, out, note):
    """Eje común entre dispositivos: cada punto es una celda (dispositivo, escala, t) del modelo principal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    any_pts = False
    for s in ser.order:
        d = agg[(agg.model_id == ser.main[s]) & (agg.mechanism == "all") & (agg.noise_model.isin(MARKERS))
                & (agg.scale > 0)]
        for m, g in d.groupby("noise_model"):
            for ax, y in zip(axes, ("mean", "ret_mean")):
                ax.scatter(g.rmse, g[y], color=COLORS[s], marker=MARKERS[m], s=36, alpha=0.65, linewidths=0)
        for ax, y in zip(axes, ("mean", "ret_mean")):
            tr = binned(d.rmse, d[y], 8)
            if len(tr):
                ax.plot(tr.x, tr.y, color=COLORS[s], lw=2.6)
        any_pts |= len(d) > 0
    if not any_pts:
        return print("Sin datos para la figura de RMSE.")
    for ax in axes:
        ax.set_xlabel(r"Effective weight RMSE (units of $w_{max}$)")
    axes[0].set_ylabel("Mean episode reward")
    axes[1].set_ylabel("Retained performance (% of noise-free)")
    axes[1].axhline(100, color="0.5", lw=1, ls=":")
    for ax in axes:
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    handles = [Line2D([], [], color=COLORS[s], lw=2.6, label=ser.label(s)) for s in ser.order]
    handles += [Line2D([], [], color="0.3", marker=MARKERS[m], ls="", ms=7, label=SHORT[m]) for m in ("pcm", "reram", "reramcmo")]
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.075, note + " Points: one per (device, noise scale, time); line: 8 quantile bins.",
             ha="center", va="top", fontsize=10, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def grid_matrix(agg, model_id, noise_model, scales, times, col):
    z = np.full((len(scales), len(times)), np.nan)
    d = agg[(agg.model_id == model_id) & (agg.noise_model == noise_model) & (agg.mechanism == "all")]
    for i, s in enumerate(scales):
        for k, t in enumerate(times):
            r = d[np.isclose(d.scale, s) & np.isclose(d.t_inference, t)]
            if len(r):
                z[i, k] = r[col].iloc[0]
    return z


def annotate(ax, z, signed, dark_fn):
    """Número en cada celda; texto blanco donde el fondo es oscuro (dark_fn(v) -> bool)."""
    for i in range(z.shape[0]):
        for k in range(z.shape[1]):
            v = z[i, k]
            if np.isnan(v):
                continue
            n = int(round(v))      # int(round()) evita los "-0" de redondear -0.3
            txt = f"{n:+d}" if (signed and n != 0) else f"{n:d}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=8.5, color="white" if dark_fn(v) else "0.15")


def fig_heatmap(agg, ser, m, out, note):
    """Fila 1: retención escala×tiempo de cada método. Fila 2: método − WANN (puntos porcentuales)."""
    if "wann" not in ser.main or len(ser.order) < 2:
        return print(f"Heatmap {m}: hacen falta WANN y al menos un SGRL.")
    d = agg[(agg.noise_model == m) & (agg.mechanism == "all")]
    scales = sorted(d.scale.unique())
    times = sorted(d.t_inference.unique())
    if len(scales) < 2 or len(times) < 2:
        return print(f"Heatmap {m}: grilla incompleta, se omite.")
    z = {s: grid_matrix(agg, ser.main[s], m, scales, times, "ret_mean") for s in ser.order}
    diffs = {s: z[s] - z["wann"] for s in ser.order if s != "wann"}
    allz = np.concatenate([v.ravel() for v in z.values()])
    vmax = max(100.0, np.nanmax(allz)) if np.isfinite(allz).any() else 100.0
    alld = np.concatenate([v.ravel() for v in diffs.values()])
    dmax = max(5.0, np.nanmax(np.abs(alld))) if np.isfinite(alld).any() else 5.0
    n = len(ser.order)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n + 1.8, 2 * (0.5 * len(scales) + 3.0)), squeeze=False)
    im_ret = im_diff = None
    for j, s in enumerate(ser.order):
        ax = axes[0][j]
        im_ret = ax.imshow(np.ma.masked_invalid(z[s]), origin="lower", aspect="auto", cmap=RET_CMAP, vmin=0, vmax=vmax)
        annotate(ax, z[s], False, lambda v: v > 0.55 * vmax)
        ax.set_title(ser.label(s).replace(", winner", "").replace(", best seed", "").replace(", best model", ""),
                     fontsize=12)
        ax2 = axes[1][j]
        if s == "wann":
            ax2.axis("off")
            continue
        im_diff = ax2.imshow(np.ma.masked_invalid(diffs[s]), origin="lower", aspect="auto", cmap=DIFF_CMAP,
                             vmin=-dmax, vmax=dmax)
        annotate(ax2, diffs[s], True, lambda v: abs(v) > 0.6 * dmax)
        ax2.set_title(f"SGRL {ser.hidden(s)}{' (conn.)' if s == 'sgrl_con' else ' (neur.)' if s == 'sgrl_neu' else ''} − WANN",
                      fontsize=12)
    for row in (0, 1):
        for j in range(n):
            ax = axes[row][j]
            if not ax.axison:
                continue
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels([TIME_TICK_LABELS.get(int(t), str(t)) for t in times], rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(len(scales)))
            ax.set_yticklabels([f"{v:g}" for v in scales] if j == 0 or row == 1 else [])
            ax.set_xlabel("Time since programming")
            ax.grid(False)
    axes[0][0].set_ylabel("Noise scale")
    axes[1][1].set_ylabel("Noise scale")
    fig.suptitle(f"{SHORT[m]}: retained performance over noise scale × time", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    if im_ret is not None:
        cax = fig.add_axes([0.94, 0.56, 0.015, 0.34])
        fig.colorbar(im_ret, cax=cax, label="Retained performance (%)")
    if im_diff is not None:
        cax = fig.add_axes([0.94, 0.10, 0.015, 0.34])
        fig.colorbar(im_diff, cax=cax, label="Method − WANN (pp); orange: method better")
    if m == "reramcmo":
        note += " t = 0 is a total noise bypass in this model."
    fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=10, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def fig_ablation(agg, ser, m, out, note):
    d = agg[(agg.noise_model == m) & agg.mechanism.isin(MECH_TITLE)]
    if d.empty:
        return print(f"Ablación {m}: sin datos.")
    t = float(d.t_inference.iloc[0])
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8), sharex=True, squeeze=False)
    for j, mech in enumerate(MECH_TITLE):
        for r, y in enumerate(("mean", "ret_mean")):
            ax = axes[r][j]
            # el punto s=0 de las tres curvas es el mismo: la celda 'all' con escala 0
            for s in ser.order:
                for mid in ser.others.get(s, []) + [ser.main[s]]:
                    c = pd.concat([curve(agg, mid, m, "all", "scale", t_inference=t, scale=0.0),
                                   curve(agg, mid, m, mech, "scale", t_inference=t)]).sort_values("scale")
                    if c.empty:
                        continue
                    is_main = ser.main[s] == mid
                    if is_main:
                        lo, hi = ("lo", "hi") if y == "mean" else ("ret_lo", "ret_hi")
                        ax.fill_between(c.scale, c[lo], c[hi], color=COLORS[s], alpha=0.14, linewidth=0, zorder=3)
                    ax.plot(c.scale, c[y], color=COLORS[s], lw=2.4 if is_main else 1.0, ls="-" if is_main else "--",
                            alpha=1 if is_main else 0.4, marker=METHODS[s]["marker"] if is_main else None, ms=6,
                            zorder=4 if is_main else 2)
        axes[0][j].set_title(MECH_TITLE[mech], fontsize=14)
        axes[1][j].set_xlabel("Noise scale of the active mechanism")
    finish_rows(axes)
    fig.suptitle(f"{SHORT[m]} ablation at t = {TIME_TICK_LABELS.get(int(t), t)}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    legend_and_save(fig, ser, out, note)


def fig_summary(agg, ser, out, note, scales=(1.0, 3.0)):
    """Barras: retención del modelo principal de cada método, por dispositivo, a escalas de ruido fijas."""
    devs = [m for m in ("pcm", "reram", "reramcmo")
            if not curve(agg, ser.main[ser.order[0]], m, "all", "scale", t_inference=INTENSITY_T[m]).empty]
    if not devs:
        return print("Sin datos para el resumen.")
    fig, axes = plt.subplots(1, len(scales), figsize=(6.2 * len(scales), 5.4), sharey=True, squeeze=False)
    w = 0.8 / len(ser.order)
    for k, sc in enumerate(scales):
        ax = axes[0][k]
        for i, s in enumerate(ser.order):
            ys, errs = [], []
            for m in devs:
                c = curve(agg, ser.main[s], m, "all", "scale", scale=sc, t_inference=INTENSITY_T[m])
                if c.empty or np.isnan(c.ret_mean.iloc[0]):
                    ys.append(np.nan); errs.append([0, 0])
                    continue
                r = c.iloc[0]
                ys.append(r.ret_mean); errs.append([r.ret_mean - r.ret_lo, r.ret_hi - r.ret_mean])
            x = np.arange(len(devs)) + (i - (len(ser.order) - 1) / 2) * w
            ax.bar(x, ys, w * 0.92, color=COLORS[s], yerr=np.array(errs).T, ecolor="0.25",
                   error_kw=dict(lw=1.2, capsize=3), label=ser.label(s), zorder=3)
        ax.set_xticks(range(len(devs)))
        ax.set_xticklabels([SHORT[m] for m in devs])
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.set_title(f"Noise scale {sc:g}", fontsize=14)
        ax.set_ylim(0, 112)
    axes[0][0].set_ylabel("Retained performance (% of noise-free)")
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.legend(handles=ser.handles()[:len(ser.order)], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    base = {s: agg[(agg.model_id == ser.main[s]) & (agg.noise_model == "none")]["mean"].iloc[0] for s in ser.order
            if len(agg[(agg.model_id == ser.main[s]) & (agg.noise_model == "none")])}
    fig.text(0.5, 0.005, note + " Noise-free mean reward: " +
             "; ".join(f"{'WANN' if s == 'wann' else 'SGRL ' + ser.hidden(s)} {v:.1f}" for s, v in base.items()) + ".",
             ha="center", va="bottom", fontsize=9.5, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("eval_results/car_wann_vs_sgrl"))
    ap.add_argument("--out-dir", default="graficos/Racing Car")
    ap.add_argument("--prefix", default="car_wann_vs_sgrl")
    ap.add_argument("--only", default="noise,time,rmse,heatmap,ablation,summary",
                    help="subconjunto de figuras, separado por comas")
    ap.add_argument("--time-scale", type=float, default=1.0, help="escala de ruido fija de la figura de tiempo")
    ap.add_argument("--boot", type=int, default=2000, help="remuestreos bootstrap por celda")
    ap.add_argument("--sources", nargs="+", choices=list(METHODS), default=None,
                    help="métodos a dibujar (def.: todos los que haya en models.csv)")
    ap.add_argument("--summary-scales", default="1,3", help="escalas de ruido de las barras de la figura summary")
    ap.add_argument("--env-label", default=None, help="def.: task_label de protocol.json")
    ap.add_argument("--floor", type=float, default=None, help="piso de la retención; def.: ret_floor de protocol.json")
    a = ap.parse_args()

    sweep, mfile = a.data_dir / "sweep_all.csv", a.data_dir / "models.csv"
    for p in (sweep, mfile):
        if not p.is_file():
            sys.exit(f"Error: no existe {p}. Corré run_noise_compare.py primero.")
    df = pd.read_csv(sweep)
    models = pd.read_csv(mfile)
    setup_publication_style()

    proto = {}
    if (a.data_dir / "protocol.json").is_file():
        proto = json.loads((a.data_dir / "protocol.json").read_text())
    label = a.env_label or proto.get("task_label", "CarTrack")
    floor = a.floor if a.floor is not None else float(proto.get("ret_floor", 0.0))
    agg = aggregate(df, a.boot, floor)
    agg.to_csv(a.data_dir / "aggregated.csv", index=False)
    print(f"Agregado: {a.data_dir / 'aggregated.csv'} ({len(agg)} celdas × modelo)")
    ser = Series(models, set(agg.model_id), a.sources)
    if not ser.main:
        sys.exit("Error: models.csv no marca ningún modelo principal.")
    if len(ser.order) < len(set(models.source) & set(METHODS)) and not a.sources:
        print("  (aviso: hay métodos en models.csv sin resultados completos; se omiten)")
    for s in ser.order:
        mid = ser.main[s]
        b = agg[(agg.model_id == mid) & (agg.noise_model == "none")]
        print(f"  {s}: modelo principal {mid}, retorno sin ruido = {b['mean'].iloc[0]:.1f}" if len(b)
              else f"  {s}: modelo principal {mid} SIN línea base sin ruido")

    n_chips = int(agg.n_chips.max())
    note_ret = ("Retention is relative to each model's own noise-free return" +
                (f" (floor {floor:g} = failure)." if floor else "."))
    note = f"{label}. Bands: 95% bootstrap CI over chips (n≈{n_chips}). " + note_ret
    out = Path(a.out_dir)
    which = set(a.only.split(","))
    present = set(agg.noise_model.unique()) - {"none"}
    if "noise" in which:
        fig_intensity(agg, ser, present, out / f"{a.prefix}_noise_robustness.png", note)
    if "time" in which:
        fig_time(agg, ser, out / f"{a.prefix}_time_robustness.png", note, a.time_scale)
    if "rmse" in which:
        fig_rmse(agg, ser, out / f"{a.prefix}_rmse_robustness.png", f"{label}. " + note_ret)
    if "heatmap" in which:
        for m in ("pcm", "reram", "reramcmo"):
            if m in present:
                fig_heatmap(agg, ser, m, out / f"{a.prefix}_heatmap_{m}.png", f"{label}. " + note_ret)
    if "ablation" in which:
        for m in ("pcm", "reramcmo"):
            if m in present:
                fig_ablation(agg, ser, m, out / f"{a.prefix}_ablation_{m}.png", note)
    if "summary" in which:
        fig_summary(agg, ser, out / f"{a.prefix}_summary.png", f"{label}. " + note_ret,
                    tuple(float(x) for x in a.summary_scales.split(",")))


if __name__ == "__main__":
    main()

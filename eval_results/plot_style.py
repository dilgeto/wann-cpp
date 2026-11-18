#!/usr/bin/env python3
"""
Estilo compartido para los gráficos de robustez a ruido de memristores
(plot_car_noise*.py) — pensado para quedar listo para publicación: fuentes
más grandes, dpi alto, bbox ajustado, y una paleta de color segura para
daltonismo (Okabe & Ito, 2008; la misma referencia estándar que usan la
mayoría de las guías de gráficos científicos). No corre nada ni lee datos —
solo constantes y una función que ajusta `plt.rcParams`.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

# "simple" queda gris a propósito: es el placeholder sin calibrar, no un
# modelo de dispositivo real — el color lo marca como referencia, no como par
# de los otros tres.
MODEL_COLORS = {
    "simple":   "#999999",
    "pcm":      "#0072B2",
    "reram":    "#D55E00",
    "reramcmo": "#009E73",
}
MODEL_LABELS = {
    "simple":   "Simple (uncalibrated placeholder)",
    "pcm":      "PCM (Nandakumar et al., 2019)",
    "reram":    "ReRAM (Wan et al., 2022)",
    "reramcmo": "ReRAM CMO/HfOx (Falcone et al.)",
}

# Segundos -> etiqueta legible en inglés, para los ejes de tiempo.
TIME_TICK_LABELS = {
    0: "0", 1: "1 s", 60: "1 min", 300: "5 min", 3600: "1 h", 21600: "6 h",
    86400: "1 day", 172800: "2 days", 604800: "1 week", 2592000: "30 days",
    31536000: "1 year",
}


def setup_publication_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Journal ("Neural Networks", Elsevier): vector PDF/EPS with embedded fonts; widths 90 mm (single column) / 190 mm (full page).
JOURNAL_FULL_W = 7.48    # in = 190 mm
JOURNAL_COL_W = 3.54     # in = 90 mm


def setup_journal_style() -> None:
    """Style for figures drawn AT the final journal size (no later down-scaling): sans-serif (Liberation Sans, Arial-metric),
    text >= 7 pt, TrueType fonts embedded in PDF/PS (fonttype 42), 600 dpi raster previews. Use with figsize width JOURNAL_*_W."""
    setup_publication_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
        "savefig.dpi": 600, "savefig.pad_inches": 0.03,
    })


_TASK_PREFIX = re.compile(r"^(car|disc_mc|acrobot)_")


def figure_tag(prefix: str) -> str:
    """Tag de un prefijo de figura: el prefijo sin el nombre de la tarea (car_replicas_fs -> replicas_fs)."""
    return _TASK_PREFIX.sub("", prefix)


def tagged_dir(out_dir, prefix: str, flat: bool = False) -> Path:
    """<out_dir>/ruido/<tag>/ para las figuras de ruido de memristores (graficos/<Tarea>/ruido/<tag>/); con
    flat=True devuelve out_dir tal cual (organización antigua, todo en una carpeta)."""
    out = Path(out_dir)
    return out if flat else out / "ruido" / figure_tag(prefix)

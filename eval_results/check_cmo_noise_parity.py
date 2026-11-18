"""Prueba de paridad del lado SG: nuestro sgrl/noise.py (mapping single/pair) vs. una REFERENCIA ESCRITA A MANO con las fórmulas
de aihwkit (inference/noise/reram.py + converter/conductance.py, commit 86681d9, leídas el 2026-11-18). No ejecuta código de aihwkit:
si querés la prueba contra el aihwkit real, instalalo y comparalo con este mismo muestreo.
Muestrea N veces (programación + lectura) y compara media y desviación por peso. Esperado: |Δmedia| < ~4 errores estándar y
cociente de desviaciones ~1 (0.96-1.07). Con los coeficientes del 2 % (el error que había) da 22 σ.
Uso:  /ruta/al/venv/python eval_results/check_cmo_noise_parity.py ../surrogate-gradient-rl   (con apply_sgrl_cmo_mapping.py aplicado)"""
import math, sys, torch
sys.path.insert(0, sys.argv[1])
from sgrl.noise import Crossbar, ReRamCMOModel
GMAX, GMIN = 88.199997, 9.0
torch.manual_seed(1)

def ref_noise(weights, t, mapping, gen):
    """aihwkit: convert_to_conductances -> programming (coef. 0.2 % por la clave por defecto) -> drift/read -> clamp -> back."""
    c = [0.00081107, 0.00106879]                       # coeff_dict[min(keys)] = clave 0.2
    def prog(g):
        sig = c[0] + g * c[1]                           # _apply_poly (g_max/ref = 1), SIN recorte
        return g + sig * torch.randn(g.shape, generator=gen, dtype=g.dtype)
    def drift(g):
        if t == 0: return g
        lt = math.log(t)
        gm = g + (-0.08900206 * lt)
        sr = 0.04201137 * lt + 0.41183342
        gd = gm + torch.randn(g.shape, generator=gen, dtype=g.dtype) * sr
        sd = 0.0277316483 * torch.log10(gd) * math.sqrt(math.log((t + 1e-6) / (2e-6)))
        return (gd + sd * torch.randn(g.shape, generator=gen, dtype=g.dtype)).clamp(GMIN, GMAX)
    if mapping == "single":
        lo, hi = weights.min(), weights.max()
        sc = (GMAX - GMIN) / (hi - lo)
        g = (weights - lo) * sc + GMIN
        return lo + (drift(prog(g)) - GMIN) / sc
    am = weights.abs().max(); sc = (GMAX - GMIN) / am   # pair (SinglePairConductanceConverter)
    gp = (weights * sc).clamp(0, GMAX) + GMIN; gn = (-weights * sc).clamp(0, GMAX) + GMIN
    return ((drift(prog(gp)) - GMIN) - (drift(prog(gn)) - GMIN)) / sc

def ours(weights, t, mapping, gen):
    m = ReRamCMOModel(g_max=GMAX, g_min=GMIN, mapping=mapping)
    x = Crossbar(weights, float(weights.abs().max()), m); x.program(gen); return x.read(t, gen)

W = torch.cat([torch.randn(1, 40, dtype=torch.float64) * 0.4, torch.tensor([[1.0, -1.0, 0.0, 0.02]], dtype=torch.float64)], 1)
N = 6000; worst = 0.0
for mapping in ("single", "pair"):
    for t in (1.0, 3600.0, 31536000.0):
        g1, g2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(12)
        R = torch.stack([ref_noise(W, t, mapping, g1)[0] for _ in range(N)])
        O = torch.stack([ours(W, t, mapping, g2)[0] for _ in range(N)])
        dm = ((R.mean(0) - O.mean(0)).abs() / (R.std(0) / N ** 0.5 * 2 ** 0.5)).max().item()     # |Δmedia| en errores estándar
        rs = (R.std(0) / O.std(0)); 
        print(f"{mapping:6s} t={t:>10g}:  max|Δmedia|={dm:4.1f} σ_est   cociente de desv. min/max = {rs.min():.3f}/{rs.max():.3f}")
        worst = max(worst, dm)
print("PEOR |Δmedia| (en errores estándar):", round(worst, 1), "(~<4 = compatible con muestreo)")

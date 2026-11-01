"""QF2_BRIER: CE + Brier por trozos (propuesta f2-ce-brier, astra ronda 37 sección 3; banco CORE ronda 4).

Con z = cap * tanh(x @ w[:vocab].T / cap) (los logits tras el softcap, los de ce_trozos) y p = softmax(z), cada token
válido (objetivo != -1) aporta
    -log p_y + lam * (S2 - 2 p_y + 1),  S2 = sum_v p_v^2,
y la pérdida del micro-lote es la media sobre los válidos, como la CE del kit; los -1 no aportan CE ni Brier.
El gradiente respecto al logit tras el softcap es
    dL/dz_j = p_j * (1 + 2 lam (p_j - S2 + p_y)) - [j = y] * (1 + 2 lam p_y)
por token válido y entre el número de válidos, multiplicado por la derivada del softcap (1 - tanh^2) como en ce_trozos.
Forward y backward recorren trozos fijos de filas sin materializar one-hot ni las probabilidades de toda la secuencia:
el forward añade a la CE por trozos p = exp(z - lse), S2 y p_y por fila (guarda lse, S2 y p_y, tres vectores [filas]);
el backward recalcula los logits del trozo como ce_trozos, reutiliza p = exp(z - lse) y añade el factor por fila y el
scatter del objetivo. Formas fijas y bucle estático, sin ramas que dependan de los datos.

Con lam = 0 la pérdida y los gradientes son los de ce_trozos bit a bit. El mando vacío apaga la rama (train.py usa
ce_trozos); '0' entra por esta función con lam = 0; cualquier otro valor ha de ser un número finito >= 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ce_trozos import _trozo


class _CETrozosBrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo, lam):
        vale = y != -1
        total = x.new_zeros((), dtype=torch.float32)
        lses, s2s, pys = [], [], []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            zt = zc.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            p = torch.exp(zc - lse.unsqueeze(1))
            s2 = (p * p).sum(dim=-1)
            py = p.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            total = total + torch.where(vale[i:i + trozo], lse - zt + lam * (s2 - 2.0 * py + 1.0), 0.0).sum()
            lses.append(lse)
            s2s.append(s2)
            pys.append(py)
        ctx.save_for_backward(x, w, y, torch.cat(lses), torch.cat(s2s), torch.cat(pys))
        ctx.cap, ctx.vocab, ctx.trozo, ctx.lam = cap, vocab, trozo, lam
        return total / vale.sum()

    @staticmethod
    def backward(ctx, g):
        x, w, y, lse, s2, py = ctx.saved_tensors
        cap, vocab, trozo, lam = ctx.cap, ctx.vocab, ctx.trozo, ctx.lam
        vale = y != -1
        escala = g.float() / vale.sum()
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc, vc = x[i:i + trozo], y[i:i + trozo], vale[i:i + trozo]
            pyc = py[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d = d * (1.0 + 2.0 * lam * (d + (pyc - s2[i:i + trozo]).unsqueeze(1)))
            d.scatter_add_(1, yc.clamp_min(0).unsqueeze(1), -(1.0 + 2.0 * lam * pyc).unsqueeze(1))
            d = d * (escala * vc.float()).unsqueeze(1) * (1.0 - t * t)
            dz = d.to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None, None


def ce_brier_trozos(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int, lam: float) -> torch.Tensor:
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozosBrier.apply(x, w, targets, cap, vocab, trozo, float(lam))


@dataclass(frozen=True)
class Brier:
    lam: float

    def descripcion(self) -> str:
        return f"qf2_brier: ce + {self.lam} * brier por trozos (sum p^2 - 2 p_y + 1 por token válido, media sobre válidos)"


def leer(texto: str) -> Brier | None:
    texto = texto.strip()
    if not texto:
        return None
    try:
        lam = float(texto)
    except ValueError:
        raise ValueError(f"QF2_BRIER={texto!r} no es un número") from None
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"QF2_BRIER={texto!r} ha de ser un número finito >= 0")
    return Brier(lam)

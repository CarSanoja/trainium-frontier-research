"""Pérdida de la unión del banco CORE (paquete 302): QF2_CE_DIFICILES, QF2_CE_TRAMO, QF2_EQUILIBRIO y QF2_BRIER a la vez.

leer() devuelve None con menos de dos de esos mandos encendidos: entonces train.py va por la rama de su paquete de origen
(qf2_brier.ce_brier_trozos del 298, OBJETIVO del 295, EQUILIBRIO del 300 o ce_trozos), bit a bit. Con dos o más, Union
recibe los estados que entran en la CE por trozos, los objetivos [B, T] y es_bos = (idx == BOS) y devuelve

    w        = w_dificiles * w_equilibrio en los válidos (1 el que esté apagado), renormalizado: w * V / sum(w)
    pérdida  = (1 - f) * sum(w * nll) / V + f * tramo(nll) + lam * sum_válidos(brier) / V

con V los objetivos válidos del micro-lote, nll y brier por token de la CE por trozos (0 en los -1), w_dificiles de
qf2_objetivos.pesos_dificiles y w_equilibrio de qf2_equilibrio (cada uno ya con media 1 sobre los válidos; el producto se
vuelve a normalizar), tramo(nll) el segundo término de qf2_objetivos.perdida_tramo sobre la NLL sin ponderar (f = 0 sin
QF2_CE_TRAMO) y el Brier sin ponderar por w (lam = 0 sin QF2_BRIER). Los pesos no llevan gradiente. Con un solo mando la
fórmula es la de su paquete de origen; la rama de ese paquete la calcula bit a bit.

nll y brier salen juntos de _CETrozosBrierTokens (trozos fijos de filas como ce_trozos, sin one-hot ni probabilidades de
toda la secuencia): con a = d pérdida / d nll y b = d pérdida / d brier por token, dL/dz_j = p_j (a + 2 b (p_j - S2 + p_y))
- [j = y] (a + 2 b p_y) por la derivada del softcap; con a = 1 / V y b = lam / V es el gradiente de qf2_brier. Sin
QF2_BRIER la NLL sale de ce_trozos.ce_trozos_tokens.

Enganche para el 301 (QF2_MAESTRO, sin fusionar): la pérdida mezclada sería (1 - alfa) * (lo que devuelve Union, o la rama
de un solo mando) + alfa * KL(maestro || alumno). Enganche para el 297 (QF2_MARGEN, sin fusionar): su término de margen se
sumaría a la pérdida como otro término aditivo, sobre sus filas, que los sustituidores de qf2_filas evitan.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import qf2_objetivos
from ce_trozos import _trozo, ce_trozos_tokens

MINIMO = 1e-30


class _CETrozosBrierTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo):
        vale = y != -1
        nlls, briers, lses, s2s, pys = [], [], [], [], []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            yc, vc = y[i:i + trozo].clamp_min(0).unsqueeze(1), vale[i:i + trozo]
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            p = torch.exp(zc - lse.unsqueeze(1))
            s2 = (p * p).sum(dim=-1)
            py = p.gather(1, yc).squeeze(1)
            nlls.append(torch.where(vc, lse - zc.gather(1, yc).squeeze(1), 0.0))
            briers.append(torch.where(vc, s2 - 2.0 * py + 1.0, 0.0))
            lses.append(lse)
            s2s.append(s2)
            pys.append(py)
        ctx.save_for_backward(x, w, y, torch.cat(lses), torch.cat(s2s), torch.cat(pys))
        ctx.cap, ctx.vocab, ctx.trozo = cap, vocab, trozo
        return torch.cat(nlls), torch.cat(briers)

    @staticmethod
    def backward(ctx, g_nll, g_brier):
        x, w, y, lse, s2, py = ctx.saved_tensors
        cap, vocab, trozo = ctx.cap, ctx.vocab, ctx.trozo
        vale = (y != -1).float()
        a, b = g_nll.float() * vale, g_brier.float() * vale
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc = x[i:i + trozo], y[i:i + trozo].clamp_min(0).unsqueeze(1)
            ac, bc, pyc = a[i:i + trozo], b[i:i + trozo], py[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d = d * (ac.unsqueeze(1) + 2.0 * bc.unsqueeze(1) * (d + (pyc - s2[i:i + trozo]).unsqueeze(1)))
            d.scatter_add_(1, yc, -(ac + 2.0 * bc * pyc).unsqueeze(1))
            dz = (d * (1.0 - t * t)).to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None


def ce_brier_tokens(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int):
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozosBrierTokens.apply(x, w, targets, cap, vocab, trozo)


@dataclass(frozen=True)
class Union:
    dificiles: object = None
    tramo: object = None
    equilibrio: object = None
    brier: object = None

    def pesos(self, nll: torch.Tensor, vale: torch.Tensor, es_bos: torch.Tensor) -> torch.Tensor:
        w = vale.to(torch.float32)
        if self.dificiles is not None:
            w = w * qf2_objetivos.pesos_dificiles(nll, vale, self.dificiles.factor)
        if self.equilibrio is not None:
            w = w * self.equilibrio.pesos(vale, es_bos)
        return w * (vale.sum() / w.sum().clamp_min(MINIMO))

    def combinar(self, nll: torch.Tensor, brier: torch.Tensor | None, vale: torch.Tensor, es_bos: torch.Tensor) -> torch.Tensor:
        validos = vale.sum()
        perdida = (self.pesos(nll, vale, es_bos) * nll).sum() / validos
        if self.tramo is not None:
            tramo = qf2_objetivos.perdida_tramo(nll, vale, 1.0, self.tramo.tau, self.tramo.k)
            perdida = (1.0 - self.tramo.f) * perdida + self.tramo.f * tramo
        if self.brier is not None:
            perdida = perdida + self.brier.lam * brier.sum() / validos
        return perdida

    def __call__(self, x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, es_bos: torch.Tensor, cap: float, vocab: int,
                 trozo: int) -> torch.Tensor:
        if self.brier is None:
            nll, brier = ce_trozos_tokens(x, w, targets.reshape(-1), cap, vocab, trozo), None
        else:
            nll, brier = ce_brier_tokens(x, w, targets.reshape(-1), cap, vocab, trozo)
            brier = brier.view_as(targets)
        return self.combinar(nll.view_as(targets), brier, targets != -1, es_bos)

    def partes(self) -> list:
        return [p for p in (self.dificiles, self.tramo, self.equilibrio, self.brier) if p is not None]

    def descripcion(self, ruido=None) -> str:
        texto = "; ".join(p.descripcion() for p in self.partes())
        extra = f"; {ruido.descripcion()}" if ruido is not None else ""
        return (f"qf2_union: pérdida = (1 - f) * sum(w * nll) / V + f * tramo(nll) + lam * sum(brier) / V, "
                f"w = w_dificiles * w_equilibrio renormalizado a media 1; {texto}{extra}")


def leer(dificiles, tramo, equilibrio, brier) -> Union | None:
    union = Union(dificiles, tramo, equilibrio, brier)
    return union if len(union.partes()) >= 2 else None

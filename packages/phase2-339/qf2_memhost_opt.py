"""QF2_MEMHOST: optimiser for the host tables, applied only to the rows touched in the step.

rms (default): one fp32 second moment per row, bias corrected by the number of touches of that row, and a decay per
touch scaled by the lr multiplier of the schedule: acc = b*acc + (1-b)*mean(g^2); ahat = acc / (1 - b^n);
w = w*(1-decay*lrm) - lr*g/(sqrt(ahat)+eps). The first touch moves every row by an RMS of exactly lr: eps sits far below
the real per-row gradient (~1.9e-10 at n_embd 1536 over 64 ranks).
adam: lazy Adam per element with the exact arithmetic of torch.optim.SparseAdam (the control arm, three times the
master size in RAM).
"""
from __future__ import annotations

import math
import os

import torch

import qf2_memhost_tabla as tab

BETA = float(os.environ.get("QF2_MEMHOST_BETA", "0.99"))
DECAY = float(os.environ.get("QF2_MEMHOST_DECAY", "1e-3"))
EPS = float(os.environ.get("QF2_MEMHOST_EPS", "1e-24"))


@torch.no_grad()
def paso_rms(t: tab.Tabla, ids: torch.Tensor, g: torch.Tensor, lr: float, beta: float = BETA, decay: float | None = None, eps: float = EPS,
             lrm: float = 1.0) -> None:
    if ids.numel() == 0:
        return
    decay = DECAY if decay is None else decay
    w = t.q.index_select(0, ids)
    acc = t.acc.index_select(0, ids)
    n = t.cnt.index_select(0, ids).float() + 1.0
    acc = beta * acc + (1.0 - beta) * g.pow(2).mean(dim=-1)
    ahat = acc / (1.0 - beta ** n)
    w = w * (1.0 - decay * lrm) - lr * g / (ahat.sqrt() + eps).unsqueeze(-1)
    t.q.index_copy_(0, ids, w)
    t.acc.index_copy_(0, ids, acc)
    t.cnt.index_copy_(0, ids, n.clamp_(max=tab.CNT_TOPE).to(torch.int16))


@torch.no_grad()
def paso_adam(t: tab.Tabla, ids: torch.Tensor, g: torch.Tensor, lr: float, paso: int, b1: float = 0.8, b2: float = 0.995, eps: float = 1e-10) -> None:
    if ids.numel() == 0:
        return
    m = t.m.index_select(0, ids)
    v = t.v.index_select(0, ids)
    m = m + (g - m) * (1.0 - b1)
    v = v + (g.pow(2) - v) * (1.0 - b2)
    denom = v.sqrt() + eps
    paso_tam = lr * math.sqrt(1.0 - b2 ** paso) / (1.0 - b1 ** paso)
    t.q.index_copy_(0, ids, t.q.index_select(0, ids) - paso_tam * (m / denom))
    t.m.index_copy_(0, ids, m)
    t.v.index_copy_(0, ids, v)


def aplicar(t: tab.Tabla, modo: str, ids: torch.Tensor, g: torch.Tensor, lr: float, paso: int, lrm: float = 1.0) -> None:
    if modo == "adam":
        paso_adam(t, ids, g, lr, paso)
    else:
        paso_rms(t, ids, g, lr, lrm=lrm)

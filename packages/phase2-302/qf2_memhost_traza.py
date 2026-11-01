"""QF2_MEMHOST_TRAZA: read-only trace of the host-table optimiser, to tell a badly scaled update from a lack of signal.

Every N optimiser steps (N = TRAZA, 0 = off: nothing is read, nothing is printed) the rank that owns the central row of
a hash head takes up to MAX_FILAS touched rows of that head before the update and the same rows after it, and prints one
line per head (tagged rango=R, so every head is observed on the rank that holds its middle rows at any world size): percentiles 10/50/90 of
|dw|/|w| over the rows with a non-zero w, the fraction of sampled rows still at zero, the median share of the decay term
in the update, and the mean gradient norm per bucket of touches (1, 2-3, 4-15, 16-63, 64+). Only copies are made: no value
of the table or of the optimiser state is written. train.py sets TRAZA from QF2_MEMHOST_TRAZA.
"""
from __future__ import annotations

import time

import torch

TRAZA = 0
MAX_FILAS = 4096
CUBOS = ("1", "2_3", "4_15", "16_63", "64mas")
BORDES = (2, 4, 16, 64)


def activa(paso: int, rango: int, cfg, mundo: int) -> tuple:
    if TRAZA <= 0 or paso % TRAZA:
        return ()
    ro = -(-cfg.filas // mundo)
    return tuple(k for k, (b, (_o, f)) in enumerate(zip(cfg.bases, cfg.cabezas)) if (b + f // 2) // ro == rango)


def _elegir(n: int) -> torch.Tensor:
    if n <= MAX_FILAS:
        return torch.arange(n)
    return torch.linspace(0, n - 1, MAX_FILAS).long()


@torch.no_grad()
def previa(t, ids: torch.Tensor, g: torch.Tensor, cfg, cabs: tuple) -> list:
    t0 = time.perf_counter()
    cab = torch.bucketize(ids, torch.tensor(cfg.bases[1:], dtype=ids.dtype), right=True)
    out = []
    for k in cabs:
        pos = (cab == k).nonzero().reshape(-1)
        if pos.numel() == 0:
            continue
        pos = pos[_elegir(pos.numel())]
        sel = ids.index_select(0, pos)
        out.append((k, sel, t.q.index_select(0, sel), t.cnt.index_select(0, sel).float() + 1.0, g.index_select(0, pos).float().norm(dim=-1)))
    return [out, time.perf_counter() - t0]


def _cuantiles(x: torch.Tensor) -> tuple:
    if x.numel() == 0:
        return 0.0, 0.0, 0.0
    return tuple(float(v) for v in torch.quantile(x, torch.tensor([0.1, 0.5, 0.9])))


@torch.no_grad()
def lineas(est: list, t, paso: int, lrm: float, decay: float, rango: int) -> list:
    t0 = time.perf_counter()
    out = []
    for k, sel, w0, n, gn in est[0]:
        dw = (t.q.index_select(0, sel) - w0).norm(dim=-1)
        nw = w0.norm(dim=-1)
        nz = nw > 0
        p10, p50, p90 = _cuantiles(dw[nz] / nw[nz])
        mov = dw > 0
        _a, dec, _b = _cuantiles(decay * lrm * nw[mov] / dw[mov])
        cubo = torch.bucketize(n, torch.tensor(BORDES, dtype=n.dtype), right=True)
        gcub = " ".join(f"gn_{c}={float(gn[cubo == i].mean()) if (cubo == i).any() else 0.0:.3e} c_{c}={int((cubo == i).sum())}" for i, c in enumerate(CUBOS))
        ms = 1000.0 * (est[1] + time.perf_counter() - t0)
        out.append(f"qf2_memhost_traza paso={paso} rango={rango} cabeza={k} filas={sel.numel()} cero={float((~nz).float().mean()):.4f} "
                   f"dw_p10={p10:.3e} dw_p50={p50:.3e} dw_p90={p90:.3e} decay_parte={dec:.3e} {gcub} ms={ms:.2f}")
    return out

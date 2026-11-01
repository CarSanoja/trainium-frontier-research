"""QF2_NGRAM_FILAS: causal hashed bigram and trigram memory tables added to the input embedding.

Two tables of FILAS rows (the same for every rank, like any embedding) are indexed by a deterministic hash of the last two
(bigram) or three (trigram) tokens ending at the current one. The hash is computed in int64 inside the forward from the
input ids only: it never reads the next token, the target, or other rows of the batch. Positions without enough history
since the start of the row or the most recent BOS token map to a fixed sentinel row; the (BOS, t) bigram is kept. Both tables are normalised, summed and scaled by one learned scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MULT_BI = (10007,)
MULT_TRI = (10009, 12979)
ESCALA_INICIAL = 0.1
MINIMO = 64


def primo_bajo(filas: int) -> int:
    n = filas - 3
    while n > 2 and any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n -= 1
    return n


def comprobar(filas: int) -> None:
    if filas < MINIMO:
        raise ValueError(f"QF2_NGRAM_FILAS={filas} es menor que {MINIMO}")


def desplazar(idx: torch.Tensor, k: int) -> torch.Tensor:
    return F.pad(idx, (k, 0))[:, :idx.size(1)]


def indices(idx: torch.Tensor, filas: int, bos: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = primo_bajo(filas)
    i0 = idx.to(torch.int64)
    i1, i2 = desplazar(i0, 1), desplazar(i0, 2)
    pos = torch.arange(i0.size(1), device=i0.device).unsqueeze(0)
    h2 = (i1 * MULT_BI[0] + i0) % p
    h3 = (((i2 * MULT_TRI[0] + i1) % p) * MULT_TRI[1] + i0) % p
    vale2 = (i0 != bos) & (pos >= 1)
    vale3 = vale2 & (i1 != bos) & (pos >= 2)
    centinela = torch.full_like(h2, p)
    return torch.where(vale2, h2, centinela), torch.where(vale3, h3, centinela)


def tablas(filas: int, n_embd: int) -> nn.ModuleDict:
    comprobar(filas)
    with torch.random.fork_rng(devices=[]):
        return nn.ModuleDict({"2": nn.Embedding(filas, n_embd), "3": nn.Embedding(filas, n_embd)})


def escala() -> nn.Parameter:
    return nn.Parameter(torch.full((1,), ESCALA_INICIAL))


@torch.no_grad()
def iniciar(mods: nn.ModuleDict, n_embd: int) -> None:
    for mod in mods.values():
        torch.nn.init.normal_(mod.weight, mean=0.0, std=0.8)


def sumar(mods: nn.ModuleDict, esc: torch.Tensor, idx: torch.Tensor, filas: int, bos: int, tabla, x: torch.Tensor) -> torch.Tensor:
    h2, h3 = indices(idx, filas, bos)
    m = F.rms_norm(tabla(mods["2"], h2).to(x.dtype), (x.size(-1),)) + F.rms_norm(tabla(mods["3"], h3).to(x.dtype), (x.size(-1),))
    return x + esc.to(x.dtype) * m

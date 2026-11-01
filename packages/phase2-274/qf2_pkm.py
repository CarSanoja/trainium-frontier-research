"""QF2_PKM_FILAS: product-key memory (PKM) residual branch in one central block.

After block n_layer // 2 the normalised residual stream of each position is projected to two query halves of dk dims. Each half is
scored against its own bank of sqrt(FILAS) sub-keys, the top k of each half are combined into k * k candidate rows (row = i * sqrt(FILAS) + j),
the best k of those are kept and weighted by a softmax over their scores. The weighted sum of the selected value rows (a FILAS x VALOR
table read by index) is projected back to n_embd and added to the residual stream through one learned scalar. Everything is computed
per position from that position's state only: no mixing across positions or batch rows. The module is built with a forked RNG and
initialised last, so with FILAS = 0 the model and its initialisation stream are untouched.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ESCALA_INICIAL = 0.1
VALOR_STD = 0.5


def raiz_exacta(filas: int) -> int:
    r = math.isqrt(filas)
    if r * r != filas:
        raise ValueError(f"QF2_PKM_FILAS={filas} debe ser un cuadrado perfecto")
    return r


def comprobar(filas: int, k: int, valor: int, dk: int) -> None:
    r = raiz_exacta(filas)
    if not 1 <= k <= r:
        raise ValueError(f"QF2_PKM_K={k} debe estar entre 1 y sqrt(filas)={r}")
    if valor < 1 or dk < 1:
        raise ValueError(f"QF2_PKM_VALOR={valor} y QF2_PKM_DK={dk} deben ser positivos")


def seleccionar(s1: torch.Tensor, s2: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    r = s1.size(-1)
    v1, i1 = s1.topk(k, dim=-1)
    v2, i2 = s2.topk(k, dim=-1)
    comb = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
    puntos, pos = comb.topk(k, dim=-1)
    a = torch.div(pos, k, rounding_mode="floor")
    b = pos - a * k
    filas = i1.gather(-1, a) * r + i2.gather(-1, b)
    return puntos, filas


class PKM(nn.Module):
    def __init__(self, filas: int, n_embd: int, k: int, valor: int, dk: int):
        super().__init__()
        comprobar(filas, k, valor, dk)
        self.filas, self.k, self.dk, self.r = filas, k, dk, math.isqrt(filas)
        with torch.random.fork_rng(devices=[]):
            self.valores = nn.Embedding(filas, valor)
        self.consulta = nn.Parameter(torch.empty(2 * dk, n_embd))
        self.subclaves = nn.Parameter(torch.empty(2 * self.r, dk))
        self.salida = nn.Parameter(torch.empty(n_embd, valor))
        self.escala = nn.Parameter(torch.full((1,), ESCALA_INICIAL))

    @torch.no_grad()
    def iniciar(self) -> None:
        n_embd = self.consulta.size(1)
        s = math.sqrt(3.0) * n_embd ** -0.5
        nn.init.uniform_(self.consulta, -s, s)
        nn.init.normal_(self.subclaves, mean=0.0, std=self.dk ** -0.5)
        nn.init.normal_(self.valores.weight, mean=0.0, std=VALOR_STD)
        nn.init.uniform_(self.salida, -s, s)

    def puntuaciones(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.linear(x, self.consulta.to(x.dtype))
        claves = self.subclaves.to(x.dtype)
        ancho = torch.promote_types(x.dtype, torch.float32)
        s1 = F.linear(q[..., :self.dk], claves[:self.r]).to(ancho)
        s2 = F.linear(q[..., self.dk:], claves[self.r:]).to(ancho)
        return s1, s2

    def forward(self, x: torch.Tensor, tabla) -> torch.Tensor:
        s1, s2 = self.puntuaciones(x)
        puntos, filas = seleccionar(s1, s2, self.k)
        peso = F.softmax(puntos, dim=-1).to(x.dtype)
        v = tabla(self.valores, filas).to(x.dtype)
        mezcla = (v * peso.unsqueeze(-1)).sum(dim=-2)
        return self.escala.to(x.dtype) * F.linear(mezcla, self.salida.to(x.dtype))

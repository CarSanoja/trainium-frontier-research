"""QF2_DIFF_CAPAS / QF2_DIFF_SIGNO: second-query attention branch in a few layers (DIFF V2 style, adapted to the kit).

In each marked layer a second query Q2 (own projection c_q2, same rotary and qk-norm as Q1) attends over the same K and V, and
the layer output is  SDPA(Q1,K,V) + signo * sigmoid(lam(x)) * SDPA(Q2,K,V)  with signo = -1 (difference) or +1 (sum). lam is a
bias-free Linear n_embd -> n_head on the normalised input of the attention, so the gate is per head and per position and equals
0.5 at initialisation (zero weights). Everything is per position: causality and batch independence are those of SDPA. The
modules are built under a forked RNG and initialised last, so with the flags off the model and its initialisation stream are
untouched. Indices at or beyond n_layer are ignored, so a flag baked for d22 still builds in the d4 CPU validations. Rule for six layers: espaciadas(n_layer) = 1 + 4k for d22 and d24 (1,5,9,13,17,21), written colon-separated in the flag.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNOS = (-1, 1)
SEPARADOR = re.compile(r"[\s,:;+.\-]+")


def indices(capas: str, n_layer: int) -> tuple[int, ...]:
    trozos = [t for t in SEPARADOR.split(capas.strip()) if t]
    if not trozos:
        return ()
    if not all(t.isdigit() for t in trozos):
        raise ValueError(f"QF2_DIFF_CAPAS={capas!r} debe ser una lista de indices de capa")
    lista = tuple(int(t) for t in trozos)
    if len(set(lista)) != len(lista):
        raise ValueError(f"QF2_DIFF_CAPAS={capas!r} repite capas")
    return tuple(sorted(i for i in lista if i < n_layer))


def espaciadas(n_layer: int, cuantas: int = 6) -> tuple[int, ...]:
    paso = (n_layer - 2) // (cuantas - 1)
    return tuple(1 + paso * k for k in range(cuantas))


def comprobar_signo(signo: int) -> None:
    if signo not in SIGNOS:
        raise ValueError(f"QF2_DIFF_SIGNO={signo} debe ser -1 (diferencia) o 1 (suma)")


class Lineal(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class RamaDiff(nn.Module):
    def __init__(self, n_embd: int, n_head: int, signo: int):
        super().__init__()
        comprobar_signo(signo)
        self.signo = float(signo)
        with torch.random.fork_rng(devices=[]):
            self.c_q2 = Lineal(n_embd, n_embd, bias=False)
            self.lam = Lineal(n_embd, n_head, bias=False)
        nn.init.zeros_(self.lam.weight)

    @torch.no_grad()
    def iniciar(self) -> None:
        s = math.sqrt(3.0) * self.c_q2.weight.size(1) ** -0.5
        nn.init.uniform_(self.c_q2.weight, -s, s)
        nn.init.zeros_(self.lam.weight)

    def puerta(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lam(x)).transpose(1, 2).unsqueeze(-1)

    def combinar(self, y1: torch.Tensor, y2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return y1 + self.signo * self.puerta(x) * y2

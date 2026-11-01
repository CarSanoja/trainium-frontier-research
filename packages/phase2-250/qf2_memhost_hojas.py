"""QF2_MEMHOST: one gradient leaf per micro-step.

Slicing a single leaf E[B, T, S*W] per micro-step makes autograd run SliceBackward and AccumulateGrad over the whole E
outside the compiled graph, once per micro-step. Here each micro-step gets its own contiguous leaf, filled from E
without gradient, so its gradient comes straight out of the compiled backward. The step gradient is their
concatenation, padded with zeros up to the rows of E: the same values as the gradient of the sliced leaf.
"""
from __future__ import annotations

import torch


class Hojas:
    def __init__(self):
        self.hojas: list[torch.Tensor] = []

    def repartir(self, e: torch.Tensor, n_micro: int, filas: int) -> list[torch.Tensor]:
        forma = (filas,) + tuple(e.shape[1:])
        if len(self.hojas) != n_micro or tuple(self.hojas[0].shape) != forma or self.hojas[0].dtype != e.dtype:
            self.hojas = [torch.zeros(forma, dtype=e.dtype, device=e.device).requires_grad_(True) for _ in range(n_micro)]
        with torch.no_grad():
            for i, h in enumerate(self.hojas):
                h.copy_(e[i * filas:(i + 1) * filas])
        for h in self.hojas:
            h.grad = None
        return self.hojas

    def gradiente(self, filas_e: int) -> torch.Tensor | None:
        if not self.hojas or any(h.grad is None for h in self.hojas):
            return None
        g = torch.cat([h.grad for h in self.hojas])
        if g.shape[0] < filas_e:
            g = torch.cat([g, g.new_zeros((filas_e - g.shape[0],) + tuple(g.shape[1:]))])
        return g

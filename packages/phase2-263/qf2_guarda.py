"""QF2_GUARDA: skip an optimizer step whose gradients are not finite on any rank.

Before the optimizer runs, every rank sums its local gradients in fp32 (one reduction per tensor), the non-finite flag
travels in one int32 all_reduce(MAX), and the host reads it. A bad step leaves weights, moments and the step counter
untouched and is not recorded as a step boundary; the data it consumed is dropped. After LIMITE consecutive bad steps the
run stops and saves the last finite weights instead of crashing. The flag is computed from step 0 so its small graphs
compile inside the excluded startup; skipping only starts at DESDE, after the budget clock has been reset.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

LIMITE = int(os.environ.get("QF2_GUARDA_LIMITE", "8"))


class Guarda:
    def __init__(self, params, desde: int, rank: int, limite: int = LIMITE):
        self.params = [p for p in params if p.requires_grad]
        self.desde, self.rank, self.limite = desde, rank, limite
        self.saltos = 0
        self.seguidos = 0

    def descripcion(self) -> str:
        return f"qf2_guarda: tensores={len(self.params)} desde_paso={self.desde} limite_seguidos={self.limite}"

    def no_finito(self) -> torch.Tensor:
        sumas = [p.grad.sum(dtype=torch.float32) for p in self.params if p.grad is not None]
        return (~torch.isfinite(torch.stack(sumas).sum())).to(torch.int32).reshape(1)

    def saltar(self, paso: int) -> bool:
        malo = self.no_finito()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(malo, op=dist.ReduceOp.MAX)
        if int(malo.item()) == 0 or paso < self.desde:
            self.seguidos = 0
            return False
        self.saltos += 1
        self.seguidos += 1
        if self.rank == 0:
            print(f"qf2_guarda: salto paso={paso} saltos={self.saltos} seguidos={self.seguidos}", flush=True)
        return True

    def agotada(self) -> bool:
        return self.seguidos >= self.limite

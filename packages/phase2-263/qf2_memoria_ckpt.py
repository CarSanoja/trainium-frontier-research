"""Consolidation of the row-sharded memory tables into full tables for the checkpoint (QF2_MEM_FILAS). Collective: every rank
must call it; only rank 0 returns tensors (keys of the replicated evaluation model, memoria.<k>.weight)."""
from __future__ import annotations

import torch
import torch.distributed as dist

import qf2_memoria

TROZO = 1024


def consolidar(mem: qf2_memoria.Memoria, dtype: torch.dtype, trozo: int = TROZO) -> dict:
    salida = {}
    r, w = mem.r, mem.world
    for k, shard in zip(qf2_memoria.CLAVES, mem.shards):
        total = torch.empty(mem.filas, mem.ancho, dtype=dtype) if mem.rank == 0 else None
        for c in range(0, r, trozo):
            parte = shard.detach()[c:c + trozo].to(dtype).contiguous()
            n = parte.size(0)
            if w > 1:
                todo = parte.new_empty(w * n, parte.size(1))
                dist.all_gather_into_tensor(todo, parte)
            else:
                todo = parte
            if mem.rank == 0:
                todo = todo.cpu()
                for q in range(w):
                    total[q * r + c:q * r + c + n] = todo[q * n:(q + 1) * n]
        if mem.rank == 0:
            salida[f"memoria.{k}.weight"] = total
    return salida


def cargar_trozo(mem: qf2_memoria.Memoria, tablas: dict) -> None:
    with torch.no_grad():
        for k, shard in zip(qf2_memoria.CLAVES, mem.shards):
            shard.copy_(tablas[f"memoria.{k}.weight"][mem.rank * mem.r:(mem.rank + 1) * mem.r].to(shard.dtype))

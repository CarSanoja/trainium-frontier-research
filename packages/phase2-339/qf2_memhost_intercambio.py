"""QF2_MEMHOST: gradient exchange between the ranks of a host through shared memory.

Each rank aggregates its own row gradients (unique ids sorted ascending, fp32 sums), publishes them in its region and
the offsets of every owner's slice; after barrier A each owner reads its slice from all origins in rank order and sums
per row. The region holds as many rows as references per rank, so it cannot overflow.
"""
from __future__ import annotations

from pathlib import Path

import torch

import qf2_memhost_sincro as sincro


def agregar(ids: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    uid, inv = torch.unique(ids, return_inverse=True)
    out = torch.zeros(uid.numel(), g.size(1), dtype=torch.float32)
    out.index_add_(0, inv, g)
    return uid, out


class Intercambio:
    def __init__(self, ruta: Path, rango: int, mundo: int, capacidad: int, ancho: int, crear: bool):
        self.rango, self.mundo, self.cap, self.w = rango, mundo, capacidad, ancho
        self.cab = sincro.mapear(ruta / "cab.i64", torch.int64, mundo * (mundo + 2), crear).view(mundo, mundo + 2)
        self.ids = sincro.mapear(ruta / "ids.i64", torch.int64, mundo * capacidad, crear).view(mundo, capacidad)
        self.gr = sincro.mapear(ruta / "gr.f32", torch.float32, mundo * capacidad * ancho, crear).view(mundo, capacidad, ancho)

    def publicar(self, uid: torch.Tensor, g: torch.Tensor, limites: torch.Tensor) -> None:
        n = uid.numel()
        if n > self.cap:
            raise RuntimeError(f"qf2_memhost: {n} filas unicas superan la capacidad {self.cap}")
        self.ids[self.rango, :n] = uid
        self.gr[self.rango, :n] = g
        self.cab[self.rango, 0] = n
        self.cab[self.rango, 1:] = torch.searchsorted(uid, limites)

    def reunir(self, propietario: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, gs = [], []
        for s in range(self.mundo):
            a, b = int(self.cab[s, 1 + propietario]), int(self.cab[s, 2 + propietario])
            if b > a:
                ids.append(self.ids[s, a:b].clone())
                gs.append(self.gr[s, a:b].clone())
        if not ids:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, self.w)
        return agregar(torch.cat(ids), torch.cat(gs))

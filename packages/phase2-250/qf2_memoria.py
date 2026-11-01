"""QF2_MEM_FILAS: row-sharded hashed bigram/trigram memory (two tables of FILAS x ANCHO), added to the input embedding.

Same hash and sentinel rules as qf2_ngramas.indices. In training every rank owns a contiguous block of FILAS/world rows of each
table (plain nn.Parameter outside the model and outside DistMuonAdamW). A lookup is: all_gather of the int32 row ids of all
ranks, a masked local gather of the owned rows, and a reduce_scatter (SUM) of the outputs, which gives each rank back its own
queries exactly (one non-zero term plus zeros). The backward is an all_gather of the output gradient and a local scatter-add
into the shard, scaled by 1/world to match the AVG gradient average of the replicated parameters. Evaluation uses the same
tables replicated in the model (combinar is shared by both paths). QF2_MEM_TRANSPORTE=a2a replaces the lookup transport by
an all_to_all_single to the owner and back (qf2_memoria_a2a); "b" is the one described above.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import qf2_memoria_a2a
import qf2_ngramas

BLOQUE_INICIO = 64
STD_INICIO = 0.8
CLAVES = ("2", "3")
TROZO = 65536
TOPE_BYTES = 8 * 10 ** 9
TRANSPORTES = ("b", "a2a")


def comprobar(filas: int, ancho: int, world: int = 1) -> None:
    qf2_ngramas.comprobar(filas)
    if ancho < 1:
        raise ValueError(f"QF2_MEM_ANCHO={ancho} no es valido")
    if filas % (world * BLOQUE_INICIO) != 0:
        raise ValueError(f"QF2_MEM_FILAS={filas} debe ser multiplo de world*{BLOQUE_INICIO}={world * BLOQUE_INICIO}")


def iniciar_filas(semilla: int, tabla: int, fila0: int, n: int, ancho: int) -> torch.Tensor:
    out = torch.empty(n, ancho)
    for b in range(n // BLOQUE_INICIO):
        g = torch.Generator().manual_seed(((semilla * 2 + tabla) << 32) + fila0 // BLOQUE_INICIO + b)
        out[b * BLOQUE_INICIO:(b + 1) * BLOQUE_INICIO] = torch.randn(BLOQUE_INICIO, ancho, generator=g) * STD_INICIO
    return out


def tablas(filas: int, ancho: int, dtype: torch.dtype = torch.float32, device=None) -> nn.ModuleDict:
    comprobar(filas, ancho)
    return nn.ModuleDict({k: nn.Embedding(filas, ancho, _weight=torch.empty(filas, ancho, dtype=dtype, device=device)) for k in CLAVES})


@torch.no_grad()
def iniciar(mods: nn.ModuleDict, semilla: int) -> None:
    for i, k in enumerate(CLAVES):
        w = mods[k].weight
        w.copy_(iniciar_filas(semilla, i, 0, w.size(0), w.size(1)))


def mundo() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def trozos(n: int, world: int) -> list[tuple[int, int]]:
    c = max(1, TROZO // world)
    return [(s, min(n, s + c)) for s in range(0, n, c)]


class Consulta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, filas_g, rank, world, dtype):
        r = shard.size(0)
        n = filas_g.numel() // world
        ids = filas_g.view(world, n)
        copia = shard.detach().to(dtype)
        out = copia.new_empty(n, copia.size(1))
        for s, e in trozos(n, world):
            f = ids[:, s:e].reshape(-1)
            propia = (f // r) == rank
            loc = (f - rank * r).clamp(0, r - 1)
            local = copia[loc] * propia.unsqueeze(1).to(dtype)
            if world > 1:
                dist.reduce_scatter_tensor(out[s:e], local, op=dist.ReduceOp.SUM)
            else:
                out[s:e] = local
        ctx.save_for_backward(filas_g)
        ctx.meta = (r, rank, world)
        return out

    @staticmethod
    def backward(ctx, g):
        (filas_g,) = ctx.saved_tensors
        r, rank, world = ctx.meta
        g = g.contiguous()
        n = g.size(0)
        ids = filas_g.view(world, n)
        dw = torch.zeros(r + 1, g.size(1), dtype=torch.float32, device=g.device)
        for s, e in trozos(n, world):
            f = ids[:, s:e].reshape(-1)
            if world > 1:
                todo = g.new_empty(world * (e - s), g.size(1))
                dist.all_gather_into_tensor(todo, g[s:e].contiguous())
            else:
                todo = g[s:e]
            propia = (f // r) == rank
            destino = torch.where(propia, f - rank * r, torch.full_like(f, r))
            dw.index_add_(0, destino, todo.float())
        return dw[:r] / world, None, None, None, None


class Memoria:
    def __init__(self, filas: int, ancho: int, device, semilla: int, bos: int, transporte: str = "b"):
        if transporte not in TRANSPORTES:
            raise ValueError(f"QF2_MEM_TRANSPORTE={transporte} no es valido, debe ser uno de {TRANSPORTES}")
        self.transporte = transporte
        self.desbordes = torch.zeros((), dtype=torch.int64, device=device)
        self.rank, self.world = mundo()
        comprobar(filas, ancho, self.world)
        por_rango = filas * ancho * len(CLAVES) * 16 // self.world
        if por_rango > TOPE_BYTES:
            raise ValueError(f"memoria repartida: {filas} filas x {ancho} x {len(CLAVES)} tablas x 16 B / {self.world} rangos = {por_rango} B por rango, supera {TOPE_BYTES}; hace falta mas rangos")
        self.filas, self.ancho, self.bos = filas, ancho, bos
        self.r = filas // self.world
        self.shards = [nn.Parameter(iniciar_filas(semilla, i, self.rank * self.r, self.r, ancho).to(device)) for i in range(len(CLAVES))]

    def parametros(self) -> list[nn.Parameter]:
        return list(self.shards)

    def zero_grad(self) -> None:
        for p in self.shards:
            p.grad = None

    def reunir(self, h: torch.Tensor) -> torch.Tensor:
        h = h.reshape(-1).to(torch.int32).contiguous()
        if self.world > 1:
            todo = h.new_empty(self.world * h.numel())
            dist.all_gather_into_tensor(todo, h)
            h = todo
        return h.to(torch.int64)

    def consultar(self, idx: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.shape
        hs = qf2_ngramas.indices(idx, self.filas, self.bos)
        if self.transporte == "a2a":
            return tuple(qf2_memoria_a2a.consultar(s, h, self.rank, self.world, self.desbordes, dtype).view(b, t, self.ancho) for s, h in zip(self.shards, hs))
        return tuple(Consulta.apply(s, self.reunir(h), self.rank, self.world, dtype).view(b, t, self.ancho) for s, h in zip(self.shards, hs))


def consulta_replicada(mods: nn.ModuleDict, idx: torch.Tensor, filas: int, bos: int) -> tuple[torch.Tensor, torch.Tensor]:
    h2, h3 = qf2_ngramas.indices(idx, filas, bos)
    return mods["2"](h2), mods["3"](h3)


def combinar(m2: torch.Tensor, m3: torch.Tensor, proy: nn.Module, esc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = torch.cat([F.rms_norm(m2.to(x.dtype), (m2.size(-1),)), F.rms_norm(m3.to(x.dtype), (m3.size(-1),))], dim=-1)
    return x + esc.to(x.dtype) * proy(m)

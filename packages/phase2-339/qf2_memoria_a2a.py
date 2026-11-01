"""QF2_MEM_TRANSPORTE=a2a: each query row travels only to the rank that owns it and back (all_to_all_single), instead of every
query visiting every rank (all_gather of ids + reduce_scatter of a dense output).

Every rank groups its int32 row ids by owner (owner = id // rows_per_rank) into W blocks of fixed capacity C (static shapes):
the position of a query inside its block is a running count per owner (cumsum of a one-hot). all_to_all_single swaps the id
blocks, the owner gathers its rows (-1 pads give zero rows), and all_to_all_single swaps the rows back; the original order is
restored by a gather on the destination index. The backward is symmetric: gradients go to the owner with all_to_all_single and
are added in fp32 into the shard (extra trash row r for pads), scaled by 1/world like the b transport.

Overflow (more than C queries of one rank for one owner in one micro-step): the excess queries are deterministic (the last ones
in token order), get a zero row in the forward and send no gradient, and are added to the counter tensor `contador`, which the
training loop prints. C = ceil(HOLGURA * n / W) rounded up to MULTIPLO. With n = 8192 tokens and W = 64 each pair receives
Binomial(8192, 1/64): mean 128, sd 11.2; P(count > 256) = 2.9e-24 and P(count > 192) = 4.1e-8 under independent uniform hashing (exact binomial tail), i.e. 5e-20 and 7e-4 expected overflow events per step over 64 ranks x 64 owners x 2 tables x 2 micro-steps.
Real text repeats n-grams (and the BOS sentinel row sits in one owner), so the independent model is a floor, not a promise.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist

HOLGURA = 2.0
MULTIPLO = 8


def capacidad(n: int, world: int) -> int:
    c = math.ceil(n * HOLGURA / world)
    return max(MULTIPLO, -(-c // MULTIPLO) * MULTIPLO)


def intercambiar(salida: torch.Tensor, entrada: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_to_all_single(salida, entrada)
    else:
        salida.copy_(entrada)
    return salida


@torch.no_grad()
def plan(h: torch.Tensor, r: int, world: int, c: int, contador: torch.Tensor):
    n = h.numel()
    dueno = (h // r).to(torch.int64)
    una = (dueno.unsqueeze(1) == torch.arange(world, device=h.device)).to(torch.int32)
    acum = torch.cumsum(una, dim=0)
    puesto = acum.gather(1, dueno.unsqueeze(1)).squeeze(1) - 1
    contador += (acum[-1] - c).clamp(min=0).sum().to(contador.dtype)
    cabe = puesto < c
    base = world * c
    sitio = torch.where(cabe, dueno * c + puesto, base + torch.arange(n, device=h.device))
    envio = torch.full((base + n,), -1, dtype=torch.int32, device=h.device)
    envio.index_copy_(0, sitio, h)
    recibo = intercambiar(torch.empty(base, dtype=torch.int32, device=h.device), envio[:base].contiguous(), world)
    return sitio.clamp(max=base - 1), cabe, recibo


class ConsultaA2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, destino, cabe, recibo, rank, world, dtype):
        r = shard.size(0)
        valido = recibo >= 0
        loc = (recibo.to(torch.int64) - rank * r).clamp(0, r - 1)
        copia = shard.detach().to(dtype)
        filas = copia[loc] * valido.unsqueeze(1).to(dtype)
        vuelta = intercambiar(torch.empty_like(filas), filas, world)
        ctx.save_for_backward(destino, cabe, loc, valido)
        ctx.meta = (r, world)
        return vuelta[destino] * cabe.unsqueeze(1).to(dtype)

    @staticmethod
    def backward(ctx, g):
        destino, cabe, loc, valido = ctx.saved_tensors
        r, world = ctx.meta
        g = (g * cabe.unsqueeze(1).to(g.dtype)).contiguous()
        envio = torch.zeros(loc.numel(), g.size(1), dtype=g.dtype, device=g.device).index_add_(0, destino, g)
        recibo = intercambiar(torch.empty_like(envio), envio, world)
        dw = torch.zeros(r + 1, g.size(1), dtype=torch.float32, device=g.device)
        dw.index_add_(0, torch.where(valido, loc, torch.full_like(loc, r)), recibo.float())
        return dw[:r] / world, None, None, None, None, None, None


def consultar(shard: torch.Tensor, h: torch.Tensor, rank: int, world: int, contador: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    h = h.reshape(-1).to(torch.int32).contiguous()
    c = capacidad(h.numel(), world)
    destino, cabe, recibo = plan(h, shard.size(0), world, c, contador)
    return ConsultaA2A.apply(shard, destino, cabe, recibo, rank, world, dtype)

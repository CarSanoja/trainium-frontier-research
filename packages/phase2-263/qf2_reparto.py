"""Optimizer communication at many ranks (QF2_COM_ESCALARES, QF2_COM_MUON, QF2_COM_GATHER_BF16; all default OFF).

ESCALARES: every AdamW parameter with numel < 1024 is averaged with one flattened all_reduce per step.
MUON: all Muon matrices of every shape travel in one reduce_scatter and one all_gather. Whole matrices go to ranks by a
longest-processing-time greedy on numel (ties by index); each rank's chunk is padded only to the largest chunk. The
update of the owned matrices is DistMuonAdamW._muon_owned, the same code the per-shape path runs, batched by shape.
GATHER_BF16: the all_gather of updated Muon matrices and big AdamW tables sends bf16. Each rank keeps the exact fp32
master of what it owns and stores the bf16 values of what it does not. restaurar_fp32 puts the fp32 masters back on
every rank (one fp32 all_gather) before anything consumes full fp32 parameters.
RS_BF16: the reduce_scatter of gradients (the Muon slab and the big AdamW tables) sends bf16 and widens the averaged
slice back to fp32. Needs MUON; small parameters keep their fp32 all_reduce.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

import qf2_plano

PEQUENO = 1024


def _alias_bf16(buf: torch.Tensor, forma: tuple) -> torch.Tensor:
    n = math.prod(forma)
    if buf.dtype != torch.float32 or not buf.is_contiguous() or buf.numel() < n:
        return torch.empty(forma, dtype=torch.bfloat16, device=buf.device)
    return buf.view(-1).view(torch.bfloat16)[:n].view(forma)


def _copiar(destinos: list[torch.Tensor], fuente: torch.Tensor) -> None:
    tamanos = [p.numel() for p in destinos]
    if fuente.dtype == destinos[0].dtype:
        torch.split_with_sizes_copy(fuente[:sum(tamanos)], tamanos, out=[p.data.view(-1) for p in destinos])
        return
    inicio = 0
    for p, n in zip(destinos, tamanos):
        p.copy_(fuente[inicio:inicio + n].view(p.shape))
        inicio += n


def pequenos(param_groups) -> list[torch.Tensor]:
    return [p for g in param_groups if g["kind"] == "adamw" for p in g["params"] if p.numel() < PEQUENO]


@torch.no_grad()
def promediar_pequenos(param_groups) -> None:
    params = pequenos(param_groups)
    if not params:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    plano = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(plano, op=dist.ReduceOp.AVG)
    torch.split_with_sizes_copy(plano, [p.numel() for p in params], out=[p.grad.view(-1) for p in params])


def plan_lpt(tamanos: list[int], mundo: int) -> tuple[list[int], list[int]]:
    carga = [0] * mundo
    dueno = [0] * len(tamanos)
    for i in sorted(range(len(tamanos)), key=lambda i: (-tamanos[i], i)):
        r = min(range(mundo), key=lambda k: (carga[k], k))
        dueno[i] = r
        carga[r] += tamanos[i]
    return dueno, carga


class RepartoMuon:
    def __init__(self, grupos, rango: int, mundo: int, transitorio: bool = False):
        self.rango, self.mundo = rango, mundo
        self.params = [p for g in grupos for p in g["params"]]
        if not self.params:
            raise ValueError("QF2_COM_MUON=1 needs at least one Muon matrix")
        if any(tuple(p.shape) != tuple(g["params"][0].shape) for g in grupos for p in g["params"]):
            raise ValueError("Muon groups must contain equal-shape matrices")
        if len({(p.dtype, p.device) for p in self.params}) != 1:
            raise ValueError("QF2_COM_MUON=1 needs one dtype and device for every Muon matrix")
        self.dueno, self.carga = plan_lpt([p.numel() for p in self.params], mundo)
        self.trozo = max(self.carga)
        cursor = [r * self.trozo for r in range(mundo)]
        self.inicio = []
        for p, r in zip(self.params, self.dueno):
            self.inicio.append(cursor[r])
            cursor[r] += p.numel()
        self.por_rango = [[p for p, d in zip(self.params, self.dueno) if d == r] for r in range(mundo)]
        self.lotes, base = [], 0
        for g in grupos:
            mios = [i for i in range(base, base + len(g["params"])) if self.dueno[i] == rango]
            if mios:
                ini = self.inicio[mios[0]] - rango * self.trozo
                self.lotes.append((g, tuple(g["params"][0].shape), ini, [self.params[i] for i in mios]))
            base += len(g["params"])
        p0 = self.params[0]
        self.losa = qf2_plano.Losa(mundo * self.trozo, p0.dtype, p0.device) if transitorio else None
        self.plano = None if transitorio else torch.zeros(mundo * self.trozo, dtype=p0.dtype, device=p0.device)
        self.grad = torch.zeros(self.trozo, dtype=p0.dtype, device=p0.device)
        self.nuevo = torch.zeros(self.trozo, dtype=p0.dtype, device=p0.device)
        self.grad16 = None

    def descripcion(self) -> str:
        real = sum(p.numel() for p in self.params)
        return (f"muon_matrices={len(self.params)} chunk={self.trozo} flat={self.mundo * self.trozo} real={real} "
                f"own={len(self.por_rango[self.rango])} load={self.carga[self.rango]}")

    def _propios(self) -> list[tuple]:
        salida = []
        for group, forma, ini, params in self.lotes:
            fin = ini + len(params) * params[0].numel()
            salida.append((group, forma, params, self.nuevo[ini:fin].view(len(params), *forma), ini, fin))
        return salida

    @torch.no_grad()
    def paso(self, opt, bf16: bool, rs_bf16: bool = False) -> None:
        if self.losa is not None:
            self.plano = self.losa.abrir()
        plano = _alias_bf16(self.plano, (self.plano.numel(),)) if rs_bf16 else self.plano
        for p, i0 in zip(self.params, self.inicio):
            vista = plano[i0:i0 + p.numel()].view(p.shape)
            if p.grad is None:
                vista.zero_()
            else:
                vista.copy_(p.grad)
            p.grad = None
        if rs_bf16:
            if self.grad16 is None:
                self.grad16 = torch.empty(self.trozo, dtype=torch.bfloat16, device=self.grad.device)
            dist.reduce_scatter_tensor(self.grad16, plano, op=dist.ReduceOp.AVG)
            self.grad.copy_(self.grad16)
        else:
            dist.reduce_scatter_tensor(self.grad, self.plano, op=dist.ReduceOp.AVG)
        for group, forma, params, w, ini, fin in self._propios():
            for k, p in enumerate(params):
                w[k].copy_(p)
            g = self.grad[ini:fin].view(len(params), *forma)
            opt._muon_owned(group, forma, g, w, opt.state[group["params"][0]])
        self._reunir(bf16)
        if self.losa is not None:
            self.plano = self.losa.cerrar()

    @torch.no_grad()
    def _reunir(self, bf16: bool) -> None:
        if bf16:
            salida = _alias_bf16(self.plano, (self.plano.numel(),))
            dist.all_gather_into_tensor(salida, self.nuevo.to(torch.bfloat16))
        else:
            salida = self.plano
            dist.all_gather_into_tensor(salida, self.nuevo)
        for r, params in enumerate(self.por_rango):
            if params:
                fuente = self.nuevo if bf16 and r == self.rango else salida[r * self.trozo:(r + 1) * self.trozo]
                _copiar(params, fuente)

    @torch.no_grad()
    def restaurar(self) -> None:
        if self.losa is not None:
            self.plano = self.losa.abrir()
        for _group, _forma, params, w, _ini, _fin in self._propios():
            for k, p in enumerate(params):
                w[k].copy_(p)
        self._reunir(False)
        if self.losa is not None:
            self.plano = self.losa.cerrar()


@torch.no_grad()
def reducir_tabla_bf16(grad_slice, grad) -> None:
    trozo = torch.empty(grad_slice.shape, dtype=torch.bfloat16, device=grad.device)
    dist.reduce_scatter_tensor(trozo, grad.to(torch.bfloat16), op=dist.ReduceOp.AVG)
    grad_slice.copy_(trozo)


@torch.no_grad()
def reunir_tabla_bf16(p, p_slice, grad, inicio: int, filas: int) -> None:
    salida = _alias_bf16(grad, tuple(p.shape))
    dist.all_gather_into_tensor(salida, p_slice.to(torch.bfloat16))
    p[:inicio].copy_(salida[:inicio])
    p[inicio + filas:].copy_(salida[inicio + filas:])


@torch.no_grad()
def reunir_muon_bf16(stacked, updated, params, inicio: int, propios: int) -> None:
    salida = _alias_bf16(stacked, tuple(stacked.shape))
    dist.all_gather_into_tensor(salida, updated.to(torch.bfloat16))
    for i, p in enumerate(params):
        p.copy_(updated[i - inicio] if inicio <= i < inicio + propios else salida[i])


@torch.no_grad()
def _restaurar_forma(opt, params) -> None:
    forma, p0 = tuple(params[0].shape), params[0]
    trozo = -(-len(params) // opt.world_size)
    stacked = opt._shared_muon_buffer("stacked", trozo * opt.world_size, forma, p0.dtype, p0.device, zero=True)
    updated = opt._shared_muon_buffer("updated", trozo, forma, p0.dtype, p0.device)
    inicio = opt.rank * trozo
    propios = min(trozo, max(0, len(params) - inicio))
    for i in range(propios):
        updated[i].copy_(params[inicio + i])
    updated[propios:].zero_()
    dist.all_gather_into_tensor(stacked, updated)
    opt._copy_muon_params(params, stacked)


@torch.no_grad()
def restaurar_fp32(opt, reparto: RepartoMuon | None) -> None:
    for group in opt.param_groups:
        if group["kind"] == "adamw":
            for p in group["params"]:
                if p.numel() >= PEQUENO:
                    filas = p.shape[0] // opt.world_size
                    dist.all_gather_into_tensor(p, p[opt.rank * filas:(opt.rank + 1) * filas])
        elif group["kind"] == "muon" and reparto is None and group["params"]:
            _restaurar_forma(opt, group["params"])
    if reparto is not None:
        reparto.restaurar()


def descripcion(escalares: bool, muon: bool, bf16: bool, opt, rs_bf16: bool = False) -> str:
    texto = f"qf2_reparto: escalares={int(escalares)} muon={int(muon)} gather_bf16={int(bf16)} rs_bf16={int(rs_bf16)}"
    if hasattr(opt, "world_size"):
        texto += f" world={opt.world_size} small_params={len(pequenos(opt.param_groups))}"
    reparto = getattr(opt, "_reparto", None)
    return f"{texto} {reparto.descripcion()}" if reparto is not None else texto

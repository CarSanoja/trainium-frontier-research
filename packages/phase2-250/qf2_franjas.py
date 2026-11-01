"""Muon transport in K strips (QF2_COM_FRANJAS=K, default 0 = the flat slab of RepartoMuon untouched).

The flat path builds a slab of world * chunk elements (row r = the segment owned by rank r), reduce_scatters it, runs
Newton-Schulz on the owner and all_gathers into the same slab. Here the logical slab is cut by columns: strip k holds
columns [k*s, k*s + w_k) of every row, s = ceil(chunk / K). One transport buffer of world * s elements is reused by
all strips. Owners, offsets inside the segment and the update (DistMuonAdamW._muon_owned) are the ones of RepartoMuon
because this class inherits them; only the movement of bytes changes. A gradient is released as soon as its last
strip has been packed. Per element the reduction sees the same world values in the same rank slot, so the result
matches the flat path bit for bit on backends whose reduction order depends on the row and not on the strip width.
Collectives per step: K reduce_scatter + K all_gather instead of 1 + 1; restaurar() adds K all_gather.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from qf2_reparto import RepartoMuon


class Franjas:
    def __init__(self, reparto: RepartoMuon, k: int):
        if k < 1:
            raise ValueError("QF2_COM_FRANJAS must be >= 1")
        self.r, self.k = reparto, k
        trozo, mundo = reparto.trozo, reparto.mundo
        self.s = -(-trozo // k)
        self.ancho = [min(self.s, trozo - j * self.s) for j in range(k) if j * self.s < trozo]
        self.tramos = len(self.ancho)
        self.piezas = [[] for _ in self.ancho]
        self.ultimo = {}
        for i, p in enumerate(reparto.params):
            off = reparto.inicio[i] - reparto.dueno[i] * trozo
            for j, w in enumerate(self.ancho):
                lo = j * self.s
                a, b = max(off, lo), min(off + p.numel(), lo + w)
                if a < b:
                    self.piezas[j].append((i, a - off, b - off, reparto.dueno[i] * w + (a - lo), a))
                    self.ultimo[i] = j
        self.buf = None
        self.pico_bytes = 0

    def bytes_transporte(self) -> int:
        return self.r.mundo * self.s * torch.empty((), dtype=self.r.grad.dtype).element_size()

    def abrir(self) -> None:
        g = self.r.grad
        self.buf = torch.empty(self.r.mundo * self.s, dtype=g.dtype, device=g.device)
        self.pico_bytes = max(self.pico_bytes, self.bytes_transporte())

    def cerrar(self) -> None:
        self.buf = None

    @torch.no_grad()
    def reducir(self) -> None:
        r = self.r
        liberar = [[] for _ in self.ancho]
        for i, j in self.ultimo.items():
            liberar[j].append(i)
        for j, w in enumerate(self.ancho):
            envio = self.buf[:r.mundo * w]
            envio.zero_()
            for i, a, b, dst, _lo in self.piezas[j]:
                g = r.params[i].grad
                if g is not None:
                    envio[dst:dst + b - a].copy_(g.reshape(-1)[a:b])
            lo = j * self.s
            dist.reduce_scatter_tensor(r.grad[lo:lo + w], envio, op=dist.ReduceOp.AVG)
            for i in liberar[j]:
                r.params[i].grad = None

    @torch.no_grad()
    def reunir(self, bf16: bool) -> None:
        r = self.r
        for j, w in enumerate(self.ancho):
            lo = j * self.s
            n = r.mundo * w
            if bf16:
                salida = self.buf.view(torch.bfloat16)[:n]
                dist.all_gather_into_tensor(salida, r.nuevo[lo:lo + w].to(torch.bfloat16))
            else:
                salida = self.buf[:n]
                dist.all_gather_into_tensor(salida, r.nuevo[lo:lo + w])
            for i, a, b, dst, pos in self.piezas[j]:
                propio = bf16 and r.dueno[i] == r.rango
                fuente = r.nuevo[pos:pos + b - a] if propio else salida[dst:dst + b - a]
                r.params[i].data.view(-1)[a:b].copy_(fuente)


class RepartoFranjas(RepartoMuon):
    def __init__(self, grupos, rango: int, mundo: int, franjas: int):
        super().__init__(grupos, rango, mundo, True)
        self.losa = None
        self.plano = None
        self.franjas = Franjas(self, franjas)

    def descripcion(self) -> str:
        f = self.franjas
        return f"{super().descripcion()} strips={f.tramos} transport={f.bytes_transporte()}"

    @torch.no_grad()
    def paso(self, opt, bf16: bool, rs_bf16: bool = False) -> None:
        if rs_bf16:
            raise ValueError("QF2_COM_FRANJAS needs QF2_COM_RS_BF16=0")
        self.franjas.abrir()
        self.franjas.reducir()
        for group, forma, params, w, ini, fin in self._propios():
            for k, p in enumerate(params):
                w[k].copy_(p)
            g = self.grad[ini:fin].view(len(params), *forma)
            opt._muon_owned(group, forma, g, w, opt.state[group["params"][0]])
        self.franjas.reunir(bf16)
        self.franjas.cerrar()

    @torch.no_grad()
    def _reunir(self, bf16: bool) -> None:
        self.franjas.reunir(bf16)

    @torch.no_grad()
    def restaurar(self) -> None:
        self.franjas.abrir()
        for _group, _forma, params, w, _ini, _fin in self._propios():
            for k, p in enumerate(params):
                w[k].copy_(p)
        self.franjas.reunir(False)
        self.franjas.cerrar()

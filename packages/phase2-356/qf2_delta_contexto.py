"""Rama contextual efimera de escritura delta o control aditivo."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from qf2_delta_arbol import estados_entrantes


class DeltaContexto(nn.Module):
    def __init__(self, ancho, cabezas=2, clave=32, bos=0, semilla=86401, modo="delta"):
        super().__init__()
        if modo not in ("delta", "aditiva"):
            raise ValueError("Escritura delta o aditiva")
        self.ancho, self.cabezas, self.clave = ancho, cabezas, clave
        self.bos, self.modo = bos, modo
        with torch.random.fork_rng(devices=[]):
            generador = torch.Generator(device="cpu").manual_seed(semilla)
            self.entidad = nn.Linear(ancho, cabezas*clave, bias=False)
            self.atributo = nn.Linear(ancho, cabezas*clave, bias=False)
            self.consulta = nn.Linear(ancho, cabezas*clave, bias=False)
            self.valor = nn.Linear(ancho, cabezas*clave, bias=False)
            self.beta = nn.Linear(ancho, cabezas, bias=True)
            self.salida = nn.Linear(cabezas*clave, ancho, bias=False)
            for p in self.matrices()[:-1]:
                nn.init.normal_(p, std=1/math.sqrt(ancho), generator=generador)
            nn.init.zeros_(self.beta.bias)
            nn.init.zeros_(self.salida.weight)
        self.register_buffer("identidad", torch.eye(clave), persistent=False)

    def matrices(self):
        return [m.weight for m in (self.entidad, self.atributo, self.consulta, self.valor, self.beta, self.salida)]

    @staticmethod
    def proyectar(m, x):
        sesgo = m.bias.to(x.dtype) if m.bias is not None else None
        return F.linear(x, m.weight.to(x.dtype), sesgo)

    def componentes(self, x, ids):
        b, t, _ = x.shape
        h = F.rms_norm(x, (self.ancho,))
        def cabeza(m):
            y = self.proyectar(m, h).float()
            return y.reshape(b, t, self.cabezas, self.clave).permute(0, 2, 1, 3).contiguous()
        e, atributo = cabeza(self.entidad), cabeza(self.atributo)
        k = F.normalize(e*atributo, dim=-1, eps=1e-6)
        q = F.normalize(cabeza(self.consulta), dim=-1, eps=1e-6)
        valor = cabeza(self.valor)
        beta = self.proyectar(self.beta, h).float().sigmoid().permute(0, 2, 1).contiguous()
        c = (ids != self.bos).to(torch.float32)[:, None, :, None, None]
        return k, q, valor, beta, c

    def retornos(self, x, ids):
        k, q, valor, beta, c = self.componentes(x, ids)
        kk = k.unsqueeze(-1) * k.unsqueeze(-2)
        identidad = self.identidad.to(dtype=k.dtype)
        a = (identidad - beta[..., None, None]*kk) * c if self.modo == "delta" else identidad.expand_as(kk)*c
        b = beta[..., None, None] * k.unsqueeze(-1) * valor.unsqueeze(-2)
        p = estados_entrantes(a, b)
        lote, cabezas, t, d = q.shape
        r = torch.bmm(q.reshape(-1, 1, d), p.reshape(-1, d, d)).reshape(lote, cabezas, t, d)
        return r * c.squeeze(-1)

    def forward(self, x, ids):
        r = self.retornos(x, ids)
        b, _, t, _ = r.shape
        r = r.permute(0, 2, 1, 3).reshape(b, t, -1).to(x.dtype)
        return x + self.proyectar(self.salida, r)


def aplicar(x, ids, modulo=None):
    return x if modulo is None else modulo(x, ids)


def crear(config, bos):
    if not config.delta_ctx:
        return None
    campos = config.delta_ctx.split(":")
    if len(campos) != 2 or campos[0] not in ("delta", "aditiva"):
        raise ValueError("QF2_DELTA_CTX requiere delta:capa o aditiva:capa")
    capa = int(campos[1])
    if not 0 <= capa < config.n_layer or config.registro or config.prefijo_bos:
        raise ValueError("Delta necesita capa valida y posiciones documentales sin registro/prefijo artificial")
    modulo = DeltaContexto(config.n_embd, cabezas=2, clave=32, bos=bos, modo=campos[0])
    modulo.capa = capa
    return modulo

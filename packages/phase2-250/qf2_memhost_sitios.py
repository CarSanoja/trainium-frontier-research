"""QF2_MEMHOST: where the recovered vector enters the network.

One vector per token [B, T, S*W] feeds three kinds of sites. The input site adds scale * Linear(e) to the normalised
embedding. A gated site after block i follows Engram: a = sigmoid(<rms(x), rms(Linear_key(e))> / sqrt(d)),
x += a * Linear_val(e). With QF2_MEMHOST_PUERTA_CABEZA (default on in this package) the gate is one per head: the d channels
of q*k split into S contiguous groups, u_h = sum(group h) / sqrt(d/S), and Linear_val receives e with each head slice scaled
by its a_h (scalar gate when d is not divisible by S, as in the d4 CPU check). cfg.puerta swaps the sigmoid for a(u) = 0.5 + tanh(u / 4) ("firm", range [-0.5, 1.5], so the
context can invert a recovered association) or for its positive part ("rec", the control with the same amplification
and no sign); both keep a(0) = 0.5 and a'(0) = 0.25 of the sigmoid, so the start and its gradient are unchanged. The tables start at zero, so every site adds zero and the network starts equal to the one
without memory whatever Linear_val holds. With Linear_val at zero too (val_aleatorio=False) the gated sites send no
gradient to the rows until the input site has moved them; val_aleatorio=True starts Linear_val like Linear_key so every
site trains the rows from the first step. The recovered rows are not normalised: with rows at zero an RMSNorm would
amplify noise.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import qf2_memhost_cfg as cfgm

ESCALA_INICIAL = 0.1
PUERTA_CABEZA = os.environ.get("QF2_MEMHOST_PUERTA_CABEZA", "1") != "0"


class Tablas(nn.Module):
    def __init__(self, cfg: cfgm.Cfg, vocab: int):
        super().__init__()
        if cfg.canon:
            self.register_buffer("canon", torch.arange(vocab, dtype=torch.int32))


class Sitios(nn.Module):
    def __init__(self, cfg: cfgm.Cfg, n_embd: int, linear):
        super().__init__()
        self.cfg, self.n_embd = cfg, n_embd
        with torch.random.fork_rng(devices=[]):
            if cfg.entrada:
                self.proy_in = linear(cfg.ancho_e, n_embd, bias=False)
                self.escala = nn.Parameter(torch.full((1,), ESCALA_INICIAL))
            self.key = nn.ModuleDict({str(i): linear(cfg.ancho_e, n_embd, bias=False) for i in cfg.gated})
            self.val = nn.ModuleDict({str(i): linear(cfg.ancho_e, n_embd, bias=False) for i in cfg.gated})

    @torch.no_grad()
    def iniciar(self, val_aleatorio: bool = False) -> None:
        s = math.sqrt(3.0) * self.cfg.ancho_e ** -0.5
        with torch.random.fork_rng(devices=[]):
            if self.cfg.entrada:
                torch.nn.init.uniform_(self.proy_in.weight, -s, s)
            for k in self.key.values():
                torch.nn.init.uniform_(k.weight, -s, s)
            for v in self.val.values():
                if val_aleatorio:
                    torch.nn.init.uniform_(v.weight, -s, s)
                else:
                    torch.nn.init.zeros_(v.weight)

    def matrices(self) -> list:
        out = [self.proy_in.weight] if self.cfg.entrada else []
        return out + [m.weight for d in (self.key, self.val) for m in d.values()]

    def escalares(self) -> list:
        return [self.escala] if self.cfg.entrada else []

    def entrada(self, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + self.escala.to(x.dtype) * self.proy_in(e.to(x.dtype))

    def puerta(self, u: torch.Tensor) -> torch.Tensor:
        if self.cfg.puerta == "sig":
            return torch.sigmoid(u)
        a = 0.5 + torch.tanh(u * 0.25)
        if self.cfg.puerta == "rec":
            a = a.clamp_min(0.0)
        return a

    def delta(self, i: int, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        e = e.to(x.dtype)
        d = x.size(-1)
        k = F.rms_norm(self.key[str(i)](e), (d,))
        q = F.rms_norm(x, (d,))
        if not PUERTA_CABEZA or d % self.cfg.s:
            return self.puerta((q * k).sum(dim=-1, keepdim=True) / math.sqrt(d)) * self.val[str(i)](e)
        s = self.cfg.s
        u = (q * k).unflatten(-1, (s, d // s)).sum(dim=-1) / math.sqrt(d // s)
        a = self.puerta(u).unsqueeze(-1).expand(*u.shape, self.cfg.ancho).flatten(-2)
        return self.val[str(i)](e * a)

    def compuerta(self, i: int, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(i, e, x)

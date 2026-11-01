"""Compiled optimizer update (QF2_OPT_COMPILADO=1; default OFF).

The Neuron eager optimizer launches every elementwise op of every AdamW parameter and of every Muon shape on its own: with
22 layers that is ~570 launches for the 44 lambdas alone, once per step. This module traces the same math once per call
signature with torch.compile and replays it. The math duplicates DistMuonAdamW._muon_owned and MuonAdamW._adamw_update
(QF2_OPT unset) op for op, so the default eager path stays untouched; tools/contrato_opt_compilado.py checks the copy
bit-exact on CPU. Values that change per step (lr, wd, bias corrections) travel in one fp32 device vector per call, so the
schedule never retraces. Backend: neuron on the chip, aot_eager elsewhere; QF2_OPT_COMPILADO_BACKEND overrides.
"""
from __future__ import annotations

import os

import torch


def adamw_lista(ps, grads, exp_avgs, exp_avg_sqs, esc):
    lr_t, wd_t, omb1_t, omb2_t, eps_t, bias1_t, bias2_t = (esc[i] for i in range(7))
    for p, grad, exp_avg, exp_avg_sq in zip(ps, grads, exp_avgs, exp_avg_sqs):
        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, omb1_t)
        exp_avg_sq.lerp_(grad.square(), omb2_t)
        update = exp_avg / bias1_t / ((exp_avg_sq / bias2_t).sqrt() + eps_t)
        p.sub_(lr_t * update)


def muon_nucleo(g, w, momentum, second, esc, coefs: tuple, bf16: bool):
    momentum_t, one_minus_momentum_t, one_minus_beta2_t, lr_t, wd_t = (esc[i] for i in range(5))
    shape = tuple(w.shape[1:])
    momentum.lerp_(g, one_minus_momentum_t)
    g = g.lerp_(momentum, momentum_t)
    x = g.bfloat16() if bf16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    squeezed = x.dim() == 3 and x.size(0) == 1
    if squeezed:
        x = x.squeeze(0)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in coefs:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in coefs:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    if squeezed:
        x = x.unsqueeze(0)
    g = x
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second.lerp_(v_mean, one_minus_beta2_t)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = (g * w) >= 0
    w.sub_(lr_t * g + lr_t * wd_t * w * mask)


class Compilado:
    def __init__(self, device: torch.device):
        self.backend = os.environ.get("QF2_OPT_COMPILADO_BACKEND", "").strip() or (
            "neuron" if device.type == "neuron" else "aot_eager")
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.adamw = torch.compile(adamw_lista, backend=self.backend, dynamic=False, fullgraph=True)
        self.muon = torch.compile(muon_nucleo, backend=self.backend, dynamic=False, fullgraph=True)
        self.vectores: dict = {}

    def escalares(self, clave, valores: list[float], device: torch.device) -> torch.Tensor:
        v = self.vectores.get(clave)
        if v is None or v.device != device:
            v = torch.zeros(len(valores), dtype=torch.float32, device=device)
            self.vectores[clave] = v
        v.copy_(torch.tensor(valores, dtype=torch.float32))
        return v

    @staticmethod
    def estado_adamw(p: torch.Tensor, state: dict, beta1: float, beta2: float) -> tuple[float, float]:
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["bias1_t"] = torch.tensor(0.0, dtype=torch.float32)
            state["bias2_t"] = torch.tensor(0.0, dtype=torch.float32)
        state["step"] += 1
        state["bias1_t"].fill_(1 - beta1 ** state["step"])
        state["bias2_t"].fill_(1 - beta2 ** state["step"])
        return float(state["bias1_t"]), float(state["bias2_t"])

    def adamw_paso(self, ps, grads, states, group, clave) -> None:
        beta1, beta2 = group["betas"]
        sesgos = {self.estado_adamw(p, s, beta1, beta2) for p, s in zip(ps, states)}
        if len(sesgos) != 1:
            raise RuntimeError(f"QF2_OPT_COMPILADO: AdamW group {clave} has parameters at different steps {sorted(sesgos)}")
        bias1, bias2 = sesgos.pop()
        esc = self.escalares(clave, [group["lr"], group["weight_decay"], 1 - beta1, 1 - beta2, group["eps"], bias1, bias2],
                             ps[0].device)
        self.adamw(list(ps), list(grads), [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states], esc)

    def muon_paso(self, group, shape, g, w, momentum, second, coefs, bf16: bool) -> None:
        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        esc = self.escalares(("muon", id(group), tuple(shape), g.shape[0]),
                             [group["momentum"], 1 - group["momentum"], 1 - group["beta2"], lr, group["weight_decay"]], g.device)
        self.muon(g, w, momentum, second, esc, tuple(tuple(c) for c in coefs), bf16)

    def descripcion(self) -> str:
        return f"qf2_opt_compilado: backend={self.backend} cache_size_limit={torch._dynamo.config.cache_size_limit}"

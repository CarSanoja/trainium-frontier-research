from __future__ import annotations

import torch

from .fp8_mlp_comun import MLP_RECORTE_GRAD, _FEP, _H_CLAMP, _SCALE_H

E4_MAX, E5_MAX = 240.0, 57344.0


def e4(v: torch.Tensor) -> torch.Tensor:
    return v.float().clamp(-E4_MAX, E4_MAX).to(torch.float8_e4m3fn).float()


def e5(v: torch.Tensor) -> torch.Tensor:
    return v.float().clamp(-E5_MAX, E5_MAX).to(torch.float8_e5m2).float()


def bf(v: torch.Tensor) -> torch.Tensor:
    return v.float().to(torch.bfloat16).float()


def fila(sc: torch.Tensor) -> list[torch.Tensor]:
    return list(sc[0].float().unbind(0))


def fwd_emulado(x, w1, w2, sc):
    assert _FEP == 1
    s1, s2, c_y, c_h = fila(sc)
    w1s = e4(w1.float() * s1)
    w2s = e4(w2.float() * s2)
    ps = e4(x) @ w1s
    tope = s1 * (_H_CLAMP / _SCALE_H)
    hc = bf(torch.minimum(torch.relu(ps), tope))
    hs = e4((hc * (c_h * c_h)) * hc)
    y = bf((hs @ w2s) * c_y)
    return y.to(x.dtype), hc.t().contiguous().to(x.dtype)


def bwd_act_emulado(dy, w2T, w1T, rT, sc):
    s2, s1, c_g, c_x, S = fila(sc)
    w2s = e4(w2T.float() * s2)
    w1s = e4(w1T.float() * s1)
    ese = (c_x * S) * _SCALE_H
    kgs = (c_g * S) * ese
    dys = e5(dy.float() * S)
    rt = rT.float().t()
    rtm = rt * (rt < s1 * (_H_CLAMP / _SCALE_H)) if MLP_RECORTE_GRAD else rt
    dps = e5((dys @ w2s) * kgs * rtm)
    h8 = e4((rt * (ese * ese)) * rt)
    dx = bf((dps @ w1s) * c_x)
    return dps, h8, dx.to(dy.dtype)


def dw8_x_emulado(x, dpk, sd):
    bs = e4(x.float().clamp(-E4_MAX, E4_MAX))
    return bf((dpk.t() @ bs) * sd[0, 1].float()).to(x.dtype)


def dw8_y_emulado(dy, hpk, sd):
    bs = e5(dy.float() * sd[0, 0].float())
    return bf((bs.t() @ hpk) * sd[0, 2].float()).to(dy.dtype)

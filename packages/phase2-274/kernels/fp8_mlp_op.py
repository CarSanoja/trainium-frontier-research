from __future__ import annotations

import torch

from .fp8_mlp_bwd_act import mlp_fp8_bwd_act
from .fp8_mlp_comun import (HAY_NKI, MLP_DW8, MLP_FP8_BWDK, MLP_FP8_EAGER, MLP_FP8_LNC, MLP_RECORTE_GRAD, RESUMEN, _FEP, _H_CLAMP,
                            _P8, _S8G, _SCALE_H, _nki, formas_ok)
from .fp8_mlp_dw import mlp_fp8_dw8_x, mlp_fp8_dw8_y
from .fp8_mlp_emula import bwd_act_emulado, dw8_x_emulado, dw8_y_emulado, fwd_emulado
from .fp8_mlp_fwd import mlp_fp8_fwd_r


def _pow2_scale(w):
    return torch.exp2(torch.floor(7.0 - torch.log2(w.detach().abs().float().amax().clamp_min(1e-12))))


def fwd_scales(w1, w2):
    s1, s2 = _pow2_scale(w1), _pow2_scale(w2)
    row = torch.stack((s1, s2, 1.0 / (s2 * _SCALE_H * _SCALE_H), _SCALE_H / s1)).view(1, 4)
    return row.expand(_P8, 4).contiguous().to(w1.device), s1, s2


def bwd_scales(s1, s2):
    if _S8G:
        row = torch.stack((s2, s1, 2.0 / (_SCALE_H * s2 * _S8G), 1.0 / (s1 * _S8G), torch.full_like(s1, _S8G))).view(1, 5)
    else:
        row = torch.stack((s2, s1, 2.0 / (_SCALE_H * s2), 1.0 / s1)).view(1, 4)
    return row.expand(_P8, row.shape[1]).contiguous()


def mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1):
    r = rT.t() * (1.0 / s1) if _FEP else rT.t() * (1.0 / _SCALE_H)
    h = r * r
    rm = r * (r < _H_CLAMP / _SCALE_H) if MLP_RECORTE_GRAD else r
    dpre = (dy @ w2o) * (2.0 * rm)
    dx = dpre @ w1o
    dw1o = dpre.t() @ x2
    dw2o = dy.t() @ h
    return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)


def dw8_scales(device):
    row = torch.tensor([_S8G, 1.0 / _S8G, 1.0 / (_S8G * _SCALE_H * _SCALE_H)], dtype=torch.float32).view(1, 3)
    return row.expand(_P8, 3).contiguous().to(device)


def mlp_fp8_backward_kernel(bwd, x2, w1o, w2o, rT, dy, s1, dwk=None):
    s2 = _pow2_scale(w2o)
    if dwk is not None:
        dpk, hpk, dx = bwd(dy, w2o, w1o, rT, bwd_scales(s1, s2).to(dy.device))
        sd = dw8_scales(dy.device)
        dw1o, dw2o = dwk[0](x2, dpk, sd), dwk[1](dy, hpk, sd)
        return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)
    dpreT, dx = bwd(dy, w2o, w1o, rT, bwd_scales(s1, s2).to(dy.device))
    rTs = rT * (1.0 / s1) if _FEP else rT * (1.0 / _SCALE_H)
    dw1o = dpreT @ x2
    dw2o = dy.t() @ (rTs * rTs).t()
    return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)


def build_op(lnc=1, simulate=False, bwd_kernel=False, eager=False):
    if simulate:
        fwd = _nki.simulate(mlp_fp8_fwd_r)
        bwd = _nki.simulate(mlp_fp8_bwd_act) if bwd_kernel else None
        dwk = (_nki.simulate(mlp_fp8_dw8_x), _nki.simulate(mlp_fp8_dw8_y)) if bwd_kernel and MLP_DW8 else None

        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x2, w1o, w2o):
                sc, s1, s2 = fwd_scales(w1o, w2o)
                y, rT = fwd(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)
                ctx.save_for_backward(x2, w1o, w2o, rT, s1)
                return y

            @staticmethod
            def backward(ctx, dy):
                if bwd is not None:
                    return mlp_fp8_backward_kernel(bwd, *ctx.saved_tensors[:4], dy, ctx.saved_tensors[4], dwk)
                return mlp_fp8_backward_math(*ctx.saved_tensors[:4], dy, ctx.saved_tensors[4])

        return _F.apply
    from torch_neuronx import nki_op, wrap_nki
    k = wrap_nki(mlp_fp8_fwd_r)[lnc] if lnc > 1 else wrap_nki(mlp_fp8_fwd_r)
    kb = (wrap_nki(mlp_fp8_bwd_act)[lnc] if lnc > 1 else wrap_nki(mlp_fp8_bwd_act)) if bwd_kernel else None
    kd = tuple(wrap_nki(f)[lnc] if lnc > 1 else wrap_nki(f) for f in (mlp_fp8_dw8_x, mlp_fp8_dw8_y)) if bwd_kernel and MLP_DW8 else None
    if eager:
        class _FE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x2, w1o, w2o):
                sc, s1, s2 = fwd_scales(w1o, w2o)
                y, rT = k(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)
                ctx.save_for_backward(x2, w1o, w2o, rT, s1)
                return y

            @staticmethod
            def backward(ctx, dy):
                x2, w1o, w2o, rT, s1 = ctx.saved_tensors
                if kb is not None:
                    return mlp_fp8_backward_kernel(kb, x2, w1o, w2o, rT, dy, s1, kd)
                return mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1)

        return _FE.apply

    @nki_op("arch::mlp_fp8_fwd", mutates_args={})
    def _fwd_op(x2: torch.Tensor, w1o: torch.Tensor, w2o: torch.Tensor, sc: torch.Tensor, s1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return k(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)

    def _setup(ctx, inputs, output):
        x2, w1o, w2o, sc, s1 = inputs
        ctx.save_for_backward(x2, w1o, w2o, output[1], s1)
        ctx.mark_non_differentiable(output[1])

    def _backward(ctx, dy, drT):
        x2, w1o, w2o, rT, s1 = ctx.saved_tensors
        if kb is not None:
            dx, dw1o, dw2o = mlp_fp8_backward_kernel(kb, x2, w1o, w2o, rT, dy, s1, kd)
        else:
            dx, dw1o, dw2o = mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1)
        return dx, dw1o, dw2o, None, None

    _fwd_op.register_autograd(_backward, setup_context=_setup)

    def f(x2, w1o, w2o):
        sc, s1, s2 = fwd_scales(w1o, w2o)
        return _fwd_op(x2, w1o, w2o, sc, s1)[0]
    return f


_MLP_FP8_OP = None


def _mlp_fp8_impl(x2, w1o, w2o):
    global _MLP_FP8_OP
    if _MLP_FP8_OP is None:
        _MLP_FP8_OP = build_op(lnc=MLP_FP8_LNC, bwd_kernel=MLP_FP8_BWDK, eager=MLP_FP8_EAGER)
    return _MLP_FP8_OP(x2, w1o, w2o)

_mlp_fp8 = torch._dynamo.disable(_mlp_fp8_impl) if MLP_FP8_EAGER else _mlp_fp8_impl


class _MlpFp8Emulado(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, w1o, w2o):
        sc, s1, s2 = fwd_scales(w1o, w2o)
        y, rT = fwd_emulado(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)
        ctx.save_for_backward(x2, w1o, w2o, rT, s1)
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w1o, w2o, rT, s1 = ctx.saved_tensors
        return mlp_fp8_backward_kernel(bwd_act_emulado, x2, w1o, w2o, rT, dy.contiguous(), s1, (dw8_x_emulado, dw8_y_emulado))


def mlp_fp8(x: torch.Tensor, w1o: torch.Tensor, w2o: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w1o, w2o = w1o.to(dtype=x.dtype), w2o.to(dtype=x.dtype)
    if x.device.type == "neuron":
        if not HAY_NKI:
            raise RuntimeError("QF2_FP8_MLP=1 en neuron exige nki importable")
        if not formas_ok(x2.shape[0], x2.shape[1], w1o.shape[0]):
            raise ValueError(f"QF2_FP8_MLP=1: formas M={x2.shape[0]} D={x2.shape[1]} H={w1o.shape[0]} fuera de los kernels")
        return _mlp_fp8(x2, w1o, w2o).view(x.shape)
    if x.device.type == "cpu":
        return _MlpFp8Emulado.apply(x2, w1o, w2o).view(x.shape)
    raise RuntimeError(f"QF2_FP8_MLP=1 no tiene camino en {x.device.type}")

from __future__ import annotations

import torch
import torch.nn.functional as F

from cefa_nucleo import _CF_CAP, _CF_LNC, _CF_P, _CF_TW, cef_op


def _cef_fwd_ref(z, almacen):
    t = torch.tanh(z.float() * (1.0 / _CF_CAP)).to(almacen)
    return t, torch.log(torch.exp(_CF_CAP * t.float()).sum(-1, keepdim=True))


def _cef_bwd_ref(t, nlse, g, almacen):
    tf = t.float()
    return (((tf * -g) * tf + g) * torch.exp(_CF_CAP * tf + nlse)).to(almacen)


class _CEFRef(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, almacen):
        t, lse = _cef_fwd_ref(z, almacen)
        ctx.save_for_backward(t, lse)
        ctx.zdtype = z.dtype
        ctx.almacen = almacen
        return lse.squeeze(1)

    @staticmethod
    def backward(ctx, g):
        t, lse = ctx.saved_tensors
        return _cef_bwd_ref(t, -lse, g.float().reshape(-1, 1), ctx.almacen).to(ctx.zdtype), None


def cef_lse(z, almacen=None):
    if z.device.type == "cpu":
        return _CEFRef.apply(z, z.dtype if almacen is None else almacen)
    op = cef_op()
    assert op is not None, "QF2_CEFA=1 en el dispositivo necesita nki y torch_neuronx"
    assert z.shape[0] % (_CF_P * _CF_LNC) == 0 and z.shape[1] % _CF_TW == 0, z.shape
    return op(z.contiguous())[1].squeeze(1)


def cefa_perdida(model, x, targets, cap, almacen=None):
    assert cap == _CF_CAP, (cap, _CF_CAP)
    _v = model.config.vocab_size
    _w = model.lm_head.weight
    _xf = x.reshape(-1, x.size(-1))
    _lse = cef_lse(F.linear(_xf, _w[:_v].to(dtype=_xf.dtype), None), almacen)
    _t = targets.reshape(-1)
    _valid = (_t >= 0).float()
    _ids = [_t.clamp_min(0)]
    _zt = torch.stack([(_xf * F.embedding(_i, _w).to(dtype=_xf.dtype)).float().sum(-1) for _i in _ids], dim=1)
    _nll = _lse.unsqueeze(1) - cap * torch.tanh(_zt / cap)
    return (_nll[:, 0] * _valid).sum() / _valid.sum()


def cefa_descripcion() -> str:
    op = "arch::cef_fwd + arch::cef_bwd" if cef_op() is not None else "emulacion torch (sin nki)"
    return f"cefa: QF2_CEFA=1 op={op} lnc={_CF_LNC} tw={_CF_TW} p={_CF_P} cap={_CF_CAP}"

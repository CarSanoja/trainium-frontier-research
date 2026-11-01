from __future__ import annotations

import os

import torch

_CF_P, _CF_TW, _CF_CAP = 128, 2048, 15.0
_CF_LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))

_CEF_OP = None
_CEF_BWD_OP = None

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    _HAS_NKI = True
except Exception:
    _HAS_NKI = False

if _HAS_NKI:

    @nki.jit
    def cef_fwd_kernel(z):
        n, v = z.shape
        t = nl.ndarray((n, v), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        lse = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        grid_ndim = nl.program_ndim()
        if grid_ndim != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        nb = (n // _CF_P) // nprg
        nv = v // _CF_TW
        for b0 in nl.affine_range(nb):
            r = (pid * nb + b0) * _CF_P
            parts = nl.ndarray((_CF_P, nv), dtype=nl.float32, buffer=nl.sbuf)
            for j in nl.static_range(nv):
                zt = nl.load(z[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW])
                tt = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=tt, op=nl.tanh, data=zt, scale=1.0 / _CF_CAP)
                nl.store(t[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW], tt)
                ex = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation_reduce(dst=ex, op=nl.exp, data=tt, reduce_op=nl.add, reduce_res=parts[:, j:j + 1], scale=_CF_CAP)
            s = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s, op=nl.add, data=parts, axis=1)
            ls = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ls, op=nl.log, data=s)
            nl.store(lse[r:r + _CF_P, 0:1], ls)
        return t, lse

    @nki.jit
    def cef_bwd_kernel(t, nlse, g):
        n, v = t.shape
        dz = nl.ndarray((n, v), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        grid_ndim = nl.program_ndim()
        if grid_ndim != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        nb = (n // _CF_P) // nprg
        nv = v // _CF_TW
        for b0 in nl.affine_range(nb):
            r = (pid * nb + b0) * _CF_P
            bl = nl.load(nlse[r:r + _CF_P, 0:1])
            gg = nl.load(g[r:r + _CF_P, 0:1])
            ng = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ng, data=gg, op0=nl.multiply, operand0=-1.0)
            for j in nl.static_range(nv):
                tt = nl.load(t[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW])
                p = nl.ndarray((_CF_P, _CF_TW), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p, op=nl.exp, data=tt, bias=bl, scale=_CF_CAP)
                w = nl.ndarray((_CF_P, _CF_TW), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=w, data=tt, op0=nl.multiply, operand0=ng, op1=nl.multiply, operand1=tt)
                d = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=d, data=w, op0=nl.add, operand0=gg, op1=nl.multiply, operand1=p)
                nl.store(dz[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW], d)
        return dz

    try:
        from torch_neuronx import nki_op as _cef_deco, wrap_nki as _cef_wrap
        _CEF_FWD = _cef_wrap(cef_fwd_kernel)[_CF_LNC] if _CF_LNC > 1 else _cef_wrap(cef_fwd_kernel)
        _CEF_BWD = _cef_wrap(cef_bwd_kernel)[_CF_LNC] if _CF_LNC > 1 else _cef_wrap(cef_bwd_kernel)

        @_cef_deco("arch::cef_fwd", mutates_args={})
        def _cef_fwd_op(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return _CEF_FWD(z)

        @_cef_deco("arch::cef_bwd", mutates_args={})
        def _cef_bwd_op(t: torch.Tensor, nlse: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return _CEF_BWD(t, nlse, g)

        def _cef_setup(ctx, inputs, output):
            ctx.save_for_backward(output[0], output[1])
            ctx.mark_non_differentiable(output[0])

        def _cef_backward(ctx, dt, dlse):
            t, lse = ctx.saved_tensors
            return _cef_bwd_op(t, (-lse).contiguous(), dlse.float().contiguous())

        _cef_fwd_op.register_autograd(_cef_backward, setup_context=_cef_setup)
        _CEF_OP = _cef_fwd_op
        _CEF_BWD_OP = _cef_bwd_op
    except ImportError:
        _CEF_OP = None


def cef_op():
    return _CEF_OP

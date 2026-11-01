"""
Neuron Competition Round 2 baseline training script.

Participants edit this file. It is intentionally self-contained and readable.
The fixed data/tokenizer/eval utilities live in prepare.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

# -----------------------------------------------------------------------------
# Participant-owned modules, flattened into this file.
# The submission form accepts only train.py and inference.py, so the modules this
# package ships next to train.py are embedded below as plain source text, byte for
# byte. _qf2_instalar writes them once to <tempdir>/<sha256 of their contents>/
# (fallbacks: ~/.cache, then this file's directory) with mkdtemp + atomic rename,
# re-reads every byte, and prepends that directory to sys.path. The ordinary import
# statements further down then load them as regular files, so tracebacks,
# inspect.getsource and @nki.jit tracing see real source. No network, no
# exec/eval, nothing in the forward pass; the training code below is unchanged.

_QF2_MODULOS = {
    "ce_trozos.py": r'''from __future__ import annotations

import torch


def _trozo(x: torch.Tensor, wc: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = (x @ wc.t()).float()
    t = torch.tanh(z / cap)
    return t, cap * t


class _CETrozos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo):
        n = x.size(0)
        vale = y != -1
        lses = []
        total = x.new_zeros((), dtype=torch.float32)
        wc = w[:vocab].to(x.dtype)
        for i in range(0, n, trozo):
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            zt = zc.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            total = total + torch.where(vale[i:i + trozo], lse - zt, 0.0).sum()
            lses.append(lse)
        lse = torch.cat(lses)
        ctx.save_for_backward(x, w, y, lse)
        ctx.cap, ctx.vocab, ctx.trozo = cap, vocab, trozo
        return total / vale.sum()

    @staticmethod
    def backward(ctx, g):
        x, w, y, lse = ctx.saved_tensors
        cap, vocab, trozo = ctx.cap, ctx.vocab, ctx.trozo
        vale = (y != -1)
        escala = g.float() / vale.sum()
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc, vc = x[i:i + trozo], y[i:i + trozo], vale[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d.scatter_add_(1, yc.clamp_min(0).unsqueeze(1), -torch.ones_like(d[:, :1]))
            d = d * (escala * vc.float()).unsqueeze(1) * (1.0 - t * t)
            dz = d.to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None


def ce_trozos(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int) -> torch.Tensor:
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozos.apply(x, w, targets, cap, vocab, trozo)


class _CETrozosTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo):
        vale = y != -1
        lses, nlls = [], []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            zt = zc.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            nlls.append(torch.where(vale[i:i + trozo], lse - zt, 0.0))
            lses.append(lse)
        ctx.save_for_backward(x, w, y, torch.cat(lses))
        ctx.cap, ctx.vocab, ctx.trozo = cap, vocab, trozo
        return torch.cat(nlls)

    @staticmethod
    def backward(ctx, g):
        x, w, y, lse = ctx.saved_tensors
        cap, vocab, trozo = ctx.cap, ctx.vocab, ctx.trozo
        coef = g.float() * (y != -1).float()
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc = x[i:i + trozo], y[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d.scatter_add_(1, yc.clamp_min(0).unsqueeze(1), -torch.ones_like(d[:, :1]))
            d = d * coef[i:i + trozo].unsqueeze(1) * (1.0 - t * t)
            dz = d.to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None


def ce_trozos_tokens(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int) -> torch.Tensor:
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozosTokens.apply(x, w, targets, cap, vocab, trozo)
''',
    "cefa_nucleo.py": r'''from __future__ import annotations

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
''',
    "cefa_perdida.py": r'''from __future__ import annotations

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
''',
    "kernels/__init__.py": r'''''',
    "kernels/emb_dw.py": r'''"""EMB: NKI 0.5.0 kernel dW fp32 [rows + 128, D] = scatter-add of dy [n, D] over ids [n] (port of the F1 EMBKF kernel).

dW is zeroed per program, then the ids are walked in tiles of 128 (nl.sequential_range: tiles RMW the same rows). Inside a tile the
duplicates are summed on the PE (equality matrix x dy into PSUM fp32); only the first occurrence of an id keeps its real row and each
repeat goes to its own in-range garbage row rows + m (m = its row inside the tile), so every indirect RMW instruction (nisa.dma_compute,
fp32 add) sees 128 unique in-range indices. oob_mode=skip aborted on the chip (NRT 1203) and is not used. Grid [lnc]: program p writes
only its D/lnc columns, in chunks of at most 512 fp32 columns (one PSUM bank). The caller slices [:rows].
"""
from __future__ import annotations


class _SinNki:
    @staticmethod
    def jit(f):
        return f


try:
    import nki as _nki
    import nki.isa as nisa
    import nki.language as nl
    HAY_NKI = True
except Exception:
    _nki, nl, nisa = _SinNki(), None, None
    HAY_NKI = False

EMB_P = 128


@_nki.jit
def emb_dw_kernel(dy, idx, rows):
    n, d = dy.shape
    pid = nl.program_id(0)
    dh = d // nl.num_programs(0)
    c0 = pid * dh
    nch = (dh + 511) // 512
    cw = dh // nch
    tot = rows + EMB_P
    dw = nl.ndarray((tot, d), dtype=nl.float32, buffer=nl.shared_hbm)
    zero = nl.ndarray((EMB_P, dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    nfull = tot // EMB_P
    for r in nl.affine_range(nfull):
        nisa.dma_copy(dst=dw[r * EMB_P:(r + 1) * EMB_P, c0:c0 + dh], src=zero)
    if tot > nfull * EMB_P:
        nisa.dma_copy(dst=dw[nfull * EMB_P:tot, c0:c0 + dh], src=zero[0:tot - nfull * EMB_P, :])
    pos = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pos, pattern=[[1, EMB_P]], offset=0, channel_multiplier=-1)
    low = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low, data=pos, op0=nl.less, operand0=0.0)
    slot = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(slot, pattern=[[0, 1]], offset=rows, channel_multiplier=1)
    for t in nl.sequential_range(n // EMB_P):
        idc = nl.ndarray((EMB_P, 1), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idc, src=idx.ap(pattern=[[1, EMB_P], [1, 1]], offset=t * EMB_P))
        idr = nl.ndarray((EMB_P, EMB_P), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idr, src=idx.ap(pattern=[[0, EMB_P], [1, EMB_P]], offset=t * EMB_P))
        idcf = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idcf, src=idc)
        idrf = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idrf, src=idr)
        eq = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=eq, data=idrf, op0=nl.equal, operand0=idcf)
        prev = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prev, data1=eq, data2=low, op=nl.multiply)
        cnt = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cnt, op=nl.add, data=prev, axis=1)
        fol = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fol, data=cnt, op0=nl.minimum, operand0=1.0)
        gap = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gap, data1=slot, data2=idcf, op=nl.subtract)
        pen = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pen, data1=fol, data2=gap, op=nl.multiply)
        efff = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=efff, data1=idcf, data2=pen, op=nl.add)
        eff = nl.ndarray((EMB_P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eff, src=efff)
        eqm = nl.ndarray((EMB_P, EMB_P), dtype=dy.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eqm, src=eq)
        for c in nl.static_range(nch):
            cc = c0 + c * cw
            dyt = nl.ndarray((EMB_P, cw), dtype=dy.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=dyt, src=dy[t * EMB_P:(t + 1) * EMB_P, cc:cc + cw])
            ps = nl.ndarray((EMB_P, cw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=eqm, moving=dyt)
            acc = nl.ndarray((EMB_P, cw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=acc, src=ps)
            tgt = dw.ap(pattern=[[d, EMB_P], [1, cw]], offset=cc, vector_offset=eff, indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, acc], reduce_op=nl.add)
    return dw
''',
    "kernels/emb_op.py": r'''"""EMB: embedding tables through an autograd Function. Forward is F.embedding cast to the compute dtype (the same ops as
nn.Embedding(idx).to(dtype), bit for bit). Backward (QF2_EMB_SCATTER=1) is dW fp32 by scatter-add: the NKI op arch::emb_dw on the
neuron device (no fallback: it raises if nki/torch_neuronx or the shapes are missing), index_add_ fp32 elsewhere; it returns in the
parameter dtype. QF2_EMB_DETACH=1 is TIMING ONLY: the tables get a dense all-zero gradient of their shape and dtype.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .emb_dw import EMB_P, HAY_NKI, emb_dw_kernel

EMB_LNC = int(os.environ.get("QF2_EMB_SCATTER_LNC", "2"))
EMB_OP_NOMBRE = "arch::emb_dw"
_OP = None


def formas_ok(n: int, d: int, lnc: int = EMB_LNC) -> bool:
    dh = d // lnc
    nch = (dh + 511) // 512
    return n > 0 and n % EMB_P == 0 and d % lnc == 0 and dh % nch == 0 and dh // nch <= 512


def emb_preparar():
    global _OP
    if _OP is not None:
        return _OP
    if not HAY_NKI:
        raise RuntimeError("QF2_EMB_SCATTER=1 en neuron exige nki importable")
    from torch_neuronx import nki_op, wrap_nki
    run = wrap_nki(emb_dw_kernel)[EMB_LNC] if EMB_LNC > 1 else wrap_nki(emb_dw_kernel)

    @nki_op(EMB_OP_NOMBRE, mutates_args={})
    def _emb_dw_op(dy: torch.Tensor, idx: torch.Tensor, rows: int) -> torch.Tensor:
        return run(dy, idx, rows)

    _OP = _emb_dw_op
    return _OP


def emb_dw(fi: torch.Tensor, fg: torch.Tensor, rows: int) -> torch.Tensor:
    if fg.device.type != "neuron":
        return torch.zeros((rows, fg.shape[1]), dtype=torch.float32, device=fg.device).index_add_(0, fi.long(), fg.float())
    if not formas_ok(fi.shape[0], fg.shape[1]) or fg.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"QF2_EMB_SCATTER=1: n={fi.shape[0]} D={fg.shape[1]} dtype={fg.dtype} lnc={EMB_LNC} fuera del kernel")
    return emb_preparar()(fg.contiguous(), fi.to(torch.int32).contiguous(), rows)[:rows]


class _EmbScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, weight, dtype):
        ctx.save_for_backward(idx)
        ctx.forma, ctx.wdtype = weight.shape, weight.dtype
        return F.embedding(idx, weight).to(dtype)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        dw = emb_dw(idx.reshape(-1), g.reshape(-1, g.shape[-1]), ctx.forma[0])
        return None, dw.to(ctx.wdtype), None


class _EmbDetach(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, weight, dtype):
        ctx.forma, ctx.wdtype = weight.shape, weight.dtype
        return F.embedding(idx, weight).to(dtype)

    @staticmethod
    def backward(ctx, g):
        return None, torch.zeros(ctx.forma, dtype=ctx.wdtype, device=g.device), None


def emb_tabla(idx: torch.Tensor, weight: torch.Tensor, dtype: torch.dtype, detach: bool) -> torch.Tensor:
    return (_EmbDetach if detach else _EmbScatter).apply(idx, weight, dtype)


def emb_descripcion(device: torch.device, detach: bool, tablas: int, forma, dtype: torch.dtype) -> str:
    nombres = f"wte+{tablas - 1}ve" if tablas > 1 else "wte"
    if detach:
        return (f"qf2_emb: !!! QF2_EMB_DETACH=1 SOLO RELOJ, NUNCA PARA UN PAQUETE: las {tablas} tablas ({nombres}) reciben un "
                f"gradiente denso a cero {tuple(forma)}; el modelo NO aprende sus embeddings !!!")
    op = f"{EMB_OP_NOMBRE} lnc={EMB_LNC}" if device.type == "neuron" else "index_add_ fp32"
    return (f"qf2_emb: scatter op={op} tablas={tablas} ({nombres}) forma={tuple(forma)} filas_basura={EMB_P} dy={dtype} "
            f"nki={int(HAY_NKI)} formas_ok_d={int(formas_ok(EMB_P, forma[1]))}")
''',
    "kernels/emb_puerta.py": r'''"""EMB gate (QF2_EMB_SCATTER_GATE=1, rank 0, before the training loop): dW of emb_dw on the device over the ids of the first
micro-batch and over the same ids with the first half of the positions set to the most frequent id (whole tiles of one id), with
dy = 1e-3 * randn (seed 0) in the compute dtype, against index_add_ fp32 on CPU. Rule: finite, max|dW - ref| / max|ref| <= 1e-3 and
||dW - ref||F / ||ref||F <= 1e-4 in both cases. Prints one EMB_GATE line and returns ok.
"""
from __future__ import annotations

import os

import torch

from .emb_op import emb_dw

TOL_MAX, TOL_FROB = 1e-3, 1e-4


def emb_compara(k: torch.Tensor, ref: torch.Tensor) -> dict:
    k64, r64 = k.detach().cpu().double(), ref.detach().cpu().double()
    fin = bool(torch.isfinite(k64).all())
    d = k64 - r64
    maxrel = d.abs().max().item() / max(r64.abs().max().item(), 1e-300)
    frob = d.norm().item() / max(r64.norm().item(), 1e-300)
    return {"fin": fin, "maxrel": maxrel, "frob": frob, "ok": fin and maxrel <= TOL_MAX and frob <= TOL_FROB}


def emb_puerta_entradas(ids: torch.Tensor, d: int, dtype: torch.dtype, semilla: int = 0):
    fi = ids.reshape(-1).detach().cpu().long()
    g = torch.Generator().manual_seed(semilla)
    dy = (torch.randn(fi.numel(), d, generator=g) * 1e-3).to(dtype)
    rep = fi.clone()
    rep[:fi.numel() // 2] = torch.bincount(fi).argmax()
    return fi, rep, dy


def emb_ref(dy: torch.Tensor, idx: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.zeros((rows, dy.shape[1]), dtype=torch.float32).index_add_(0, idx.long(), dy.float())


def _llamada(dy, idx, rows):
    return emb_dw(idx, dy, rows)


def emb_puerta_casos(device: torch.device, ids: torch.Tensor, forma, dtype: torch.dtype, dw=_llamada) -> dict:
    rows, d = forma
    fi, rep, dy = emb_puerta_entradas(ids, d, dtype)
    f = torch.compile(dw, backend="neuron", dynamic=False) if device.type == "neuron" else dw
    r = {"n": fi.numel(), "maxdup": int(torch.bincount(fi).max()), "maxdup_rep": int(torch.bincount(rep).max())}
    peor = {"fin": True, "maxrel": 0.0, "frob": 0.0, "ok": True}
    for nombre, ii in (("real", fi), ("rep", rep)):
        with torch.no_grad():
            k = f(dy.to(device), ii.to(torch.int32).to(device), rows).cpu()
        c = emb_compara(k, emb_ref(dy, ii, rows))
        c["filas"] = k.shape[0]
        c["ok"] = c["ok"] and k.shape[0] == rows
        r[nombre] = c
        peor = {"fin": peor["fin"] and c["fin"], "maxrel": max(peor["maxrel"], c["maxrel"]), "frob": max(peor["frob"], c["frob"]),
                "ok": peor["ok"] and c["ok"]}
    r.update(peor)
    return r


def emb_linea(r: dict) -> str:
    casos = " ".join(f"{k}={v['maxrel']:.3e}/{v['frob']:.3e}" for k, v in r.items() if isinstance(v, dict))
    sr = os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN", "")
    return (f"EMB_GATE ok={r['ok']} maxrel={r['maxrel']:.4e} frob={r['frob']:.4e} n={r['n']} maxdup={r['maxdup']} "
            f"maxdup_rep={r['maxdup_rep']} fin={r['fin']} sr={sr} {casos}")


def emb_puerta(device: torch.device, ids: torch.Tensor, forma, dtype: torch.dtype) -> bool:
    r = emb_puerta_casos(device, ids, forma, dtype)
    print(emb_linea(r), flush=True)
    return bool(r["ok"])
''',
    "kernels/fp8_mlp_bwd_act.py": r'''from .fp8_mlp_comun import (MLP_DW8, MLP_DW8_EVSB, MLP_RECORTE_GRAD, _DR, _EPI, _FEP, _FP8, _FP8G, _H_CLAMP, _KAM, _NT8, _P8,
                            _S8G, _SCALE_H, _TB8, _nki, nisa, nl)


@_nki.jit
def mlp_fp8_bwd_act(dy, w2T, w1T, rT, sc):
    M, D = dy.shape
    _, H = w2T.shape
    dpreT = nl.ndarray((H, M), dtype=dy.dtype, buffer=nl.shared_hbm)
    dx = nl.ndarray((M, D), dtype=dy.dtype, buffer=nl.shared_hbm)
    scs = nl.load(sc)
    if _EPI:
        kgs = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        if _FEP:
            eun = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(eun, scs[:, 3:4], scs[:, 4:5], nl.multiply)
            ese = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(ese, eun, nl.multiply, _SCALE_H)
            kg0 = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(kg0, scs[:, 2:3], scs[:, 4:5], nl.multiply)
            nisa.tensor_tensor(kgs, kg0, ese, nl.multiply)
            qe = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(qe, ese, ese, nl.multiply)
            if MLP_RECORTE_GRAD:
                tpb = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(tpb, scs[:, 1:2], nl.multiply, _H_CLAMP / _SCALE_H)
        else:
            nisa.tensor_tensor(kgs, scs[:, 2:3], scs[:, 4:5], nl.multiply)
    KD = D // _P8
    NCD = KD // 2
    TLD = (KD % 2) * (_P8 // 2)
    KH = H // _P8
    NCH = KH // 2
    ND = (D + _NT8 - 1) // _NT8
    if MLP_DW8:
        dpk = nl.ndarray((KH, _P8, M // (2 * _P8), 2, _P8), dtype=nl.uint8, buffer=nl.shared_hbm)
        hpk = nl.ndarray((KH, _P8, M // (2 * _P8), 2, _P8), dtype=nl.uint8, buffer=nl.shared_hbm)
    w2s = nl.ndarray((_P8, NCD, 2, H), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCD):
        for j in nl.static_range(2):
            t = nl.load(w2T[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w2s[:, c, j, :], t, nl.multiply, scs[:, 0:1])
    if TLD:
        w2t = nl.ndarray((TLD, 2, H), dtype=_FP8, buffer=nl.sbuf)
        for j in nl.static_range(2):
            t = nl.load(w2T[NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD, :])
            nisa.tensor_scalar(w2t[:, j, :], t, nl.multiply, scs[0:TLD, 0:1])
    w1s = nl.ndarray((_P8, NCH, 2, D), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCH):
        for j in nl.static_range(2):
            t = nl.load(w1T[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w1s[:, c, j, :], t, nl.multiply, scs[:, 1:2])
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        nprg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
    else:
        nprg = 1
        pid = 0
    NB = (M // _TB8) // nprg
    for mb0 in nl.affine_range(NB):
        mb = pid * NB + mb0
        dys = nl.ndarray((_P8, NCD, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
        for c in nl.static_range(NCD):
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(dy[mb * _TB8:(mb + 1) * _TB8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                    if _S8G:
                        nisa.tensor_scalar(dys[:, c, j, :], xt, nl.multiply, scs[:, 4:5])
                    else:
                        nisa.tensor_copy(dys[:, c, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(dy[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                        if _S8G:
                            nisa.tensor_scalar(dys[:, c, j, t * _P8:(t + 1) * _P8], xt, nl.multiply, scs[:, 4:5])
                        else:
                            nisa.tensor_copy(dys[:, c, j, t * _P8:(t + 1) * _P8], xt)
        if TLD:
            dytl = nl.ndarray((TLD, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(dy[mb * _TB8:(mb + 1) * _TB8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                    if _S8G:
                        nisa.tensor_scalar(dytl[:, j, :], xt, nl.multiply, scs[0:TLD, 4:5])
                    else:
                        nisa.tensor_copy(dytl[:, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(dy[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                        if _S8G:
                            nisa.tensor_scalar(dytl[:, j, t * _P8:(t + 1) * _P8], xt, nl.multiply, scs[0:TLD, 4:5])
                        else:
                            nisa.tensor_copy(dytl[:, j, t * _P8:(t + 1) * _P8], xt)
        dps = nl.ndarray((_P8, NCH, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
        for c in nl.static_range(NCH):
            for j in nl.static_range(2):
                ht = 2 * c + j
                ps = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.psum)
                for cd in nl.static_range(NCD):
                    nisa.nc_matmul(ps, w2s[:, cd, :, ht * _P8:(ht + 1) * _P8], dys[:, cd, :, :], accumulate=(cd > 0), perf_mode=_DR)
                if TLD:
                    nisa.nc_matmul(ps, w2t[:, :, ht * _P8:(ht + 1) * _P8], dytl, accumulate=True, perf_mode=_DR)
                rt = nl.load(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8])
                if _EPI == 1:
                    rtm = rt
                    if MLP_RECORTE_GRAD:
                        rtm = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.scalar_tensor_tensor(dst=rtm, data=rt, op0=nl.less, operand0=tpb, op1=nl.multiply, operand1=rt)
                    nisa.scalar_tensor_tensor(dst=dps[:, c, j, :], data=ps, op0=nl.multiply, operand0=kgs, op1=nl.multiply, operand1=rtm)
                elif _EPI == 2:
                    g = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(g, ps, rt, nl.multiply)
                    nisa.tensor_scalar(dps[:, c, j, :], g, nl.multiply, kgs)
                else:
                    g = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(g, ps, rt, nl.multiply)
                    gb = nl.ndarray((_P8, _TB8), dtype=dy.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(gb, g, nl.multiply, scs[:, 2:3])
                    if not MLP_DW8:
                        nl.store(dpreT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], gb)
                    if _S8G:
                        nisa.tensor_scalar(dps[:, c, j, :], gb, nl.multiply, scs[:, 4:5])
                    else:
                        nisa.tensor_copy(dps[:, c, j, :], gb)
                if MLP_DW8:
                    h8 = nl.ndarray((_P8, _TB8), dtype=_FP8, buffer=nl.sbuf)
                    if _FEP:
                        nisa.scalar_tensor_tensor(dst=h8, data=rt, op0=nl.multiply, operand0=qe, op1=nl.multiply, operand1=rt)
                    else:
                        nisa.tensor_tensor(h8, rt, rt, nl.multiply)
                    if _KAM:
                        pk = nl.ndarray((_P8, 4, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        src = dps[:, c, j, :]
                        sv = src.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        sh = h8.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, 2 + k, :], sh[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 4, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 4, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(dpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk[:, 0:2, :, :])
                        nl.store(hpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk[:, 2:4, :, :])
                    else:
                        pk = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        src = dps[:, c, j, :]
                        sv = src.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 2, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(dpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk)
                        pk = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        sv = h8.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 2, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(hpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk)
        for t in nl.static_range(_TB8 // _P8):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                ps2 = nl.ndarray((_P8, w), dtype=nl.float32, buffer=nl.psum)
                for c in nl.static_range(NCH):
                    nisa.nc_matmul(ps2, dps[:, c, :, t * _P8:(t + 1) * _P8], w1s[:, c, :, nd * _NT8:nd * _NT8 + w], accumulate=(c > 0), perf_mode=_DR)
                o = nl.ndarray((_P8, w), dtype=dy.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(o, ps2, nl.multiply, scs[:, 3:4])
                nl.store(dx[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, nd * _NT8:nd * _NT8 + w], o)
    if MLP_DW8:
        return dpk, hpk, dx
    return dpreT, dx
''',
    "kernels/fp8_mlp_comun.py": r'''from __future__ import annotations

import os


class _SinNki:
    @staticmethod
    def jit(f):
        return f


try:
    import nki as _nki
    import nki.isa as nisa
    import nki.language as nl
    HAY_NKI = True
except Exception:
    _nki, nl, nisa = _SinNki(), None, None
    HAY_NKI = False

MLP_FP8_LNC = 2
MLP_FP8_BWDK = os.environ.get("QF2_FP8_BWDK", "1") != "0"
MLP_FP8_EAGER = False
MLP_FP8_DYSCALE = int(os.environ.get("QF2_FP8_DYSCALE", "16"))
MLP_DW8 = True
MLP_DW8_EVSB = False
MLP_EPIPASS = 1
MLP_FWDEPI = 1
KA_MLP = 1
MLP_RECORTE_GRAD = os.environ.get("QF2_FP8_RECORTE_GRAD", "1") != "0"

_P8, _NT8, _TB8 = 128, 512, 512
_FP8 = nl.float8_e4m3 if HAY_NKI else None
_FP8G = nl.float8_e5m2 if HAY_NKI else None
_SCALE_H = 0.25
_H_CLAMP = 15.4375
_DR = nisa.matmul_perf_mode.double_row if HAY_NKI else None
_S8G = float(2 ** MLP_FP8_DYSCALE) if MLP_FP8_DYSCALE > 0 else 0.0
_EPI = MLP_EPIPASS if MLP_DW8 and _S8G else 0
_FEP = MLP_FWDEPI if _EPI == 1 else 0
_KAM = KA_MLP == 1
if MLP_RECORTE_GRAD and not (_EPI == 1 and _FEP == 1):
    raise ValueError("MLP_RECORTE_GRAD necesita MLP_EPIPASS=1 y MLP_FWDEPI=1 (el recorte se enmascara en la vuelta de esa rama)")


def formas_ok(M: int, D: int, H: int, rejilla: int = MLP_FP8_LNC) -> bool:
    return (D % _P8 == 0 and H % (2 * _P8) == 0 and M % (_TB8 * rejilla) == 0 and M % (2 * _P8) == 0
            and (H // _P8) % rejilla == 0)


RESUMEN = (f"nki={int(HAY_NKI)} lnc={MLP_FP8_LNC} bwdk={int(MLP_FP8_BWDK)} eager={int(MLP_FP8_EAGER)} dyscale=2^{MLP_FP8_DYSCALE} "
           f"dw8={int(MLP_DW8)} epipass={_EPI} fwdepi={_FEP} ka_mlp={int(_KAM)} tb8={_TB8} h_clamp={_H_CLAMP} "
           f"recorte_grad={int(MLP_RECORTE_GRAD)}")
''',
    "kernels/fp8_mlp_dw.py": r'''from .fp8_mlp_comun import _DR, _FP8, _FP8G, _NT8, _P8, _nki, nisa, nl


def _dw8_fase(a, pk, sc, fase):
    M, D = a.shape
    KH = pk.shape[0]
    H = KH * _P8
    NS = M // (2 * _P8)
    KD = D // _P8
    ND = (D + _NT8 - 1) // _NT8
    out = nl.ndarray((H, D) if fase == 0 else (D, H), dtype=a.dtype, buffer=nl.shared_hbm)
    scs = nl.load(sc)
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        nprg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
    else:
        nprg = 1
        pid = 0
    KHP = KH // nprg
    bs = nl.ndarray((_P8, NS, 2, D), dtype=_FP8 if fase == 0 else _FP8G, buffer=nl.sbuf)
    for s in nl.static_range(NS):
        t = nl.ndarray((_P8, 2, D), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t, a.ap(pattern=[[2 * D, _P8], [D, 2], [1, D]], offset=s * 2 * _P8 * D))
        if fase == 0:
            nisa.tensor_scalar(bs[:, s, :, :], t, nl.minimum, 240.0, op1=nl.maximum, operand1=-240.0)
        else:
            nisa.tensor_scalar(bs[:, s, :, :], t, nl.multiply, scs[:, 0:1])
    for i in nl.affine_range(KHP):
        ht = pid * KHP + i
        st = nl.ndarray((_P8, NS, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(st, pk[ht, :, :, :, :])
        sv = st.view(_FP8G if fase == 0 else _FP8)
        pss = nl.ndarray((_P8, ND, _NT8), dtype=nl.float32, buffer=nl.psum)
        for s in nl.static_range(NS):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                nisa.nc_matmul(pss[:, nd, 0:w], sv[:, s, :, :], bs[:, s, :, nd * _NT8:nd * _NT8 + w], accumulate=(s > 0), perf_mode=_DR)
        o = nl.ndarray((_P8, D), dtype=a.dtype, buffer=nl.sbuf)
        for nd in nl.static_range(ND):
            w = min(_NT8, D - nd * _NT8)
            nisa.tensor_scalar(o[:, nd * _NT8:nd * _NT8 + w], pss[:, nd, 0:w], nl.multiply, scs[:, 1 + fase:2 + fase])
        if fase == 0:
            nl.store(out[ht * _P8:(ht + 1) * _P8, :], o)
        else:
            for kd in nl.static_range(KD):
                pt = nl.ndarray((_P8, _P8), dtype=a.dtype, buffer=nl.psum)
                nisa.nc_transpose(pt, o[:, kd * _P8:(kd + 1) * _P8], engine=nisa.engine.tensor)
                ot = nl.ndarray((_P8, _P8), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(ot, pt)
                nl.store(out[kd * _P8:(kd + 1) * _P8, ht * _P8:(ht + 1) * _P8], ot)
    return out


@_nki.jit
def mlp_fp8_dw8_x(x, dpk, sc):
    return _dw8_fase(x, dpk, sc, 0)


@_nki.jit
def mlp_fp8_dw8_y(dy, hpk, sc):
    return _dw8_fase(dy, hpk, sc, 1)
''',
    "kernels/fp8_mlp_emula.py": r'''from __future__ import annotations

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
''',
    "kernels/fp8_mlp_fwd.py": r'''from .fp8_mlp_comun import _DR, _FEP, _FP8, _H_CLAMP, _KAM, _NT8, _P8, _SCALE_H, _TB8, _nki, nisa, nl


@_nki.jit
def mlp_fp8_fwd_r(x, w1, w2, sc):
    M, D = x.shape
    scs = nl.load(sc)
    if _FEP:
        tope = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(tope, scs[:, 0:1], nl.multiply, _H_CLAMP / _SCALE_H)
        qe = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(qe, scs[:, 3:4], scs[:, 3:4], nl.multiply)
    _, H = w1.shape
    out = nl.ndarray((M, D), dtype=x.dtype, buffer=nl.shared_hbm)
    rT = nl.ndarray((H, M), dtype=x.dtype, buffer=nl.shared_hbm)
    KD = D // _P8
    NCD = KD // 2
    TLD = (KD % 2) * (_P8 // 2)
    KH = H // _P8
    NCH = KH // 2
    ND = (D + _NT8 - 1) // _NT8
    w1s = nl.ndarray((_P8, NCD, 2, H), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCD):
        for j in nl.static_range(2):
            t = nl.load(w1[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w1s[:, c, j, :], t, nl.multiply, scs[:, 0:1])
    if TLD:
        w1t = nl.ndarray((TLD, 2, H), dtype=_FP8, buffer=nl.sbuf)
        for j in nl.static_range(2):
            t = nl.load(w1[NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD, :])
            nisa.tensor_scalar(w1t[:, j, :], t, nl.multiply, scs[0:TLD, 0:1])
    w2s = nl.ndarray((_P8, NCH, 2, D), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCH):
        for j in nl.static_range(2):
            t = nl.load(w2[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w2s[:, c, j, :], t, nl.multiply, scs[:, 1:2])
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        nprg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
    else:
        nprg = 1
        pid = 0
    NB = (M // _TB8) // nprg
    for mb0 in nl.affine_range(NB):
        mb = pid * NB + mb0
        xs = nl.ndarray((_P8, NCD, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
        for c in nl.static_range(NCD):
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(x[mb * _TB8:(mb + 1) * _TB8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                    nisa.tensor_copy(xs[:, c, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(x[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                        nisa.tensor_copy(xs[:, c, j, t * _P8:(t + 1) * _P8], xt)
        if TLD:
            xtl = nl.ndarray((TLD, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(x[mb * _TB8:(mb + 1) * _TB8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                    nisa.tensor_copy(xtl[:, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(x[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                        nisa.tensor_copy(xtl[:, j, t * _P8:(t + 1) * _P8], xt)
        hs = nl.ndarray((_P8, NCH, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
        for c in nl.static_range(NCH):
            for j in nl.static_range(2):
                ht = 2 * c + j
                ps = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.psum)
                for cd in nl.static_range(NCD):
                    nisa.nc_matmul(ps, w1s[:, cd, :, ht * _P8:(ht + 1) * _P8], xs[:, cd, :, :], accumulate=(cd > 0), perf_mode=_DR)
                if TLD:
                    nisa.nc_matmul(ps, w1t[:, :, ht * _P8:(ht + 1) * _P8], xtl, accumulate=True, perf_mode=_DR)
                if _FEP:
                    hc = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if _FEP == 2:
                        nisa.tensor_scalar(hc, ps, nl.maximum, 0.0, op1=nl.minimum, operand1=tope, engine=nisa.engine.scalar)
                    else:
                        nisa.tensor_scalar(hc, ps, nl.maximum, 0.0, op1=nl.minimum, operand1=tope)
                    nisa.scalar_tensor_tensor(dst=hs[:, c, j, :], data=hc, op0=nl.multiply, operand0=qe, op1=nl.multiply, operand1=hc)
                    nl.store(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], hc)
                else:
                    hb = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(hb, nl.relu, ps, scale=scs[:, 3:4])
                    hc = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(hc, hb, nl.minimum, _H_CLAMP)
                    nisa.tensor_tensor(hs[:, c, j, :], hc, hc, nl.multiply)
                    nl.store(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], hc)
        for t in nl.static_range(_TB8 // _P8):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                ps2 = nl.ndarray((_P8, w), dtype=nl.float32, buffer=nl.psum)
                for c in nl.static_range(NCH):
                    nisa.nc_matmul(ps2, hs[:, c, :, t * _P8:(t + 1) * _P8], w2s[:, c, :, nd * _NT8:nd * _NT8 + w], accumulate=(c > 0), perf_mode=_DR)
                o = nl.ndarray((_P8, w), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(o, ps2, nl.multiply, scs[:, 2:3])
                nl.store(out[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, nd * _NT8:nd * _NT8 + w], o)
    return out, rT
''',
    "kernels/fp8_mlp_op.py": r'''from __future__ import annotations

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
''',
    "qf2_agrupa.py": r'''"""QF2_AGRUPA: una fracción fija de las filas de entreno pasa a ser documentos enteros agrupados por parecido (in-context
pretraining, Shi et al. 2023, arXiv 2310.10638): lo que hay antes de un BOS es evidencia útil y no un documento ajeno
(docs/bitacora/20261023T003408Z: el modelo aprendió a ignorarlo). Vacío (por defecto): cargador() devuelve el mismo generador.

'f:pool:modo': envolver() sustituye en cada lote, antes de la precarga, n filas con el acumulador de QF2_RECOLOCADO (fracción
exacta a la larga, mismas filas y tokens); cuáles, por (semilla, rango, paso). La fila (ancho del cargador, x.shape[1] + 1)
son k = (ancho - 1) / T cadenas independientes, T = --seq-len (T tokens, la última T + 1), cada una con su semilla y sus
vecinos: fila[j·T] es BOS y ninguna de las secuencias de T en que la parte qf2_cola.lote ve dos cadenas (ancho - 1 no
múltiplo de T es error; con T = ancho - 1, una cadena). Un hilo lee con qf2_recolocado.documentos_de_shards (respeta
QF2_BANCO_SHARDS_DESDE; solo los rangos de igual rango % shards, que leen los mismos shards, se reparten sus documentos por
turno), tokeniza con BOS delante como el kit y guarda un pool de 'pool' documentos. Cadena: uno del pool al azar y, mientras
quepa, el más parecido al último puesto (relacionados) o, en azar (el control), uno al azar entre los CERCANOS de longitud
más parecida a ese vecino: mismas longitudes, sin relación. Documentos enteros desde su BOS, el último de la cadena recortado
por el final; lo puesto sale del pool y se repone del flujo (ninguno se repite); y = x desplazada. Parecido: coseno entre
bolsas de los PREFIJO primeros tokens sin BOS con hashing a DIMENSIONES cubetas, peso (1 + log tf) * IDF del pool, normas
cada REFRESCO filas; el vecino lee solo las cubetas del último con einsum. Solo texto de los shards de entreno.
"""
from __future__ import annotations

import math
import queue
import random
import threading
import time

import numpy as np
import torch

import qf2_hilos_nice as nice
import qf2_recolocado as rec

BITS, PREFIJO, CERCANOS, REFRESCO, TOKENIZAR_DE_A, PROFUNDIDAD, TRAZA_CADA, ESPERA_S = 12, 1024, 32, 64, 256, 16, 1024, 120.0
DIMENSIONES, MEZCLA, MODOS = 1 << BITS, np.uint64(0x9E3779B1), ("relacionados", "azar")


def leer(texto: str) -> tuple[float, int, str] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    try:
        fraccion, pool = float(partes[0]), int(partes[1])
    except (ValueError, IndexError):
        fraccion, pool = math.nan, 0
    if len(partes) != 3 or not 0 < fraccion <= 1 or pool < 2 * CERCANOS or partes[2] not in MODOS:
        raise ValueError(f"QF2_AGRUPA={texto!r}: se espera 'f:pool:modo' con 0 < f <= 1, pool >= {2 * CERCANOS} y modo en {MODOS}")
    return fraccion, pool, partes[2]


def documentos_de_rango(rank: int, world: int, semilla: int):
    import prepare
    import qf2_inicio
    rutas = prepare.list_shard_paths(qf2_inicio.shards_banco() or prepare.PUBLIC_TRAIN_SHARDS, require_all=False)
    turnos, turno = len(range(rank % len(rutas), world, len(rutas))), rank // len(rutas)
    for i, doc in enumerate(rec.documentos_de_shards(rank, world, semilla, rutas)):
        if i % turnos == turno:
            yield doc


class Pool:
    def __init__(self, n: int, documentos, tokenizador):
        self.n, self.documentos, self.tok, self.bos = n, documentos, tokenizador, tokenizador.get_bos_token_id()
        self.tf, self.df, self.idf = np.zeros((DIMENSIONES, n), np.float32), np.zeros(DIMENSIONES, np.int64), np.zeros(DIMENSIONES, np.float32)
        self.normas, self.largos = np.zeros(n, np.float32), np.zeros(n, np.int64)
        self.cub, self.peso, self.tokens, self.indice, self.leidos, self.espera = [None] * n, [None] * n, [None] * n, [-1] * n, 0, []
        for s in range(n):
            self.poner(s)
        self.refrescar()

    def leer_doc(self) -> tuple[int, np.ndarray]:
        if not self.espera:
            lote = [next(self.documentos) for _ in range(TOKENIZAR_DE_A)]
            self.espera = [np.asarray(t, dtype=np.int32) for t in self.tok.encode(lote, prepend=self.bos, num_threads=1)][::-1]
        self.leidos += 1
        return self.leidos - 1, self.espera.pop()

    def norma(self, s: int) -> float:
        return math.sqrt(float(np.square(self.peso[s] * self.idf[self.cub[s]]).sum()))

    def poner(self, s: int) -> None:
        if self.cub[s] is not None:
            self.tf[self.cub[s], s] = 0.0
            self.df[self.cub[s]] -= 1
        self.indice[s], t = self.leer_doc()
        u, c = np.unique(((t[1:1 + PREFIJO].astype(np.uint64) * MEZCLA) & np.uint64(0xFFFFFFFF)) >> np.uint64(32 - BITS), return_counts=True)
        u = u.astype(np.int64)
        self.cub[s], self.peso[s], self.tokens[s], self.largos[s] = u, (1.0 + np.log(c)).astype(np.float32), t, len(t)
        self.tf[u, s] = self.peso[s]
        self.df[u] += 1
        self.normas[s] = self.norma(s)

    def refrescar(self) -> None:
        self.idf = np.log((1.0 + self.n) / (1.0 + self.df)).astype(np.float32)
        for s in range(self.n):
            self.normas[s] = self.norma(s)

    def parecidos(self, s: int) -> np.ndarray:
        u = self.cub[s]
        sim = np.einsum("i,ij->j", self.peso[s] * np.square(self.idf[u]), self.tf[u])
        sim /= np.maximum(self.normas, 1e-12) * max(float(self.normas[s]), 1e-12)
        sim[s] = -np.inf
        return sim

    def al_azar(self, rng: random.Random, s: int, v: int) -> int:
        if v < 0:
            v = rng.randrange(self.n - 1)
            return v + (v >= s)
        d = np.abs(self.largos - self.largos[v])
        d[[s, v]] = np.iinfo(np.int64).max
        return int(np.argpartition(d, CERCANOS)[rng.randrange(CERCANOS)])

    def cadena(self, rng: random.Random, capacidad: int, modo: str) -> tuple[torch.Tensor, list, list]:
        s, partes, largo, sims = rng.randrange(self.n), [], 0, []
        while True:
            partes.append((self.indice[s], self.tokens[s]))
            largo += len(self.tokens[s])
            if largo >= capacidad:
                break
            sim = self.parecidos(s)
            v = int(np.argmax(sim)) if self.normas[s] > 0 else -1
            if modo == "azar" or v < 0:
                v = self.al_azar(rng, s, v)
            sims.append(float(sim[v]))
            self.poner(s)
            s = v
        self.poner(s)
        inicios = np.cumsum([0] + [len(t) for _, t in partes[:-1]])
        meta = [(i, min(len(t), capacidad - int(a)), len(t)) for (i, t), a in zip(partes, inicios)]
        return torch.from_numpy(np.concatenate([t for _, t in partes])[:capacidad].astype(np.int64)), meta, sims


class Fuente:
    def __init__(self, tokenizador, documentos, semilla: int, rank: int, ancho: int, seq_len: int, pool: int, modo: str):
        if seq_len < 1 or ancho <= seq_len or (ancho - 1) % seq_len:
            raise ValueError(f"qf2_agrupa: el ancho del cargador menos uno ({ancho - 1}) no es un múltiplo de T = --seq-len = {seq_len}")
        self.cola, self.error = queue.Queue(maxsize=PROFUNDIDAD), None
        self.cpu_s, self.arranque_s, self.suma_sim, self.filas, self.docs, self.pares = 0.0, 0.0, 0.0, 0, 0, 0
        self.config = (ancho, seq_len, (ancho - 1) // seq_len, pool, modo)
        rng = random.Random(f"agrupa:{semilla}:{rank}")
        self.hilo = threading.Thread(target=self._correr, args=(tokenizador, documentos, rng, rank), name="qf2_agrupa", daemon=True)
        self.hilo.start()

    def traza(self) -> str:
        return ("qf2_agrupa: ancho={} T={} cadenas={} pool={} modo={} ".format(*self.config) + f"dimensiones={DIMENSIONES} rango 0: {self.filas} "
                f"filas, pool lleno en {self.arranque_s:.1f} s de cpu, {self.filas / max(self.cpu_s - self.arranque_s, 1e-9):.1f} filas/s de cpu, "
                f"{self.docs / max(self.filas, 1):.2f} documentos/fila, coseno medio entre consecutivos {self.suma_sim / max(self.pares, 1):.4f}")

    def _correr(self, tokenizador, documentos, rng, rank) -> None:
        _, seq_len, k, pool, modo = self.config
        try:
            nice.ceder("agrupa")
            t0 = time.thread_time()
            p = Pool(pool, documentos, tokenizador)
            self.arranque_s = time.thread_time() - t0
            while True:
                cadenas = [p.cadena(rng, seq_len + (j == k - 1), modo) for j in range(k)]
                meta, sims = [m for _, m, _ in cadenas], [x for _, _, c in cadenas for x in c]
                self.filas, self.docs, self.pares = self.filas + 1, self.docs + sum(map(len, meta)), self.pares + len(sims)
                self.suma_sim += sum(sims)
                if self.filas % REFRESCO == 0:
                    p.refrescar()
                self.cpu_s = time.thread_time() - t0
                if rank == 0 and (self.filas == 1 or self.filas % TRAZA_CADA == 0):
                    print(self.traza(), flush=True)
                self.cola.put((torch.cat([f for f, _, _ in cadenas]), meta))
        except BaseException as e:
            self.error = e
            self.cola.put(None)

    def siguiente(self):
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            item = None
        if item is None:
            raise RuntimeError(f"qf2_agrupa: el hilo de filas terminó o no dio fila en {ESPERA_S:.0f} s") from self.error
        return item


def envolver(base, tokenizador, rank: int, world: int, semilla: int, seq_len: int, fraccion: float, pool: int, modo: str, documentos=None, registro=None):
    fuente, acumulado = None, 0.0
    for paso, (x, y, estado) in enumerate(base):
        if fuente is None:
            docs = documentos if documentos is not None else documentos_de_rango(rank, world, semilla)
            fuente = Fuente(tokenizador, docs, semilla, rank, x.shape[1] + 1, seq_len, pool, modo)
        acumulado += fraccion * x.shape[0]
        n = int(acumulado + 1e-9)
        acumulado -= n
        if n:
            for r in random.Random(f"agrupa-filas:{semilla}:{rank}:{paso}").sample(range(x.shape[0]), n):
                fila, meta = fuente.siguiente()
                x[r].copy_(fila[:-1])
                y[r].copy_(fila[1:])
                if registro is not None:
                    registro.append((paso, r, meta))
        yield x, y, estado


def cargador(base, tokenizador, rank: int, world: int, semilla: int, seq_len: int, config: tuple[float, int, str] | None):
    return base if config is None else envolver(base, tokenizador, rank, world, semilla, seq_len, *config)
''',
    "qf2_brier.py": r'''"""QF2_BRIER: CE + Brier por trozos (propuesta f2-ce-brier, astra ronda 37 sección 3; banco CORE ronda 4).

Con z = cap * tanh(x @ w[:vocab].T / cap) (los logits tras el softcap, los de ce_trozos) y p = softmax(z), cada token
válido (objetivo != -1) aporta
    -log p_y + lam * (S2 - 2 p_y + 1),  S2 = sum_v p_v^2,
y la pérdida del micro-lote es la media sobre los válidos, como la CE del kit; los -1 no aportan CE ni Brier.
El gradiente respecto al logit tras el softcap es
    dL/dz_j = p_j * (1 + 2 lam (p_j - S2 + p_y)) - [j = y] * (1 + 2 lam p_y)
por token válido y entre el número de válidos, multiplicado por la derivada del softcap (1 - tanh^2) como en ce_trozos.
Forward y backward recorren trozos fijos de filas sin materializar one-hot ni las probabilidades de toda la secuencia:
el forward añade a la CE por trozos p = exp(z - lse), S2 y p_y por fila (guarda lse, S2 y p_y, tres vectores [filas]);
el backward recalcula los logits del trozo como ce_trozos, reutiliza p = exp(z - lse) y añade el factor por fila y el
scatter del objetivo. Formas fijas y bucle estático, sin ramas que dependan de los datos.

Con lam = 0 la pérdida y los gradientes son los de ce_trozos bit a bit. El mando vacío apaga la rama (train.py usa
ce_trozos); '0' entra por esta función con lam = 0; cualquier otro valor ha de ser un número finito >= 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ce_trozos import _trozo


class _CETrozosBrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo, lam):
        vale = y != -1
        total = x.new_zeros((), dtype=torch.float32)
        lses, s2s, pys = [], [], []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            zt = zc.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            p = torch.exp(zc - lse.unsqueeze(1))
            s2 = (p * p).sum(dim=-1)
            py = p.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            total = total + torch.where(vale[i:i + trozo], lse - zt + lam * (s2 - 2.0 * py + 1.0), 0.0).sum()
            lses.append(lse)
            s2s.append(s2)
            pys.append(py)
        ctx.save_for_backward(x, w, y, torch.cat(lses), torch.cat(s2s), torch.cat(pys))
        ctx.cap, ctx.vocab, ctx.trozo, ctx.lam = cap, vocab, trozo, lam
        return total / vale.sum()

    @staticmethod
    def backward(ctx, g):
        x, w, y, lse, s2, py = ctx.saved_tensors
        cap, vocab, trozo, lam = ctx.cap, ctx.vocab, ctx.trozo, ctx.lam
        vale = y != -1
        escala = g.float() / vale.sum()
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc, vc = x[i:i + trozo], y[i:i + trozo], vale[i:i + trozo]
            pyc = py[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d = d * (1.0 + 2.0 * lam * (d + (pyc - s2[i:i + trozo]).unsqueeze(1)))
            d.scatter_add_(1, yc.clamp_min(0).unsqueeze(1), -(1.0 + 2.0 * lam * pyc).unsqueeze(1))
            d = d * (escala * vc.float()).unsqueeze(1) * (1.0 - t * t)
            dz = d.to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None, None


def ce_brier_trozos(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int, lam: float) -> torch.Tensor:
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozosBrier.apply(x, w, targets, cap, vocab, trozo, float(lam))


@dataclass(frozen=True)
class Brier:
    lam: float

    def descripcion(self) -> str:
        return f"qf2_brier: ce + {self.lam} * brier por trozos (sum p^2 - 2 p_y + 1 por token válido, media sobre válidos)"


def leer(texto: str) -> Brier | None:
    texto = texto.strip()
    if not texto:
        return None
    try:
        lam = float(texto)
    except ValueError:
        raise ValueError(f"QF2_BRIER={texto!r} no es un número") from None
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"QF2_BRIER={texto!r} ha de ser un número finito >= 0")
    return Brier(lam)
''',
    "qf2_cola.py": r'''from __future__ import annotations

SEQ_MAX = 0
CARGADOR_LARGO = False
COLA_FRAC = 0.0
CALIENTA_LARGO = False
PASO_CALIENTA = 1


def configurar(seq_max: str, cargador_largo: str, cola_frac: str, calienta_largo: str) -> None:
    global SEQ_MAX, CARGADOR_LARGO, COLA_FRAC, CALIENTA_LARGO
    SEQ_MAX = int(seq_max.strip() or 0)
    CARGADOR_LARGO = cargador_largo.strip() not in ("", "0")
    COLA_FRAC = float(cola_frac.strip() or 0.0)
    CALIENTA_LARGO = COLA_FRAC > 0 or calienta_largo.strip() not in ("", "0")


def maximo(seq_len: int) -> int:
    return SEQ_MAX or seq_len


def descripcion() -> str:
    return (f"qf2_cola: seq_max={SEQ_MAX or '-'} cargador_largo={int(CARGADOR_LARGO)} "
            f"cola_frac={COLA_FRAC} calienta_largo={int(CALIENTA_LARGO)} paso_calienta={PASO_CALIENTA}")


class Cola:
    def __init__(self, seq_len: int, device_batch_size: int, total_batch_size: int, world_size: int,
                 sequence_max: int, avisar=print):
        if COLA_FRAC and not 0.0 < COLA_FRAC < 1.0:
            raise ValueError(f"QF2_COLA_FRAC must be in (0, 1), got {COLA_FRAC}")
        if CALIENTA_LARGO and not CARGADOR_LARGO:
            raise ValueError("QF2_COLA_FRAC and QF2_CALIENTA_LARGO require QF2_CARGADOR_LARGO=1")
        if sequence_max < seq_len:
            raise ValueError(f"QF2_SEQ_MAX={sequence_max} is below --seq-len={seq_len}")
        if CALIENTA_LARGO and sequence_max < 2 * seq_len:
            raise ValueError(f"long rows of {2 * seq_len} need QF2_SEQ_MAX >= {2 * seq_len}, got {sequence_max}")
        self.filas_partidas = device_batch_size
        self.filas_largas_micro = device_batch_size // 2 if device_batch_size % 2 == 0 else device_batch_size
        tokens_largos = self.filas_largas_micro * 2 * seq_len * world_size
        if CARGADOR_LARGO and total_batch_size % tokens_largos:
            raise ValueError(f"total_batch_size={total_batch_size} must divide long micro tokens={tokens_largos}")
        self.seq_len = seq_len
        self.filas_largas = total_batch_size // (2 * seq_len * world_size) if CARGADOR_LARGO else 0
        self.micro_partido = 2 * self.filas_largas // device_batch_size
        self.micro_largo = self.filas_largas // self.filas_largas_micro
        self.forma = None
        self.avisar = avisar

    def cargador(self, filas: int, seq_len: int) -> tuple[int, int]:
        return (self.filas_largas, 2 * self.seq_len) if CARGADOR_LARGO else (filas, seq_len)

    def es_larga(self, step: int, progreso: float) -> bool:
        calienta = CALIENTA_LARGO and step == PASO_CALIENTA
        return calienta or (COLA_FRAC > 0 and progreso >= 1.0 - COLA_FRAC)

    def pasos_compilacion(self) -> tuple[int, ...]:
        return (0, PASO_CALIENTA) if CALIENTA_LARGO else (0,)

    def lote(self, x, y, step: int, progreso: float, micro_kit: int):
        if not CARGADOR_LARGO:
            return x, y, micro_kit, self.filas_partidas
        larga = self.es_larga(step, progreso)
        micro = self.micro_largo if larga else self.micro_partido
        filas = self.filas_largas_micro if larga else self.filas_partidas
        if larga != self.forma:
            self.forma = larga
            self.avisar(f"qf2_cola: step {step} progress {progreso:.6f} form {'long' if larga else 'split'} "
                        f"rows {x.shape[0] * (1 if larga else 2)}x{x.shape[1] // (1 if larga else 2)} "
                        f"micro_rows={filas} grad_accum_steps={micro}")
        if larga:
            return x, y, micro, filas
        return x.reshape(-1, self.seq_len), y.reshape(-1, self.seq_len), micro, filas
''',
    "qf2_corpus_constructor.py": r'''"""QF2_MEMCORPUS: el constructor de la memoria del corpus, en el anfitrion y dentro del presupuesto de entreno.

El rango local 0 crea las tablas como ficheros compartidos (un uint64 o uint32 por cubo y orden) y, antes de iniciar el dispositivo,
bifurca un gestor sin hilos de Neuron. El gestor espera la marca de arranque, que el bucle escribe cuando empieza el reloj
oficial (paso TRAIN_STARTUP_STEPS_EXCLUDED), y entonces reparte los shards de entrenamiento que la red no lee (de 06499
hacia abajo, sin los primeros EXCLUIR) entre procesos con nice 10. Cada proceso tokeniza como el cargador (BOS delante de
cada documento) y actualiza las tablas. La marca de parada, que escribe el rango 0 antes de guardar el checkpoint, congela
la tabla: los procesos acaban el grupo de filas en curso y el gestor escribe 'parado'. Si el rango 0 muere, el gestor y sus
procesos mueren con el (PR_SET_PDEATHSIG en Linux, y el gestor comprueba a su padre); los ficheros de cada corrida se crean
de nuevo (inodo nuevo), asi que un proceso de una corrida anterior no puede escribir en las tablas de la siguiente.
"""
from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

import qf2_corpus_tabla as qt

ESPERA_S = 0.5
TOKENS_POR_SHARD = 5.4e7
G = {}


def ficheros(ruta: Path, ks, ancho: int = 64) -> dict:
    return {k: ruta / f"k{k}.u{ancho}" for k in ks}


def abrir(ruta: Path, cfg, crear: bool = False) -> qt.Tabla:
    arrays = {k: np.memmap(f, dtype=qt.TIPO[cfg.ancho], mode="w+" if crear else "r+", shape=(1 << cfg.bits,))
              for k, f in ficheros(ruta, cfg.ks, cfg.ancho).items()}
    return qt.Tabla.desde_arrays(cfg.ks, cfg.bits, arrays)


def shards_del_constructor(rutas: list, excluir: int) -> list:
    return sorted((p for p in rutas if int(p.stem.split("_")[-1]) >= excluir), key=lambda p: -int(p.stem.split("_")[-1]))


def morir_con_el_padre():
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)


def _iniciar(ruta, cfg, bos, tokenizador):
    morir_con_el_padre()
    os.nice(10)
    G.update(ruta=Path(ruta), bos=bos, tok=tokenizador, tabla=abrir(Path(ruta), cfg))


def _shard(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    ntok = pares = 0
    for g in range(pf.num_row_groups):
        if (G["ruta"] / "parar").exists():
            break
        textos = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        listas = G["tok"].encode(textos, prepend=G["bos"], num_threads=1)
        t = np.concatenate([np.asarray(x, dtype=np.uint16) for x in listas]) if listas else np.zeros(0, np.uint16)
        ntok += int(t.size)
        pares += G["tabla"].actualizar(t, G["bos"])
    return str(path), ntok, pares


def _escribir(ruta: Path, nombre: str, datos: dict):
    tmp = ruta / f".{nombre}.tmp"
    tmp.write_text(json.dumps(datos))
    tmp.replace(ruta / nombre)


def _gestor(ruta, cfg, shards, procesos, bos, tokenizador, padre):
    morir_con_el_padre()
    ruta = Path(ruta)
    os.nice(10)
    while not (ruta / "empezar").exists() and not (ruta / "parar").exists():
        if os.getppid() != padre:
            return
        time.sleep(ESPERA_S)
    t0, hechos, tokens, pares = time.time(), 0, 0, 0
    if not (ruta / "parar").exists():
        with mp.get_context("fork").Pool(procesos, initializer=_iniciar, initargs=(str(ruta), cfg, bos, tokenizador)) as pool:
            for _, ntok, np_ in pool.imap_unordered(_shard, shards):
                hechos, tokens, pares = hechos + 1, tokens + ntok, pares + np_
                _escribir(ruta, "progreso.json", {"shards": hechos, "de": len(shards), "tokens": tokens, "pares": pares,
                                                  "t": time.time() - t0})
                if (ruta / "parar").exists() or os.getppid() != padre:
                    break
            pool.terminate()
    _escribir(ruta, "parado", {"shards": hechos, "de": len(shards), "tokens": tokens, "pares": pares, "t": time.time() - t0,
                               "completo": hechos == len(shards)})


class Constructor:
    def __init__(self, cfg, ruta: Path, shards: list, procesos: int, bos: int, tokenizador):
        self.cfg, self.ruta, self.shards = cfg, Path(ruta), list(shards)
        self.ruta.mkdir(parents=True, exist_ok=True)
        for f in list(ficheros(self.ruta, cfg.ks, cfg.ancho).values()) + [self.ruta / n for n in ("creada", "empezar", "parar", "parado", "progreso.json")]:
            f.unlink(missing_ok=True)
        abrir(self.ruta, cfg, crear=True)
        _escribir(self.ruta, "creada", {"corrida": corrida()})
        atexit.register(self.parar, 5.0)
        self.gestor = mp.get_context("fork").Process(target=_gestor, args=(str(self.ruta), cfg, self.shards, procesos, bos, tokenizador, os.getpid()),
                                                     name="qf2_memcorpus", daemon=False)
        self.final = None
        self.gestor.start()

    def empezar(self):
        (self.ruta / "empezar").write_text(str(time.time()))

    def acabado(self) -> bool:
        return (self.ruta / "parado").exists()

    def progreso(self) -> dict:
        try:
            return json.loads((self.ruta / "progreso.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def parar(self, espera_s: float = 120.0) -> dict:
        if self.final is not None:
            return self.final
        try:
            (self.ruta / "parar").write_text(str(time.time()))
        except OSError:
            self.gestor.terminate()
            self.final = {"ruta_perdida": True}
            return self.final
        limite = time.time() + espera_s
        while not (self.ruta / "parado").exists() and time.time() < limite and self.gestor.is_alive():
            time.sleep(ESPERA_S)
        if self.gestor.is_alive() and not (self.ruta / "parado").exists():
            self.gestor.terminate()
        try:
            self.final = json.loads((self.ruta / "parado").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.final = {"parado_a_la_fuerza": True, **self.progreso()}
        self.gestor.join(timeout=10.0)
        return self.final

    def limpiar(self):
        self.parar(5.0)
        for f in list(ficheros(self.ruta, self.cfg.ks, self.cfg.ancho).values()) + [self.ruta / n for n in ("creada", "empezar", "parar", "parado", "progreso.json")]:
            f.unlink(missing_ok=True)
        try:
            self.ruta.rmdir()
        except OSError:
            pass

    def descripcion(self) -> str:
        return (f"qf2_memcorpus: bits={self.cfg.bits} ks={self.cfg.ks} shards={len(self.shards)} "
                f"(de {self.shards[0].name if self.shards else '-'} a {self.shards[-1].name if self.shards else '-'}) ruta={self.ruta}"
                + ("" if self.shards else " AVISO: SIN SHARDS, la memoria queda vacia"))


def corrida() -> str:
    return os.environ.get("TORCHELASTIC_RUN_ID", "") or str(os.getppid())


def esperar_creada(ruta: Path, espera_s: float = 600.0):
    limite = time.time() + espera_s
    while True:
        try:
            if json.loads((Path(ruta) / "creada").read_text()).get("corrida") == corrida():
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.time() > limite:
            raise RuntimeError(f"qf2_memcorpus: las tablas de esta corrida no aparecieron en {ruta} en {espera_s:.0f} s")
        time.sleep(ESPERA_S)
''',
    "qf2_corpus_mezcla.py": r'''"""QF2_MEMCORPUS: mezcla del modelo con la memoria de n-gramas del corpus (qf2_corpus_tabla), igual en entreno y evaluacion.

La salida es p = (1 - lam) softmax(z) + lam [v == candidato], con lam = sigmoid(w[celda]) aprendido por celda (orden del
contexto encontrado, cubo de su cuenta neta); la celda 0 (sin contexto) deja p = softmax(z). En entreno la perdida de cada
posicion sale de la nll del modelo sin materializar la mezcla: -log((1 - lam) e^-nll + lam [y == candidato]). En evaluacion
el forward devuelve log p completo, cuya entropia cruzada es la misma funcion. Las tablas viven en el anfitrion: en entreno
como ficheros compartidos que llena el constructor; en evaluacion como Parametros enteros del checkpoint (memcorpus.t{k}).
Con minimo > 1 solo cuentan los contextos cuya cuenta neta llega a minimo (si el orden mas largo no llega, se usa el siguiente),
en entreno y en evaluacion, y el checkpoint guarda solo esas entradas: las de cuenta menor se ponen a cero al guardar.
Con contexto = 1 la puerta suma v . h al logit de su celda (h = salida normalizada del ultimo bloque, v iniciado a cero).
reducir(..., tronco=a) entrena con (1 - a) * mezcla + a * nll del tronco; train.py solo lo pasa con model.training, asi que
la salida del modelo y la evaluacion no cambian.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import qf2_corpus_tabla as qt

CUBOS = np.array([1, 2, 3, 5, 9, 17, 65, 257])
LAM_INICIAL = -4.0
PREFIJO = "memcorpus.t"


@dataclass(frozen=True)
class Cfg:
    bits: int
    ks: tuple
    minimo: int = 1
    ancho: int = 64
    contexto: int = 0

    @property
    def celdas(self) -> int:
        return 1 + len(self.ks) * CUBOS.size


def desde_texto(texto: str) -> Cfg:
    bits, ks, minimo, ancho, contexto = (texto.split(":") + ["", "", "", ""])[:5]
    return Cfg(int(bits), tuple(sorted(int(k) for k in (ks or ",".join(map(str, qt.KS_DEF))).split(","))), int(minimo or 1), int(ancho or 64),
               int(contexto or 0))


def tipo_torch(cfg: Cfg):
    return torch.int32 if cfg.ancho == 32 else torch.int64


def celdas(ki: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    cubo = np.clip(np.digitize(cnt, CUBOS) - 1, 0, CUBOS.size - 1)
    return np.where(ki >= 0, 1 + ki.astype(np.int64) * CUBOS.size + cubo, 0)


def consultar(tabla: qt.Tabla, ids: np.ndarray, bos: int, minimo: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    tok, ki, cnt = tabla.consultar(ids.astype(np.int64), bos, minimo)
    return torch.from_numpy(np.maximum(tok, 0)), torch.from_numpy(celdas(ki, cnt))


class Puerta(nn.Module):
    def __init__(self, cfg: Cfg, dim: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.full((cfg.celdas,), LAM_INICIAL))
        self.v = nn.Parameter(torch.zeros(dim)) if cfg.contexto else None

    def desplazamiento(self, h: torch.Tensor):
        return None if self.v is None else h.float() @ self.v.float()

    def lam(self, celda: torch.Tensor, s=None) -> tuple[torch.Tensor, torch.Tensor]:
        activa = celda > 0
        uno = (celda.unsqueeze(-1) == torch.arange(self.w.numel(), device=celda.device)).float()
        u = uno @ self.w.float()
        lam = torch.sigmoid(u if s is None else u + s).clamp(1e-6, 1 - 1e-6)
        return torch.where(activa, lam, torch.full_like(lam, 0.5)), activa


def perdida(puerta: Puerta, nll: torch.Tensor, y: torch.Tensor, cand: torch.Tensor, celda: torch.Tensor, s=None) -> torch.Tensor:
    lam, activa = puerta.lam(celda, s)
    acierta = cand == y
    resto = 1 - lam
    mezcla = torch.where(acierta, -torch.log(resto * torch.exp(-nll) + lam), nll - torch.log(resto))
    return torch.where(activa, mezcla, nll)


@torch.compiler.disable
def reducir(puerta: Puerta, nll: torch.Tensor, targets: torch.Tensor, cand: torch.Tensor, celda: torch.Tensor, reduccion: str,
            s=None, tronco: float = 0.0) -> torch.Tensor:
    y = targets.reshape(-1)
    vale = y != -1
    mezclada = torch.where(vale, perdida(puerta, nll, y, cand.reshape(-1), celda.reshape(-1), None if s is None else s.reshape(-1)), 0.0)
    if tronco:
        mezclada = (1.0 - tronco) * mezclada + tronco * torch.where(vale, nll, 0.0)
    if reduccion == "mean":
        return mezclada.sum() / vale.sum()
    if reduccion == "sum":
        return mezclada.sum()
    return mezclada.view_as(targets)


def log_probs(puerta: Puerta, z: torch.Tensor, cand: torch.Tensor, celda: torch.Tensor, s=None) -> torch.Tensor:
    lam, activa = puerta.lam(celda, s)
    lp = torch.log_softmax(z.float(), dim=-1)
    lc = lp.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
    nuevo_c = torch.logaddexp(lc + torch.log1p(-lam), torch.log(lam))
    mezcla = (lp + torch.log1p(-lam).unsqueeze(-1)).scatter(-1, cand.unsqueeze(-1), nuevo_c.unsqueeze(-1))
    return torch.where(activa.unsqueeze(-1), mezcla, lp)


def tabla_de_estado(cfg: Cfg, estado: dict) -> qt.Tabla:
    arrays = {}
    for k in cfg.ks:
        t = estado[f"{PREFIJO}{k}"]
        if tuple(t.shape) != (1 << cfg.bits,) or t.dtype != tipo_torch(cfg):
            raise RuntimeError(f"tabla memcorpus k={k} con forma/dtype inesperados: {tuple(t.shape)} {t.dtype}")
        arrays[k] = t.detach().numpy().view(qt.TIPO[cfg.ancho])
    return qt.Tabla.desde_arrays(cfg.ks, cfg.bits, arrays)


def asignar(model: nn.Module, estado: dict) -> None:
    cfg = model.memcorpus_cfg
    model.memcorpus = nn.Module()
    for k in cfg.ks:
        setattr(model.memcorpus, f"t{k}", nn.Parameter(estado[f"{PREFIJO}{k}"], requires_grad=False))
    model.memcorpus_tabla = tabla_de_estado(cfg, estado)


def claves(cfg: Cfg) -> list:
    return [f"{PREFIJO}{k}" for k in cfg.ks]


@torch.compiler.disable
def consultar_modelo(model: nn.Module, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cand, celda = consultar(model.memcorpus_tabla, idx.detach().cpu().numpy(), model.memcorpus_bos, model.memcorpus_cfg.minimo)
    return cand.to(idx.device), celda.to(idx.device)


def podar(v: np.ndarray, minimo: int, hilos: int = 32, trozo: int = 1 << 26) -> None:
    def uno(a):
        x = v[a:a + trozo]
        x *= (x & v.dtype.type(qt.cuenta_max(v.dtype))) >= v.dtype.type(minimo)
    with ThreadPoolExecutor(hilos) as ex:
        list(ex.map(uno, range(0, v.size, trozo)))


def para_guardar(tabla: qt.Tabla, minimo: int = 1) -> dict:
    if minimo > 1:
        for v in tabla.t.values():
            podar(v, minimo)
    return {f"{PREFIJO}{k}": torch.from_numpy(np.asarray(v).view(np.int64 if v.dtype == np.uint64 else np.int32)) for k, v in tabla.t.items()}


def vacias(cfg: Cfg) -> dict:
    return {f"{PREFIJO}{k}": torch.zeros(1 << cfg.bits, dtype=tipo_torch(cfg)) for k in cfg.ks}
''',
    "qf2_corpus_tabla.py": r'''"""Memoria de n-gramas largos del corpus: una tabla de cubos por orden k con el siguiente token mayoritario de cada contexto.

Cada cubo es un uint64: cuenta neta (bits 0-23), siguiente token (24-38) y huella del contexto (39-55, nunca 0); con ancho 32,
un uint32: cuenta (0-5, tope 63), siguiente (6-20) y huella (21-31, de 1 a 2047), el doble de entradas por byte. El cubo y la
huella salen del hash del contexto de k tokens (sin BOS dentro, siguiente distinto de BOS). La actualizacion es una mayoria
por cubo: la masa del par guardado suma, la de cualquier otro par resta, y si la cuenta no queda positiva el par mas
frecuente del trozo ocupa el cubo. Los contextos vistos una vez se desalojan con la primera colision; los repetidos con un
siguiente dominante se quedan. La consulta devuelve, por posicion, el candidato del k mas largo con huella y su cuenta.
"""
from __future__ import annotations

import numpy as np

VOCAB = 32768
KS_DEF = (8, 12, 16, 24, 32)
TROZO = 4_000_000
M64 = (1 << 64) - 1
CNT_MAX = (1 << 24) - 1
BITS_CUENTA = {64: 24, 32: 6}
BITS_HUELLA = {64: 17, 32: 11}
TIPO = {64: np.uint64, 32: np.uint32}


def _splitmix(i: int) -> int:
    z = ((i + 0x5EED) * 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) | 1


MULT = np.array([_splitmix(i) for i in range(64)], dtype=np.uint64)


def _mezcla(x):
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xC4CEB9FE1A85EC53)
    return x ^ (x >> np.uint64(33))


def claves(t: np.ndarray, ks, bos: int):
    """Para t [..., L] y k crecientes: (k, h, valido) del contexto de k tokens que acaba en cada posicion i >= k-1."""
    t64 = t.astype(np.uint64) + np.uint64(1)
    largo = t.shape[-1]
    s = np.zeros(t.shape, np.uint64)
    malo = np.zeros(t.shape, bool)
    j = 0
    with np.errstate(over="ignore"):
        for k in sorted(ks):
            while j < k:
                if j < largo:
                    s[..., j:] += t64[..., :largo - j] * MULT[j]
                    malo[..., j:] |= t[..., :largo - j] == bos
                j += 1
            yield k, _mezcla(s[..., k - 1:] ^ np.uint64(k)), ~malo[..., k - 1:]


def cuenta_max(tipo) -> int:
    return (1 << BITS_CUENTA[np.dtype(tipo).itemsize * 8]) - 1


def _partes(h: np.ndarray, bits: int, ancho: int = 64):
    cubo = (h & np.uint64((1 << bits) - 1)).astype(np.int64)
    if ancho == 32:
        return cubo, ((h >> np.uint64(40)) % np.uint64(2047) + np.uint64(1)).astype(np.int64)
    return cubo, ((h >> np.uint64(47)) | np.uint64(1)).astype(np.int64)


def _unicos(cubo: np.ndarray, fv: np.ndarray, desp: int, bits: int):
    if bits + desp <= 63:
        clave, c = np.unique((cubo << desp) | fv, return_counts=True)
        return clave >> desp, clave & ((1 << desp) - 1), c
    o = np.lexsort((fv, cubo))
    cb, f = cubo[o], fv[o]
    ini = np.flatnonzero(np.r_[True, (cb[1:] != cb[:-1]) | (f[1:] != f[:-1])])
    return cb[ini], f[ini], np.diff(np.r_[ini, cb.size])


class Tabla:
    def __init__(self, ks=KS_DEF, bits: int = 24, bufs=None, ancho: int = 64):
        self.ks, self.bits, self.ancho = tuple(sorted(ks)), bits, ancho
        n, tipo = 1 << bits, TIPO[ancho]
        self.t = {k: (np.zeros(n, tipo) if bufs is None else np.ndarray((n,), tipo, buffer=bufs[k])) for k in self.ks}

    @classmethod
    def desde_arrays(cls, ks, bits: int, arrays: dict) -> "Tabla":
        tb = cls.__new__(cls)
        tb.ks, tb.bits, tb.t = tuple(sorted(ks)), bits, {k: arrays[k] for k in sorted(ks)}
        tb.ancho = arrays[tb.ks[0]].dtype.itemsize * 8
        return tb

    @property
    def cb(self) -> int:
        return BITS_CUENTA[self.ancho]

    def actualizar(self, tokens: np.ndarray, bos: int) -> int:
        """Cuenta los pares (contexto, siguiente) de una secuencia de tokens de documentos con BOS delante; devuelve pares."""
        kmax, total = self.ks[-1], 0
        for a in range(0, max(tokens.size - self.ks[0], 0), TROZO):
            s = tokens[a:min(tokens.size, a + TROZO + kmax)]
            for k, h, ok in claves(s[:-1], self.ks, bos):
                h, ok = h[:TROZO], ok[:TROZO]
                nx = s[k:][:TROZO]
                ok &= nx != bos
                total += int(ok.sum())
                self._sumar(k, h[ok], nx[ok].astype(np.int64))
        return total

    def _sumar(self, k: int, h: np.ndarray, nx: np.ndarray):
        if h.size == 0:
            return
        cubo, huella = _partes(h, self.bits, self.ancho)
        cb_, cmax = self.cb, (1 << self.cb) - 1
        cb, fv, c = _unicos(cubo, (huella << 15) | nx, 15 + BITS_HUELLA[self.ancho], self.bits)
        ini = np.flatnonzero(np.r_[True, cb[1:] != cb[:-1]])
        tam = np.diff(np.r_[ini, cb.size])
        g = np.repeat(np.arange(ini.size), tam)
        tot = np.add.reduceat(c, ini)
        gmax = np.maximum.reduceat(c, ini)
        es_max = c == gmax[g]
        _, primero = np.unique(g[es_max], return_index=True)
        cand = np.flatnonzero(es_max)[primero]
        tabla = self.t[k]
        bloque = cb[ini]
        e = tabla[bloque].astype(np.int64)
        e_fv, e_c = e >> cb_, e & cmax
        m_e = np.add.reduceat(np.where(fv == e_fv[g], c, 0), ini)
        nuevo = e_c + 2 * m_e - tot
        queda = (nuevo > 0) & (e_c > 0)
        c_rep = np.maximum(2 * c[cand] - tot - e_c, 1)
        val = np.where(queda, (e_fv << cb_) | np.minimum(nuevo, cmax), (fv[cand] << cb_) | np.minimum(c_rep, cmax))
        tabla[bloque] = val.astype(tabla.dtype)

    def consultar(self, ids: np.ndarray, bos: int, minimo: int = 1):
        """ids [B, T]: candidato para el token T+1 de cada posicion (-1 si ninguno), indice del k usado y cuenta neta."""
        forma = ids.shape
        tok = np.full(forma, -1, np.int64)
        ki = np.full(forma, -1, np.int8)
        cnt = np.zeros(forma, np.int64)
        cb_ = self.cb
        for i, (k, h, ok) in enumerate(claves(ids, self.ks, bos)):
            cubo, huella = _partes(h, self.bits, self.ancho)
            e = self.t[k][cubo].astype(np.int64)
            ec = e & ((1 << cb_) - 1)
            hit = ok & ((e >> (cb_ + 15)) == huella) & (ec >= minimo)
            sl = (Ellipsis, slice(k - 1, None))
            tok[sl] = np.where(hit, (e >> cb_) & ((1 << 15) - 1), tok[sl])
            ki[sl] = np.where(hit, i, ki[sl])
            cnt[sl] = np.where(hit, ec, cnt[sl])
        return tok, ki, cnt

    def ocupacion(self, paso: int = 1) -> dict:
        return {k: float(np.count_nonzero(v[::paso]) / v[::paso].size) for k, v in self.t.items()}
''',
    "qf2_diff.py": r'''"""QF2_DIFF_CAPAS / QF2_DIFF_SIGNO: second-query attention branch in a few layers (DIFF V2 style, adapted to the kit).

In each marked layer a second query Q2 (own projection c_q2, same rotary and qk-norm as Q1) attends over the same K and V, and
the layer output is  SDPA(Q1,K,V) + signo * sigmoid(lam(x)) * SDPA(Q2,K,V)  with signo = -1 (difference) or +1 (sum). lam is a
bias-free Linear n_embd -> n_head on the normalised input of the attention, so the gate is per head and per position and equals
0.5 at initialisation (zero weights). Everything is per position: causality and batch independence are those of SDPA. The
modules are built under a forked RNG and initialised last, so with the flags off the model and its initialisation stream are
untouched. Indices at or beyond n_layer are ignored, so a flag baked for d22 still builds in the d4 CPU validations. Rule for six layers: espaciadas(n_layer) = 1 + 4k for d22 and d24 (1,5,9,13,17,21), written colon-separated in the flag.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNOS = (-1, 1)
SEPARADOR = re.compile(r"[\s,:;+.\-]+")


def indices(capas: str, n_layer: int) -> tuple[int, ...]:
    trozos = [t for t in SEPARADOR.split(capas.strip()) if t]
    if not trozos:
        return ()
    if not all(t.isdigit() for t in trozos):
        raise ValueError(f"QF2_DIFF_CAPAS={capas!r} debe ser una lista de indices de capa")
    lista = tuple(int(t) for t in trozos)
    if len(set(lista)) != len(lista):
        raise ValueError(f"QF2_DIFF_CAPAS={capas!r} repite capas")
    return tuple(sorted(i for i in lista if i < n_layer))


def espaciadas(n_layer: int, cuantas: int = 6) -> tuple[int, ...]:
    paso = (n_layer - 2) // (cuantas - 1)
    return tuple(1 + paso * k for k in range(cuantas))


def comprobar_signo(signo: int) -> None:
    if signo not in SIGNOS:
        raise ValueError(f"QF2_DIFF_SIGNO={signo} debe ser -1 (diferencia) o 1 (suma)")


class Lineal(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class RamaDiff(nn.Module):
    def __init__(self, n_embd: int, n_head: int, signo: int):
        super().__init__()
        comprobar_signo(signo)
        self.signo = float(signo)
        with torch.random.fork_rng(devices=[]):
            self.c_q2 = Lineal(n_embd, n_embd, bias=False)
            self.lam = Lineal(n_embd, n_head, bias=False)
        nn.init.zeros_(self.lam.weight)

    @torch.no_grad()
    def iniciar(self) -> None:
        s = math.sqrt(3.0) * self.c_q2.weight.size(1) ** -0.5
        nn.init.uniform_(self.c_q2.weight, -s, s)
        nn.init.zeros_(self.lam.weight)

    def puerta(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lam(x)).transpose(1, 2).unsqueeze(-1)

    def combinar(self, y1: torch.Tensor, y2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return y1 + self.signo * self.puerta(x) * y2
''',
    "qf2_equilibrio.py": r'''"""QF2_EQUILIBRIO='mezcla:tope': CE equilibrada por longitud de documento (propuesta f2-equilibrio-documental, astra
ronda 37 sección 2 punto 1; banco CORE ronda 4, prerregistro 20261029T022905Z).

Recibe la NLL por token de ce_trozos.ce_trozos_tokens con forma [B, T] (0 donde el objetivo es -1), la máscara de
objetivos válidos y es_bos = (idx == BOS) de la entrada del micro-lote, y devuelve la pérdida escalar

    w_i = (1 - mezcla) + mezcla * min(tope, c / largo_i)    en los válidos, 0 en los -1
    w   = w * V / sum(w)                                     media 1 sobre los V válidos del micro-lote
    pérdida = sum(w * nll) / V

sin detach: w sale de idx y de los objetivos, no de los parámetros, y d pérdida / d nll_i = w_i / V.

Documento de una posición: el de su entrada idx[i]; su objetivo idx[i+1] se predice leyendo ese documento, y el último
token de un documento, que predice el BOS del siguiente, cuenta con el suyo. id = L @ es_bos con L la triangular
inferior de unos T x T en float32 (la numeración de qf2_resumen.mascara: enteros exactos), y largo_i = número de
posiciones de la fila con id == id_i, la suma por fila de M = (id_i == id_j). Es pérdida de entreno, no entrada del
modelo: puede mirar el documento entero. M se cuenta por bloques de BLOQUE filas, [B, BLOQUE, T] cada uno, y se
concatena: a B=4 y T=2048 la M entera en float32 son 64 MiB transitorios y por bloques 8 MiB; medido en CPU por el
contrato, la M entera pica más memoria que la máscara que qf2_resumen ya lleva en el grafo y por bloques menos.

Recortados: un segmento que toca un borde de la secuencia cuenta como largo = T. Son el de id 0 (posiciones antes del
primer BOS: la cola de un documento anterior) y el del último id de la fila (contiene la posición T-1: sigue tras el
borde o el cargador lo recortó). Su largo real es >= el visible y no se ve desde la secuencia; con el visible, los tres
primeros tokens de un documento largo pesarían como un documento corto y el cargador best-fit recorta uno por fila:
favorecería fragmentos. Con T su peso es el mínimo posible. Una fila sin BOS es un único segmento recortado: peso
(1 - mezcla) + mezcla * min(tope, c / T), finito.

Pareado: con QF2_CARGADOR_LARGO en forma partida (qf2_cola.lote hace reshape de cada fila de 2T en dos filas
consecutivas y el micro-lote corta de a un número par), las filas 2r y 2r+1 son las dos mitades de una misma fila y sus
documentos se ven enteros en el micro-lote. Con pareado=True cada par se numera como una fila de 2T (los ids de la
segunda mitad siguen a los de la primera) y el documento que cruza la mitad recibe su largo real; solo son recortados el
tramo anterior al primer BOS de la mitad primera y el que contiene la posición 2T-1 de la segunda, con largo T. fijar()
lo pone train.py tras cada qf2_cola.lote; B debe ser par.

c = V / sum_válidos(1 / largo_i): la longitud media de documento del micro-lote (válidos entre número efectivo de
documentos; un recortado con v posiciones válidas cuenta v / T de documento). Con esta c el término documental tiene
media exactamente 1 sobre los válidos antes del tope, así que mezcla es la fracción de la pérdida que va a la CE por
documento y la normalización solo devuelve lo que quita el tope. mezcla = 0 es la CE ordinaria (w = 1 exacto).

Grafo para Neuron: formas fijas, sin ramas que dependan de los datos, sin cumsum, cummax ni sort; las reducciones son
por fila sobre el eje T salvo las dos sumas del micro-lote sobre [B, T], como las de ce_trozos. Vacío o '0' apaga.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BLOQUE = 256
MINIMO = 1e-30


def _numero(texto: str, nombre: str) -> float:
    try:
        valor = float(texto)
    except ValueError:
        raise ValueError(f"{nombre}={texto!r} no es un número") from None
    if not math.isfinite(valor):
        raise ValueError(f"{nombre}={texto!r} no es finito")
    return valor


PAREADO = False


def fijar(pareado: bool) -> None:
    global PAREADO
    PAREADO = bool(pareado)


def identificadores(es_bos: torch.Tensor, pareado: bool = False) -> torch.Tensor:
    t = es_bos.size(-1)
    pos = torch.arange(t, device=es_bos.device)
    inferior = (pos.view(1, -1) <= pos.view(-1, 1)).to(torch.float32)
    if not pareado:
        return F.linear(es_bos.to(torch.float32), inferior)
    ident = F.linear(es_bos.to(torch.float32).reshape(-1, 2, t), inferior)
    return torch.cat([ident[:, 0], ident[:, 1] + ident[:, 0, -1:]], dim=1)


def contar(ident: torch.Tensor, bloque: int = BLOQUE) -> torch.Tensor:
    partes = [(ident[:, s:s + bloque].unsqueeze(-1) == ident.unsqueeze(-2)).to(torch.float32).sum(-1)
              for s in range(0, ident.size(-1), bloque)]
    return torch.cat(partes, dim=1)


def largos(es_bos: torch.Tensor, bloque: int = BLOQUE, pareado: bool = False) -> torch.Tensor:
    t = es_bos.size(-1)
    if pareado and es_bos.size(0) % 2:
        raise ValueError(f"QF2_EQUILIBRIO pareado necesita un número par de filas, hay {es_bos.size(0)}")
    ident = identificadores(es_bos, pareado)
    recortado = (ident == 0) | (ident == ident[:, -1:])
    largo = torch.where(recortado, float(t), contar(ident, bloque))
    return largo.reshape(-1, t) if pareado else largo


def pesos(vale: torch.Tensor, es_bos: torch.Tensor, mezcla: float, tope: float, pareado: bool = False) -> torch.Tensor:
    v = vale.to(torch.float32)
    largo = largos(es_bos, pareado=pareado)
    validos = v.sum()
    c = validos / (v / largo).sum().clamp_min(MINIMO)
    w = v * ((1.0 - mezcla) + mezcla * torch.clamp(c / largo, max=tope))
    return w * (validos / w.sum().clamp_min(MINIMO))


@dataclass(frozen=True)
class Equilibrio:
    mezcla: float
    tope: float

    def pesos(self, vale: torch.Tensor, es_bos: torch.Tensor, pareado: bool | None = None) -> torch.Tensor:
        return pesos(vale, es_bos, self.mezcla, self.tope, PAREADO if pareado is None else pareado)

    def __call__(self, nll: torch.Tensor, vale: torch.Tensor, es_bos: torch.Tensor, pareado: bool | None = None) -> torch.Tensor:
        return (self.pesos(vale, es_bos, pareado) * nll).sum() / vale.sum()

    def descripcion(self) -> str:
        return (f"qf2_equilibrio: mezcla={self.mezcla} tope={self.tope} c=media de documento del micro-lote "
                f"recortados=T pareado={PAREADO} bloque={BLOQUE}")


def leer(texto: str) -> Equilibrio | None:
    texto = texto.strip()
    if texto in ("", "0"):
        return None
    partes = texto.split(":")
    if len(partes) != 2:
        raise ValueError(f"QF2_EQUILIBRIO={texto!r} no es 'mezcla:tope'")
    mezcla, tope = _numero(partes[0], "QF2_EQUILIBRIO mezcla"), _numero(partes[1], "QF2_EQUILIBRIO tope")
    if not 0.0 <= mezcla <= 1.0 or tope <= 0:
        raise ValueError(f"QF2_EQUILIBRIO={texto!r}: mezcla en [0, 1] y tope > 0")
    return Equilibrio(mezcla, tope)
''',
    "qf2_filas.py": r'''"""Filas sustituidas del lote de entreno en el paquete 302: QF2_AGRUPA y QF2_RECOLOCADO sobre filas disjuntas.

cargador() compone los dos sustituidores sobre el cargador del kit, antes de la precarga. Sin QF2_AGRUPA devuelve
qf2_recolocado.cargador sobre el cargador del kit (el 295 bit a bit). Con QF2_AGRUPA, qf2_agrupa.envolver elige primero
sus filas con su acumulador y su azar por (semilla, rango, paso) y las anota en Ocupadas, que reenvía cada anotación al
registro que se le dé; qf2_recolocado.envolver elige después, con su azar por (semilla, rango, paso), solo entre las filas
que agrupa no tomó en ese paso. El número de filas de recolocado sale de su acumulador sobre el total de filas
(fracción * filas del lote, como en el 295), recortado a las libres; lo que no cabe se queda en su acumulador y sale en los
pasos siguientes, así que con la suma de fracciones <= 1 la fracción de cada uno sobre el total sigue siendo exacta a la
larga. Con todas las filas libres la elección de recolocado es la del 295 (sample sobre range(filas)). Con QF2_RECOLOCADO
a 0 es el 296. Cada registro anota solo las filas que escribe su sustituidor y ninguna fila la reescribe el otro.
validar() falla al leer los mandos si QF2_AGRUPA f + QF2_RECOLOCADO > 1.

Documentos disjuntos: con QF2_AGRUPA y QF2_RECOLOCADO > 0 a la vez, cada sustituidor lee su flujo de siempre (documentos_de_rango
el de agrupa, documentos_de_shards el de recolocado; con un turno por shard son el mismo flujo en el mismo orden) y se queda
solo con su mitad: el bit bajo de blake2b del texto (lado()) es 0 para agrupa y 1 para recolocado. Ningún texto entra en los
dos tipos de fila, tampoco entre rangos ni si el corpus lo repite. Con uno solo encendido el flujo llega entero (296 o 295
bit a bit).

Regla (e) con QF2_AGRUPA y QF2_EQUILIBRIO: la cadena 0 de una fila de agrupa acaba cortada en la posición T - 1 y la cadena 1
empieza con BOS en T. EquilibrioAgrupa (equilibrio() la pone en lugar de Equilibrio solo con QF2_AGRUPA) cuenta ese último
documento de la cadena 0 como recortado (largo = ancho del tensor, como los extremos de la fila), tanto en la forma partida con
pareado (BOS en la posición 0 de la segunda mitad) como en la larga (BOS en la posición ancho/2). Un documento real que acabase
justo en T - 1 se trata igual (conservador). Sin ningún BOS en el corte es qf2_equilibrio bit a bit.

Enganche para el 297 (QF2_MARGEN, sin fusionar): margen toma primero sus dos filas (las dos últimas del micro-lote en sus
micro-pasos fijos) y los demás las evitan: se anotarían en Ocupadas antes de que elija agrupa, qf2_agrupa.envolver
necesitaría el mismo parámetro ocupadas que qf2_recolocado.envolver y validar() sumaría su fracción de filas.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

import qf2_agrupa as ag
import qf2_cola
import qf2_equilibrio as eq
import qf2_recolocado as rec

HOLGURA = 1e-9


class Ocupadas:
    def __init__(self, registro=None):
        self.paso, self.filas, self.registro = None, set(), registro

    def append(self, anotacion) -> None:
        paso, fila = anotacion[0], anotacion[1]
        if paso != self.paso:
            self.paso, self.filas = paso, set()
        self.filas.add(fila)
        if self.registro is not None:
            self.registro.append(anotacion)

    def libres(self, paso: int, filas: int):
        tomadas = self.filas if paso == self.paso else set()
        return [r for r in range(filas) if r not in tomadas] if tomadas else range(filas)


def lado(texto: str) -> int:
    return hashlib.blake2b(texto.encode("utf-8", "surrogatepass"), digest_size=1).digest()[0] & 1


def mitad(documentos, cual: int):
    return (d for d in documentos if lado(d) == cual)


def repartir(documentos, rank: int, world: int, semilla: int) -> tuple:
    a = documentos[0] if documentos[0] is not None else ag.documentos_de_rango(rank, world, semilla)
    r = documentos[1] if documentos[1] is not None else rec.documentos_de_shards(rank, world, semilla)
    return mitad(a, 0), mitad(r, 1)


def validar(agrupa, fraccion: float) -> None:
    suma = (agrupa[0] if agrupa is not None else 0.0) + max(fraccion, 0.0)
    if suma > 1.0 + HOLGURA:
        raise ValueError(f"QF2_AGRUPA (fracción {agrupa[0] if agrupa is not None else 0.0}) + QF2_RECOLOCADO ({fraccion}) = {suma} > 1: "
                         "las filas que sustituyen son disjuntas y no caben en el lote")


def cargador(base, tokenizador, rank: int, world: int, semilla: int, seq_len: int, agrupa, fraccion: float, ce: str, rafaga: str,
             formato: str, pares_tramo: str, pares_etiquetas: str, documentos=(None, None), registros=(None, None)):
    validar(agrupa, fraccion)
    extra = {"documentos": documentos[1], "registro": registros[1]}
    if agrupa is None:
        return rec.cargador(base, tokenizador, rank, world, semilla, fraccion, ce, rafaga, formato, pares_tramo, pares_etiquetas, **extra)
    if fraccion > 0:
        documentos = repartir(documentos, rank, world, semilla)
        extra["documentos"] = documentos[1]
    ocupadas = Ocupadas(registros[0])
    interno = ag.envolver(base, tokenizador, rank, world, semilla, seq_len, *agrupa, documentos=documentos[0], registro=ocupadas)
    return rec.cargador(interno, tokenizador, rank, world, semilla, fraccion, ce, rafaga, formato, pares_tramo, pares_etiquetas,
                        ocupadas=ocupadas, **extra)


@dataclass(frozen=True)
class EquilibrioAgrupa(eq.Equilibrio):
    def recortados(self, es_bos: torch.Tensor, pareado: bool):
        ident = eq.identificadores(es_bos, pareado)
        recortado = (ident == 0) | (ident == ident[:, -1:])
        t = es_bos.size(-1)
        if pareado:
            corte = es_bos.reshape(-1, 2, t)[:, 1, 0]
            recortado = recortado | ((ident == ident[:, t - 1:t]) & corte.unsqueeze(1))
        elif qf2_cola.CARGADOR_LARGO and t % 2 == 0:
            h = t // 2
            recortado = recortado | ((ident == ident[:, h - 1:h]) & es_bos[:, h].unsqueeze(1))
        return ident, recortado

    def pesos(self, vale: torch.Tensor, es_bos: torch.Tensor, pareado: bool | None = None) -> torch.Tensor:
        pareado = eq.PAREADO if pareado is None else pareado
        if pareado and es_bos.size(0) % 2:
            raise ValueError(f"QF2_EQUILIBRIO pareado necesita un número par de filas, hay {es_bos.size(0)}")
        v = vale.to(torch.float32)
        ident, recortado = self.recortados(es_bos, pareado)
        t = es_bos.size(-1)
        largo = torch.where(recortado, float(t), eq.contar(ident))
        largo = largo.reshape(-1, t) if pareado else largo
        validos = v.sum()
        c = validos / (v / largo).sum().clamp_min(eq.MINIMO)
        w = v * ((1.0 - self.mezcla) + self.mezcla * torch.clamp(c / largo, max=self.tope))
        return w * (validos / w.sum().clamp_min(eq.MINIMO))

    def descripcion(self) -> str:
        return super().descripcion() + " agrupa_corta=1"


def equilibrio(objeto, agrupa):
    if objeto is None or agrupa is None:
        return objeto
    return EquilibrioAgrupa(objeto.mezcla, objeto.tope)
''',
    "qf2_flujo.py": r'''from __future__ import annotations

import random
from collections import deque
from fractions import Fraction

import torch

import qf2_lotes
from prepare import _document_batches, get_dist_info

FRAC = Fraction(0)
LOTES_MEDIDA = 5
BUFFER_MIN = 512


def configurar(valor: str, cola_activa: bool) -> None:
    global FRAC
    texto = valor.strip() or "0"
    FRAC = Fraction(texto)
    if not 0 <= FRAC <= 1:
        raise ValueError(f"QF2_FLUJO_FRAC must be in [0, 1], got {texto}")
    if FRAC > 0 and cola_activa:
        raise ValueError("QF2_FLUJO_FRAC is incompatible with QF2_CARGADOR_LARGO, QF2_SEQ_MAX, QF2_COLA_FRAC "
                         "and QF2_CALIENTA_LARGO")


def activo() -> bool:
    return FRAC > 0


def filas_flujo_hasta(n: int, p: Fraction | None = None) -> int:
    p = FRAC if p is None else p
    return (n * p.numerator) // p.denominator


def es_fila_flujo(n: int, p: Fraction | None = None) -> bool:
    return filas_flujo_hasta(n + 1, p) > filas_flujo_hasta(n, p)


def fase(semilla: int, rango: int, seq_len: int) -> int:
    return random.Random(semilla * 1_000_003 + rango).randrange(seq_len)


class Documentos:
    def __init__(self, tokenizer, rango: int, mundo: int, hilos: int, lote_tok: int, split: str = "train", lotes=None):
        self.lotes = _document_batches(split, rango, mundo, lote_tok) if lotes is None else lotes
        self.tokenizer = tokenizer
        self.bos = tokenizer.get_bos_token_id()
        self.hilos = hilos
        self.cola: deque[list[int]] = deque()
        self.epoch = 0

    def tomar(self, n: int) -> list[list[int]]:
        if not self.cola:
            docs, self.epoch = next(self.lotes)
            self.cola.extend(self.tokenizer.encode(docs, prepend=self.bos, num_threads=self.hilos))
        return [self.cola.popleft() for _ in range(min(n, len(self.cola)))]


class Flujo:
    def __init__(self, docs: Documentos, seq_len: int, fase_inicial: int):
        self.docs = docs
        self.seq_len = seq_len
        self.flujo: list[int] = []
        self.llenar()
        del self.flujo[:fase_inicial]

    def llenar(self) -> None:
        while len(self.flujo) < self.seq_len + 1:
            for doc in self.docs.tomar(1):
                self.flujo.extend(doc)

    def fila(self) -> list[int]:
        self.llenar()
        fila = self.flujo[:self.seq_len + 1]
        del self.flujo[:self.seq_len]
        return fila


class MejorAjuste:
    def __init__(self, docs: Documentos, seq_len: int, lote_tok: int):
        self.docs = docs
        self.capacidad = seq_len + 1
        self.lote_tok = lote_tok
        self.buffer: list[list[int]] = []

    def fila(self, destino: torch.Tensor) -> None:
        pos = 0
        while pos < self.capacidad:
            while len(self.buffer) < BUFFER_MIN:
                self.buffer.extend(self.docs.tomar(self.lote_tok))
            restante = self.capacidad - pos
            mejor, largo = -1, 0
            for i, doc in enumerate(self.buffer):
                n = len(doc)
                if n <= restante and n > largo:
                    mejor, largo = i, n
            if mejor >= 0:
                doc = self.buffer.pop(mejor)
                destino[pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                pos += len(doc)
            else:
                corto = min(range(len(self.buffer)), key=lambda i: len(self.buffer[i]))
                doc = self.buffer.pop(corto)
                destino[pos:pos + restante] = torch.tensor(doc[:restante], dtype=torch.long)
                pos += restante


def descripcion(semilla: int, seq_len: int) -> str:
    _ddp, rango, _local, _mundo = get_dist_info()
    return f"qf2_flujo: p={FRAC} ({float(FRAC):.6f}) seed={semilla} phase_rank{rango}={fase(semilla, rango, seq_len)}"


class Cargador:
    def __init__(self, tokenizer, batch_size: int, seq_len: int, device, semilla: int, avisar=print,
                 tokenizer_threads: int = 4, tokenizer_batch_size: int = 128, split: str = "train",
                 p: Fraction | None = None, fase_inicial: int | None = None, reanudable: bool = False):
        self.p = FRAC if p is None else Fraction(p)
        _ddp, rango, _local, mundo = get_dist_info()
        self.origen = (split, rango, mundo, tokenizer_batch_size)
        lotes = qf2_lotes.Lotes(*self.origen) if reanudable else None
        self.docs = Documentos(tokenizer, rango, mundo, tokenizer_threads, tokenizer_batch_size, split, lotes)
        self.fase0 = fase(semilla, rango, seq_len) if fase_inicial is None else fase_inicial
        self.flujo = Flujo(self.docs, seq_len, self.fase0) if self.p > 0 else None
        self.empaque = MejorAjuste(self.docs, seq_len, tokenizer_batch_size)
        self.bos = self.docs.bos
        self.batch_size = batch_size
        self.avisar = avisar
        dev = torch.device(device)
        self.pin = dev.type == "cuda"
        self.filas = torch.empty((batch_size, seq_len + 1), dtype=torch.long)
        self.cpu_x = torch.empty((batch_size, seq_len), dtype=torch.int32, pin_memory=self.pin)
        self.cpu_y = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=self.pin)
        self.x = torch.empty((batch_size, seq_len), dtype=torch.int32, device=dev)
        self.y = torch.empty((batch_size, seq_len), dtype=torch.long, device=dev)
        self.n = 0
        self.lotes = 0
        self.cuenta_flujo = 0
        self.cuenta_bos = 0
        self.de_flujo: list[int] = []

    def __iter__(self):
        return self

    def close(self) -> None:
        pass

    def actual(self):
        return self.x, self.y, {"epoch": self.docs.epoch, "stream_rows": tuple(self.de_flujo)}

    def __next__(self):
        p, filas, batch_size = self.p, self.filas, self.batch_size
        de_flujo = []
        for i in range(batch_size):
            if self.flujo is not None and es_fila_flujo(self.n, p):
                filas[i] = torch.tensor(self.flujo.fila(), dtype=torch.long)
                de_flujo.append(i)
            else:
                self.empaque.fila(filas[i])
            self.n += 1
        self.lotes += 1
        if self.lotes <= LOTES_MEDIDA:
            self.cuenta_flujo += len(de_flujo)
            self.cuenta_bos += int((filas[:, 0] == self.bos).sum())
            if self.lotes == LOTES_MEDIDA:
                total = LOTES_MEDIDA * batch_size
                self.avisar(f"qf2_flujo: p={float(p):.6f} rows_per_batch={batch_size} stream_rows_per_batch="
                            f"{self.cuenta_flujo / LOTES_MEDIDA:.3f} stream_frac={self.cuenta_flujo / total:.6f} "
                            f"bos_pos0_frac={self.cuenta_bos / total:.6f} batches={LOTES_MEDIDA} phase={self.fase0}")
        self.de_flujo = de_flujo
        self.cpu_x.copy_(filas[:, :-1].to(dtype=torch.int32))
        self.cpu_y.copy_(filas[:, 1:])
        self.x.copy_(self.cpu_x, non_blocking=self.pin)
        self.y.copy_(self.cpu_y, non_blocking=self.pin)
        return self.actual()

    def estado(self) -> dict:
        if not isinstance(self.docs.lotes, qf2_lotes.Lotes):
            raise RuntimeError("qf2_flujo: the loader was not built resumable")
        return {"p": str(self.p), "forma": list(self.cpu_x.shape), "lotes": self.docs.lotes.posicion(),
                "epoch": self.docs.epoch, "cola": qf2_lotes.empaquetar(self.docs.cola),
                "buffer": qf2_lotes.empaquetar(self.empaque.buffer),
                "flujo": qf2_lotes.empaquetar([self.flujo.flujo]) if self.flujo is not None else None,
                "contadores": [self.n, self.lotes, self.cuenta_flujo, self.cuenta_bos], "fase0": self.fase0,
                "de_flujo": list(self.de_flujo), "x": self.cpu_x.clone(), "y": self.cpu_y.clone()}

    def restaurar(self, e: dict):
        if e["p"] != str(self.p) or list(e["forma"]) != list(self.cpu_x.shape) or (e["flujo"] is None) != (self.flujo is None):
            raise ValueError(f"qf2_flujo: saved loader p={e['p']} rows={e['forma']} does not match p={self.p} "
                             f"rows={list(self.cpu_x.shape)}")
        self.docs.lotes = qf2_lotes.Lotes(*self.origen, posicion=e["lotes"])
        self.docs.epoch = e["epoch"]
        self.docs.cola = deque(qf2_lotes.desempaquetar(e["cola"]))
        self.empaque.buffer = qf2_lotes.desempaquetar(e["buffer"])
        if self.flujo is not None:
            self.flujo.flujo = qf2_lotes.desempaquetar(e["flujo"])[0]
        self.n, self.lotes, self.cuenta_flujo, self.cuenta_bos = e["contadores"]
        self.fase0 = e["fase0"]
        self.de_flujo = list(e["de_flujo"])
        self.cpu_x.copy_(e["x"])
        self.cpu_y.copy_(e["y"])
        self.x.copy_(self.cpu_x, non_blocking=self.pin)
        self.y.copy_(self.cpu_y, non_blocking=self.pin)
        return self.actual()


def make_flujo_dataloader(tokenizer, batch_size: int, seq_len: int, device, semilla: int, avisar=print,
                          tokenizer_threads: int = 4, tokenizer_batch_size: int = 128, split: str = "train",
                          p: Fraction | None = None, fase_inicial: int | None = None, reanudable: bool = False):
    return Cargador(tokenizer, batch_size, seq_len, device, semilla, avisar, tokenizer_threads, tokenizer_batch_size,
                    split, p, fase_inicial, reanudable)
''',
    "qf2_franjas.py": r'''"""Muon transport in K strips (QF2_COM_FRANJAS=K, default 0 = the flat slab of RepartoMuon untouched).

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
''',
    "qf2_guarda.py": r'''"""QF2_GUARDA: skip an optimizer step whose gradients are not finite on any rank.

Before the optimizer runs, every rank sums its local gradients in fp32 (one reduction per tensor), the non-finite flag
travels in one int32 all_reduce(MAX), and the host reads it. A bad step leaves weights, moments and the step counter
untouched and is not recorded as a step boundary; the data it consumed is dropped. After LIMITE consecutive bad steps the
run stops and saves the last finite weights instead of crashing. The flag is computed from step 0 so its small graphs
compile inside the excluded startup; skipping only starts at DESDE, after the budget clock has been reset.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

LIMITE = int(os.environ.get("QF2_GUARDA_LIMITE", "8"))


class Guarda:
    def __init__(self, params, desde: int, rank: int, limite: int = LIMITE):
        self.params = [p for p in params if p.requires_grad]
        self.desde, self.rank, self.limite = desde, rank, limite
        self.saltos = 0
        self.seguidos = 0

    def descripcion(self) -> str:
        return f"qf2_guarda: tensores={len(self.params)} desde_paso={self.desde} limite_seguidos={self.limite}"

    def no_finito(self) -> torch.Tensor:
        sumas = [p.grad.sum(dtype=torch.float32) for p in self.params if p.grad is not None]
        return (~torch.isfinite(torch.stack(sumas).sum())).to(torch.int32).reshape(1)

    def saltar(self, paso: int) -> bool:
        malo = self.no_finito()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(malo, op=dist.ReduceOp.MAX)
        if int(malo.item()) == 0 or paso < self.desde:
            self.seguidos = 0
            return False
        self.saltos += 1
        self.seguidos += 1
        if self.rank == 0:
            print(f"qf2_guarda: salto paso={paso} saltos={self.saltos} seguidos={self.seguidos}", flush=True)
        return True

    def agotada(self) -> bool:
        return self.seguidos >= self.limite
''',
    "qf2_guardado_anticipado.py": r'''"""QF2_MEMCORPUS_ANTICIPADO: write the corpus tables into out/final.pt as soon as they stop changing, finish the file at the end.

Saving 5 x 2^34 tables (344 GB) took 644 s on the block 3 host (gp3 root volume, ~0.11 GB/s of writeback): past the
dirty-page threshold the kernel throttles the writer to disk speed. The tables are final once the builder stops, hours
before the end of the run, so rank 0 opens out/final.pt with torch's own zip writer, writes the table storages as the
first records from a background thread (the writer releases the GIL) and leaves the kernel the rest of the run to flush
them. At the end torch.serialization._save pickles the full payload into the same writer; the table storages, already
written under the keys the pickler gives them (tables first in model_state, so keys 0..n-1), are skipped and every other
record is written as torch.save would. The result is a standard torch zip: torch.load (with or without mmap) and
prepare.checkpoint_size_report read it like any other final.pt.
"""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

import torch
import torch.serialization as ts


class _Zip:
    def __init__(self, escritor: torch._C.PyTorchFileWriter, previos: dict):
        self.escritor = escritor
        self.previos = previos
        self.saltados = []

    def write_record(self, nombre, datos, n):
        if nombre in self.previos:
            ptr, nbytes = self.previos[nombre]
            if not isinstance(datos, torch.UntypedStorage) or datos.data_ptr() != ptr or int(n) != nbytes:
                raise RuntimeError(f"guardado anticipado: {nombre} no es la tabla escrita antes")
            self.saltados.append(nombre)
            return
        self.escritor.write_record(nombre, datos, n)

    def write_record_metadata(self, nombre, n):
        raise RuntimeError("guardado anticipado: skip_data no está soportado")


class Anticipado:
    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.escritor = None
        self.previos: dict = {}
        self.nombres: list = []
        self.hilo: threading.Thread | None = None
        self.error: BaseException | None = None
        self.segundos = 0.0

    @property
    def empezado(self) -> bool:
        return self.escritor is not None

    def empezar(self, tablas: dict) -> None:
        if self.empezado:
            raise RuntimeError("guardado anticipado ya empezado")
        self.escritor = torch._C.PyTorchFileWriter(str(self.ruta), True, ts._get_storage_alignment())
        self.nombres = list(tablas)
        trabajo = []
        for i, t in enumerate(tablas.values()):
            s = t.untyped_storage()
            if t.storage_offset() != 0 or s.nbytes() != t.numel() * t.element_size():
                raise RuntimeError("guardado anticipado: cada tabla debe ocupar su almacenamiento entero")
            self.previos[f"data/{i}"] = (s.data_ptr(), s.nbytes())
            trabajo.append((f"data/{i}", s))
        self.hilo = threading.Thread(target=self._escribir, args=(trabajo,), name="qf2_anticipado", daemon=True)
        self.hilo.start()

    def _escribir(self, trabajo) -> None:
        t0 = time.monotonic()
        try:
            for nombre, s in trabajo:
                self.escritor.write_record(nombre, s, s.nbytes())
        except BaseException as ex:
            self.error = ex
        self.segundos = time.monotonic() - t0

    def listo(self) -> bool:
        return self.empezado and self.hilo is not None and not self.hilo.is_alive()

    def terminar(self, payload: dict) -> list:
        if not self.empezado:
            raise RuntimeError("guardado anticipado sin empezar")
        self.hilo.join()
        if self.error is not None:
            raise RuntimeError(f"guardado anticipado: falló la escritura de las tablas: {self.error!r}")
        zip_ = _Zip(self.escritor, self.previos)
        ts._save(payload, zip_, pickle, ts.DEFAULT_PROTOCOL, False)
        if sorted(zip_.saltados) != sorted(self.previos):
            raise RuntimeError(f"guardado anticipado: claves escritas {sorted(self.previos)} y saltadas {sorted(zip_.saltados)} no coinciden")
        self.escritor.write_end_of_file()
        return zip_.saltados


    def ordenar(self, estado: dict) -> dict:
        faltan = [k for k in self.nombres if k not in estado]
        if faltan:
            raise RuntimeError(f"guardado anticipado: faltan las tablas {faltan} en el estado final")
        return {**{k: estado[k] for k in self.nombres}, **{k: v for k, v in estado.items() if k not in self.nombres}}
''',
    "qf2_hilos_aten.py": r'''"""QF2_HILOS_ATEN and QF2_HILOS_HILO: fix the ATen intra-op thread count inside train.py instead of trusting the launcher.

The block queues export OMP_NUM_THREADS=4 to the 64 ranks, while torchrun sets OMP_NUM_THREADS=1 when the variable is
absent, so the official evaluator may train with another thread count than our local runs [SIN VERIFICAR]. The MEMHOST
table thread does its gathers with index_select, which ATen parallelises over that count; 64 ranks x 4 threads exceed the
192 vCPU of the host. ATEN > 0 calls torch.set_num_threads(ATEN) in the main thread at the start of main(); HILO > 0 calls
it again from the table thread when it starts. With OpenMP the count of the calling thread changes and threads created
later inherit the last value, while threads already running keep theirs. 0 (default) leaves everything as the launcher
set it. train.py sets ATEN and HILO from the environment, so the values are baked with the other flags.
"""
from __future__ import annotations

import os

import torch

ATEN = 0
HILO = 0


def _fijar(n: int, nombre: str) -> int:
    if n <= 0:
        return 0
    torch.set_num_threads(n)
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_hilos_aten: {nombre} hilos={torch.get_num_threads()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '-')}", flush=True)
    return n


def fijar_proceso() -> int:
    return _fijar(ATEN, "proceso")


def fijar_hilo(nombre: str) -> int:
    return _fijar(HILO, nombre)
''',
    "qf2_hilos_nice.py": r'''"""QF2_HILOS_NICE: lower the CPU priority of a background thread (the MEMHOST table thread and the loader prefetch).

With 64 ranks the table threads spend ~0.9-1.2 s of CPU per step in the first ~1.4 s of it, and the 20p6 profile of the
122 shows the device idle +285 ms per step against the 106 in exactly that window (the last micro-step, after the table
work, matches the 106). Linux keeps the nice value per thread, so os.setpriority on the native thread id only affects the
calling thread; raising it needs no privilege. No tensor, order or value changes: only the scheduler weight. 0 (default)
leaves the thread alone. train.py sets NICE from QF2_HILOS_NICE (so the value is baked with the other flags).
"""
from __future__ import annotations

import os
import threading

NICE = 0


def ceder(nombre: str) -> int:
    n = NICE
    if n <= 0:
        return 0
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), n)
    except (OSError, AttributeError) as ex:
        print(f"qf2_hilos_nice: {nombre} sin nice ({ex})", flush=True)
        return 0
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_hilos_nice: {nombre} nice={os.getpriority(os.PRIO_PROCESS, threading.get_native_id())}", flush=True)
    return n
''',
    "qf2_inicio.py": r'''"""QF2_INICIO_DESDE y QF2_BANCO_LRM: continuar una corrida desde un final.pt del kit (banco CORE bifurcado).

abrir() mapea el final.pt con torch.load(mmap=True) (qf2_memoria_anfitrion.abrir, la lectura de la evaluación) y exige
que su model_config sea el del modelo construido. cargar_tronco() carga en cada rango el model_state sin las tablas del
anfitrión con el mismo filtro estricto que load_for_eval_memhost (qf2_memhost_ckpt.cargar_resto), sobre el modelo de CPU,
antes de model.to(device) y de crear el optimizador, que empieza nuevo.

La única diferencia de model_config que abrir() admite es la vía del sitio 11 (QF2_MEMHOST_VIA_SITIO11): un modelo con
via=v11 desde un final.pt con la vía res (el 274 o el 293). Las claves son las mismas, porque memhost_sitios.key.11 y
val.11 tienen la misma forma [n_embd, S*W] en las dos vías: la puerta y la proyección que llevaban lo recuperado al
residual tras el bloque 11 pasan a llevarlo al valor de la atención del bloque 11, el tronco carga entero y nada empieza
a cero. Cualquier otra diferencia, también v11 -> res o res -> v, sigue siendo un error.

tablas() da el gancho que qf2_memhost_hilo.Memhost llama con su Tabla antes de arrancar el hilo de tablas, así que ni un
gather ni su plazo ESPERA_S (300 s) corren durante la carga. cargar_tablas() escribe en los maestros fp32 compartidos del
anfitrión las filas propias del rango (Tabla.propias) descuantizadas de memhost.q.weight y memhost.s.weight, pone a cero
acc y cnt (y m, v con el optimizador Adam) en esas filas, deja una marca y espera la de todos los rangos: ningún gather
del primer lote lee una fila a medio cargar. Las lecturas del entreno aplican ida_vuelta a los maestros, que sobre valores
ya descuantizados es la identidad, así que el primer paso ve exactamente lo que la evaluación lee del checkpoint.

leer_lrm('a:b') y multiplicador(): el multiplicador de LR baja en línea recta de a a b con el progreso de la corrida
(reloj_progress con QF2_LR_SCHEDULE=reloj, paso / pasos objetivo con steps), sin calentamiento; vacío deja el calendario.

shards_banco(): con QF2_BANCO_SHARDS_DESDE=d (train.py lo deja en SHARDS_DESDE), los ids de los shards de entreno presentes en disco con id >= d, para que el
cargador base y los documentos de recolocado de todos los brazos lean datos que la corrida fuente no vio; vacío da None
y el cargador del kit queda como estaba.
"""
from __future__ import annotations

import json
import math
import time

import torch

import qf2_memhost_cfg as cfgm
import qf2_memhost_ckpt as ckptm
import qf2_memhost_cuantiza as cuant
import qf2_memhost_sincro as sincro
import qf2_memoria_anfitrion as anfitrion

TROZO_FILAS = 1 << 18
PLAZO_S = 1800.0


def leer_lrm(texto: str) -> tuple[float, float] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    try:
        valores = [float(p) for p in partes]
    except ValueError:
        valores = []
    if len(partes) != 2 or len(valores) != 2 or not all(math.isfinite(v) and v >= 0 for v in valores):
        raise ValueError(f"QF2_BANCO_LRM={texto!r}: se espera 'a:b' con a y b finitos y >= 0")
    return valores[0], valores[1]


def multiplicador(tramo: tuple[float, float], progreso: float) -> float:
    a, b = tramo
    return a + (b - a) * min(1.0, max(0.0, progreso))


VIA_V11 = ";via=v11"


def via_guardada(config: dict) -> dict:
    texto = config.get("memhost") or ""
    return {**config, "memhost": texto[:-len(VIA_V11)]} if texto.endswith(VIA_V11) else config


def abrir(ruta: str, config: dict) -> dict:
    payload = anfitrion.abrir(ruta)
    propio = payload.get("model_config") or {}
    if json.dumps(propio, sort_keys=True) not in (json.dumps(config, sort_keys=True), json.dumps(via_guardada(config), sort_keys=True)):
        distintas = sorted(k for k in set(propio) | set(config) if propio.get(k) != config.get(k))
        raise ValueError(f"QF2_INICIO_DESDE={ruta}: model_config distinto del modelo construido en {distintas}")
    return payload


def cargar_tronco(model: torch.nn.Module, payload: dict, tramo: tuple[float, float] | None) -> str:
    estado = payload["model_state"]
    ckptm.cargar_resto(model, estado)
    propias = set(ckptm.claves())
    tensores = [v for k, v in estado.items() if k not in propias]
    lrm = f"{tramo[0]:g}:{tramo[1]:g} lineal sobre el progreso" if tramo is not None else "el del calendario"
    propia = getattr(model, "memhost_cfg", None)
    guardada = (payload.get("model_config") or {}).get("memhost") or ""
    via = ("; sitio 11 de la vía res a la v con memhost_sitios.key.11 y val.11 del final.pt"
           if propia is not None and propia.via == "v11" and ";via=" not in guardada else "")
    return (f"qf2_inicio: tronco de {len(tensores)} tensores ({sum(v.numel() for v in tensores):,} valores) del final.pt de "
            f"step={payload.get('meta', {}).get('step')}; optimizador nuevo; lrm {lrm}{via}")


def tablas(payload: dict | None, imprimir):
    if payload is None:
        return None
    return lambda tabla: imprimir(cargar_tablas(tabla, payload))


def cargar_tablas(tabla, payload: dict, plazo: float = PLAZO_S) -> str:
    estado = payload["model_state"]
    q8, s16 = estado[cfgm.CLAVE_Q], estado[cfgm.CLAVE_S]
    if tuple(q8.shape) != (tabla.r, tabla.w) or q8.dtype != torch.int8 or tuple(s16.shape) != (tabla.r,) or s16.dtype != torch.float16:
        raise RuntimeError(f"qf2_inicio: tablas con forma/dtype inesperados q={tuple(q8.shape)} {q8.dtype} s={tuple(s16.shape)} "
                           f"{s16.dtype}, la tabla es ({tabla.r}, {tabla.w})")
    t0 = time.monotonic()
    lo, hi = tabla.propias
    for a in range(lo, hi, TROZO_FILAS):
        b = min(hi, a + TROZO_FILAS)
        tabla.q[a:b] = cuant.descuantizar(q8[a:b], s16[a:b])
    for t in (tabla.acc, tabla.cnt, tabla.m, tabla.v):
        if t is not None:
            t[lo:hi].zero_()
    cargado = time.monotonic() - t0
    (tabla.ruta / f"inicio.{tabla.rango}").write_text(tabla.id)
    for r in range(tabla.mundo):
        sincro.esperar_marca(tabla.ruta / f"inicio.{r}", tabla.id, plazo)
    return (f"qf2_inicio: tablas memhost filas [{lo:,}, {hi:,}) de {tabla.r:,} descuantizadas a los maestros del anfitrión en "
            f"{cargado:.1f} s, acc y cnt a cero; {tabla.mundo} rangos listos a los {time.monotonic() - t0:.1f} s")


SHARDS_DESDE = ""


def shards_banco():
    desde = SHARDS_DESDE
    if not desde:
        return None
    import prepare
    ids = [i for i in prepare.PUBLIC_TRAIN_SHARDS if i >= int(desde) and (prepare.DATA_DIR / prepare.shard_filename(i)).exists()]
    if not ids:
        raise RuntimeError(f"QF2_BANCO_SHARDS_DESDE={desde}: ningún shard de entreno en disco")
    return ids
''',
    "qf2_lotes.py": r'''"""Resumable document batches for QF2_PUNTOS_S / QF2_REANUDA (default OFF).

Lotes yields exactly what prepare._document_batches yields (same row groups, same slices, same epoch counter) and exposes
its position (epoch, row group, offset), so a loader restarts from a saved point without replaying the data it consumed.
empaquetar / desempaquetar turn lists of token lists into two int tensors for torch.save(weights_only) and back.
"""
from __future__ import annotations

import pyarrow.parquet as pq
import torch

from prepare import _row_groups_for_split


class Lotes:
    def __init__(self, split: str, inicio: int, paso: int, lote: int, posicion=(1, 0, 0)):
        self.grupos = _row_groups_for_split(split, inicio, paso)
        self.lote = lote
        self.epoch, self.grupo, self.desde = (int(v) for v in posicion)
        if not 0 <= self.grupo < len(self.grupos):
            raise ValueError(f"qf2_lotes: row group {self.grupo} out of range ({len(self.grupos)} groups)")
        self.filas = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.filas is None:
                ruta, rg = self.grupos[self.grupo]
                self.filas = pq.ParquetFile(ruta).read_row_group(rg).column("text").to_pylist()
            if self.desde < len(self.filas):
                docs = self.filas[self.desde:self.desde + self.lote]
                self.desde += self.lote
                return docs, self.epoch
            self.filas, self.desde, self.grupo = None, 0, self.grupo + 1
            if self.grupo == len(self.grupos):
                self.grupo, self.epoch = 0, self.epoch + 1

    def posicion(self) -> list[int]:
        return [self.epoch, self.grupo, self.desde]


def empaquetar(listas) -> dict:
    largos = torch.tensor([len(l) for l in listas], dtype=torch.int64)
    plano = torch.tensor([t for l in listas for t in l], dtype=torch.int64)
    return {"largos": largos, "plano": plano}


def desempaquetar(d: dict) -> list[list[int]]:
    plano, salida, i = d["plano"].tolist(), [], 0
    for n in d["largos"].tolist():
        salida.append(plano[i:i + n])
        i += n
    return salida
''',
    "qf2_memhost_canon.py": r'''"""QF2_MEMHOST_CANON: tokenizer compression for the hash, derived only from the provided tokenizer.

Tokens whose decoded text is equal after NFKC, lower-casing and stripping edge whitespace and punctuation share one
representative id (the smallest id of the class). BOS and tokens whose normalised text is empty map to themselves. The
map travels in the checkpoint as the int32 buffer memhost.canon.
"""
from __future__ import annotations

import string
import unicodedata

import numpy as np

BORDES = string.punctuation + string.whitespace


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKC", texto).lower().strip(BORDES)


def construir(decode, vocab: int, bos: int) -> np.ndarray:
    canon = np.arange(vocab, dtype=np.int32)
    rep: dict = {}
    for t in range(vocab):
        if t == bos:
            continue
        try:
            clave = normalizar(decode([t]))
        except Exception:
            continue
        if clave:
            canon[t] = rep.setdefault(clave, t)
    return canon
''',
    "qf2_memhost_cfg.py": r'''"""QF2_MEMHOST: configuration of the n-gram memory tables that live in the host RAM.

The configuration travels in the checkpoint as one string (GPTConfig.memhost) so that evaluation rebuilds exactly the
layout used in training: width W of one row, one (order, rows) entry per hash head, the injection sites, the sign trick
the tokenizer compression flag and the gate function of the gated sites (absent from the string when it is the default
sigmoid, so the specs written before the option keep their exact text) and the route of the gated sites (via, absent when
it is the default res): res adds a gated site to the residual after its block, v sends every gated site into the
attention values of its block (mem_v, before the SDPA) and v11 sends only site 11 into the values of block 11 while the
other sites stay residual (Cfg.via_de). A head with F rows owns rows [base, base + F) of the global table; its hash modulus
is the largest prime below F and the row base + prime is the sentinel for positions without enough history.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

CABEZAS_BASE = "2x2:128000000,3x4:240000000"
SITIOS_BASE = "in,1,11"
ANCHO_BASE = 64
PUERTAS = ("sig", "firm", "rec")
VIAS = ("res", "v", "v11")
SITIO_V11 = 11
CLAVE_Q = "memhost.q.weight"
CLAVE_S = "memhost.s.weight"
CLAVE_CANON = "memhost.canon"


@dataclass(frozen=True)
class Cfg:
    ancho: int
    cabezas: tuple
    sitios: tuple
    signos: bool
    canon: bool
    puerta: str = "sig"
    via: str = "res"

    @property
    def s(self) -> int:
        return len(self.cabezas)

    @property
    def ancho_e(self) -> int:
        return self.s * self.ancho

    @property
    def filas(self) -> int:
        return sum(f for _o, f in self.cabezas)

    @property
    def bases(self) -> tuple:
        out, acc = [], 0
        for _o, f in self.cabezas:
            out.append(acc)
            acc += f
        return tuple(out)

    @property
    def primos(self) -> tuple:
        usados, out = set(), []
        for _o, f in self.cabezas:
            p = primo_bajo(f)
            while p in usados:
                p = primo_bajo(p)
            usados.add(p)
            out.append(p)
        return tuple(out)

    @property
    def gated(self) -> tuple:
        return tuple(int(s) for s in self.sitios if s != "in")

    @property
    def entrada(self) -> bool:
        return "in" in self.sitios

    def via_de(self, sitio: int) -> str:
        if self.via == "v11":
            return "v" if sitio == SITIO_V11 else "res"
        return self.via


def es_primo(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@lru_cache(maxsize=None)
def primo_bajo(filas: int) -> int:
    n = filas - 1
    while n > 2 and not es_primo(n):
        n -= 1
    return n


def parse_cabezas(texto: str) -> tuple:
    out = []
    for grupo in texto.split(","):
        ob, filas = grupo.split(":")
        orden, n = ob.split("x")
        if int(orden) not in (2, 3):
            raise ValueError(f"orden {orden} no soportado (2 o 3)")
        out.extend((int(orden), int(filas)) for _ in range(int(n)))
    if not out or any(f < 8 for _o, f in out):
        raise ValueError(f"cabezas invalidas: {texto!r}")
    return tuple(out)


def desde_texto(spec: str) -> Cfg:
    d = dict(p.split("=", 1) for p in spec.split(";") if p)
    sitios = tuple(s for s in d.get("sitios", SITIOS_BASE).split(",") if s)
    if "in" not in sitios:
        raise ValueError(f"sitios {sitios} sin 'in': tabla y Linear_val empiezan a cero y ninguna recibe gradiente")
    puerta = d.get("puerta", "sig")
    if puerta not in PUERTAS:
        raise ValueError(f"puerta {puerta!r} fuera de {PUERTAS}")
    via = d.get("via", "res")
    if via not in VIAS:
        raise ValueError(f"via {via!r} fuera de {VIAS}")
    if via == "v11" and str(SITIO_V11) not in sitios:
        raise ValueError(f"via v11 sin el sitio {SITIO_V11} en {sitios}")
    return Cfg(int(d.get("ancho", ANCHO_BASE)), parse_cabezas(d.get("cab", CABEZAS_BASE)), sitios,
               d.get("signos", "1") != "0", d.get("canon", "0") != "0", puerta, via)


def a_texto(cfg: Cfg) -> str:
    cab: dict = {}
    for o, f in cfg.cabezas:
        cab[(o, f)] = cab.get((o, f), 0) + 1
    grupos = ",".join(f"{o}x{n}:{f}" for (o, f), n in cab.items())
    puerta = "" if cfg.puerta == "sig" else f";puerta={cfg.puerta}"
    via = "" if cfg.via == "res" else f";via={cfg.via}"
    return f"ancho={cfg.ancho};cab={grupos};sitios={','.join(cfg.sitios)};signos={int(cfg.signos)};canon={int(cfg.canon)}{puerta}{via}"


def desde_banderas(activo: str, ancho: str, cabezas: str, sitios: str, signos: str, canon: str,
                   puerta: str = "sig", via: str = "res") -> str:
    if activo == "0":
        return ""
    return a_texto(desde_texto(f"ancho={ancho};cab={cabezas};sitios={sitios};signos={signos};canon={canon};puerta={puerta};via={via}"))


def via_sitio11(via: str, sitio11: str) -> str:
    sitio11 = (sitio11 or "").strip()
    if not sitio11:
        return via
    if sitio11 != "v" or via != "res":
        raise ValueError(f"QF2_MEMHOST_VIA_SITIO11={sitio11!r} con QF2_MEMHOST_VIA={via!r}: se admite '' o 'v' sobre la via res")
    return "v11"
''',
    "qf2_memhost_ckpt.py": r'''"""QF2_MEMHOST: checkpoint writing and evaluation reads for the host tables.

Writing: after training every owner quantises its row range of the masters into two shared files (int8 rows and fp16
scales); rank 0 saves them as memhost.q.weight and memhost.s.weight. Both are Parameters of the evaluation module, so
the 1 GiB allowance for non-parameter bytes is not touched. Reading: the checkpoint is opened with mmap, the dense part
is loaded without the tables, the tables are assigned as Parameters over the mapped file and queried on the CPU outside
the graph with the same hash as in training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import qf2_memhost_cfg as cfgm
import qf2_memhost_cuantiza as cuant
import qf2_memhost_hash as hashm
import qf2_memhost_sincro as sincro

TROZO_FILAS = 1 << 20


def cuantizar_propias(tabla, epoca: int) -> dict:
    ruta = tabla.ruta
    crear = tabla.rango == 0
    r, w = tabla.r, tabla.w
    q8 = sincro.mapear(ruta / "qi8.i8", torch.int8, r * w, crear).view(r, w) if crear else None
    s16 = sincro.mapear(ruta / "esc.f16", torch.float16, r, crear) if crear else None
    marca = ruta / "q8listo"
    if crear:
        marca.write_text("listo")
    else:
        sincro.esperar_marca(marca)
        q8 = sincro.mapear(ruta / "qi8.i8", torch.int8, r * w, False).view(r, w)
        s16 = sincro.mapear(ruta / "esc.f16", torch.float16, r, False)
    lo, hi = tabla.propias
    for a in range(lo, hi, TROZO_FILAS):
        b = min(hi, a + TROZO_FILAS)
        q, s = cuant.cuantizar(tabla.q[a:b])
        q8[a:b] = q
        s16[a:b] = s
    tabla.barrera.pasar(epoca, "cuantizar")
    return {cfgm.CLAVE_Q: q8, cfgm.CLAVE_S: s16}


def vacias(cfg) -> dict:
    return {cfgm.CLAVE_Q: torch.zeros(cfg.filas, cfg.ancho, dtype=torch.int8), cfgm.CLAVE_S: torch.zeros(cfg.filas, dtype=torch.float16)}


def claves() -> list:
    return [cfgm.CLAVE_Q, cfgm.CLAVE_S]


def usa_memhost(payload: dict) -> bool:
    return bool(payload["model_config"].get("memhost"))


def cargar_resto(model: nn.Module, state: dict) -> None:
    propias = set(claves())
    res = model.load_state_dict({k: v for k, v in state.items() if k not in propias}, strict=False)
    if res.unexpected_keys or res.missing_keys:
        raise RuntimeError(f"claves inesperadas al cargar sin tablas: faltan={res.missing_keys} sobran={res.unexpected_keys}")


def asignar(model: nn.Module, state: dict) -> None:
    cfg = model.memhost_cfg
    q, s = state[cfgm.CLAVE_Q], state[cfgm.CLAVE_S]
    if tuple(q.shape) != (cfg.filas, cfg.ancho) or q.dtype != torch.int8 or tuple(s.shape) != (cfg.filas,) or s.dtype != torch.float16:
        raise RuntimeError(f"tablas memhost con forma/dtype inesperados: q={tuple(q.shape)} {q.dtype} s={tuple(s.shape)} {s.dtype}")
    model.memhost.q = nn.Module()
    model.memhost.s = nn.Module()
    model.memhost.q.weight = nn.Parameter(q, requires_grad=False)
    model.memhost.s.weight = nn.Parameter(s, requires_grad=False)


@torch.compiler.disable
def consultar(model: nn.Module, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cfg = model.memhost_cfg
    canon = model.memhost.canon.detach().cpu().numpy() if cfg.canon else None
    ids, sig = hashm.indices(idx.detach().cpu().numpy(), cfg, model.memhost_bos, canon)
    flat = torch.from_numpy(ids.reshape(-1))
    q = model.memhost.q.weight
    s = model.memhost.s.weight
    filas = cuant.descuantizar(torch.index_select(q, 0, flat), torch.index_select(s, 0, flat))
    if cfg.signos:
        filas = filas * torch.from_numpy(hashm.signos(sig.reshape(-1), cfg.ancho))
    b, t = idx.shape
    return filas.view(b, t, cfg.ancho_e).to(dtype).to(idx.device)
''',
    "qf2_memhost_cuantiza.py": r'''"""QF2_MEMHOST: per-row int8 quantisation with an fp16 scale, shared by training reads and the checkpoint.

scale = fp16(amax / 127) with a floor of 2^-14. After the fp16 rounding amax / scale lies in [126.9, 127.1], so q reaches 127
and |w - dq(q(w))| <= scale / 2 for every element (the clamp loses at most 0.07 scale); re-quantising dq(q(w)) returns the
same scale and the same q. The training gather applies ida_vuelta to the fp32 masters,
so the values the graph sees in the last step are bit for bit those evaluation reads back from the checkpoint.
"""
from __future__ import annotations

import torch

SUELO = 2.0 ** -14


def escala(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax(dim=-1)
    return (amax / 127.0).clamp_min(SUELO).to(torch.float16)


def cuantizar(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s16 = escala(w)
    q = torch.round(w / s16.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q, s16


def descuantizar(q: torch.Tensor, s16: torch.Tensor) -> torch.Tensor:
    return q.float() * s16.float().unsqueeze(-1)


def ida_vuelta(w: torch.Tensor) -> torch.Tensor:
    q, s16 = cuantizar(w)
    return descuantizar(q, s16)
''',
    "qf2_memhost_hash.py": r'''"""QF2_MEMHOST: causal 64-bit n-gram hash, the only implementation (data loader, table thread and evaluation import it).

For every position the ids of the S heads depend only on the tokens up to that position. Validity follows
qf2_ngramas.indices: the bigram needs one earlier position and a current token that is not BOS; the trigram also needs
a second earlier position and a first earlier token that is not BOS. Invalid references map to the sentinel row of the
head. Arithmetic is uint64 with wrap-around in numpy; there is no int32 anywhere.
"""
from __future__ import annotations

import numpy as np

import qf2_memhost_cfg as cfgm

SEMILLA = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S33 = np.uint64(33)


def splitmix(x: int) -> int:
    x = (x + SEMILLA) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def multiplicadores(cabeza: int) -> tuple:
    return tuple(np.uint64(splitmix(1000 * (cabeza + 1) + j) | 1) for j in range(3))


def sal(cabeza: int) -> np.uint64:
    return np.uint64(splitmix(7777 + cabeza))


def mezcla64(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> S33)
    x = x * C1
    x = x ^ (x >> S33)
    x = x * C2
    return x ^ (x >> S33)


def desplazar(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a)
    out[:, k:] = a[:, :a.shape[1] - k]
    return out


def canon_identidad(vocab: int) -> np.ndarray:
    return np.arange(vocab, dtype=np.int64)


def indices(tokens, cfg: cfgm.Cfg, bos: int, canon=None) -> tuple:
    t = np.asarray(tokens).astype(np.int64)
    if t.ndim != 2:
        raise ValueError("tokens debe ser [B, T]")
    c = (np.asarray(canon).astype(np.int64)[t] if canon is not None else t).astype(np.uint64)
    pos = np.arange(t.shape[1])[None, :]
    vale2 = (t != bos) & (pos >= 1)
    vale3 = vale2 & (desplazar(t, 1) != bos) & (pos >= 2)
    c1, c2 = desplazar(c, 1), desplazar(c, 2)
    bases, primos = cfg.bases, cfg.primos
    ids = np.empty(t.shape + (cfg.s,), dtype=np.int64)
    sig = np.empty(t.shape + (cfg.s,), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for h, (orden, _f) in enumerate(cfg.cabezas):
            ma, mb, mc = multiplicadores(h)
            mix = (c * ma) ^ (c1 * mb)
            valido = vale2
            if orden == 3:
                mix = mix ^ (c2 * mc)
                valido = vale3
            mix = mezcla64(mix)
            fila = (mix % np.uint64(primos[h])).astype(np.int64) + bases[h]
            ids[..., h] = np.where(valido, fila, bases[h] + primos[h])
            sig[..., h] = np.where(valido, mezcla64(mix ^ sal(h)), np.uint64(0))
    return ids, sig


def signos(sig: np.ndarray, ancho: int) -> np.ndarray:
    bits = (sig[..., None] >> (np.arange(ancho, dtype=np.uint64) % np.uint64(64))) & np.uint64(1)
    return bits.astype(np.float32) * 2.0 - 1.0
''',
    "qf2_memhost_hilo.py": r'''"""QF2_MEMHOST: the table thread of one rank.

Per step e: aggregate the posted gradient, exchange it (barrier A), update the owned rows (barrier B), then gather batch
e + RETRASO + 1 (quantised, signed). A skipped step still crosses both barriers. Failures are raised by revisar().
qf2_memhost_rapido covers the rms path bit for bit; adam and other configurations use the reference code here.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict

import numpy as np
import torch

import qf2_hilos_aten as aten
import qf2_hilos_nice as nice
import qf2_memhost_cfg as cfgm
import qf2_memhost_hash as hashm
import qf2_memhost_intercambio as inter
import qf2_memhost_opt as optm
import qf2_memhost_rapido as rap
import qf2_memhost_tabla as tab
import qf2_memhost_traza as traza

ESPERA_S = 300.0


class Memhost:
    def __init__(self, cfg: cfgm.Cfg, ruta: str, rango: int, mundo: int, forma: tuple, dtype: torch.dtype, bos: int,
                 retraso: int = 1, modo: str = "rms", canon=None, inicio=None):
        self.cfg, self.rango, self.mundo, self.dtype, self.bos, self.canon = cfg, rango, mundo, dtype, bos, canon
        self.b, self.t = forma
        self.retraso, self.modo = retraso, modo
        self.tabla = tab.Tabla(cfg, ruta, rango, mundo, adam=modo == "adam")
        self.inter = inter.Intercambio(self.tabla.ruta, rango, mundo, self.b * self.t * cfg.s, cfg.ancho, rango == 0)
        self.limites = self.tabla.limites()
        self.rapido = rap.Estado(cfg, self.tabla, self.inter, rango, retraso) if rap.aplica(cfg, modo, dtype) else None
        self.cond = threading.Condition()
        self.entradas: dict = {}
        self.listos: dict = {}
        self.jobs: queue.Queue = queue.Queue()
        self.cierre = threading.Event()
        self.error: BaseException | None = None
        self.aplicadas = 0
        self.pasos_opt = 0
        self.vistas: dict = {}
        self.esperas = 0
        self.servidos = 0
        self.pasos = 0
        self.tiempos: dict = defaultdict(float)
        if inicio is not None:
            inicio(self.tabla)
        self.tabla.barrera.pasar(0, "arranque")
        self.hilo = threading.Thread(target=self._correr, name="qf2_memhost", daemon=True)
        self.hilo.start()

    def indices(self, x) -> tuple:
        ids, sig = hashm.indices(x.cpu().numpy() if hasattr(x, "cpu") else x, self.cfg, self.bos, self.canon)
        return torch.from_numpy(ids.reshape(-1)), torch.from_numpy(sig.reshape(-1).view(np.int64))

    def registrar(self, k: int, x) -> None:
        ids, sig = self.indices(x)
        with self.cond:
            self.entradas[k] = (ids, sig)
            self.cond.notify_all()

    def _entrada(self, k: int):
        with self.cond:
            t0 = time.monotonic()
            while k not in self.entradas:
                if self.cierre.is_set():
                    return None
                if time.monotonic() - t0 > ESPERA_S:
                    raise TimeoutError(f"qf2_memhost: sin ids del lote {k} en {ESPERA_S:.0f} s")
                self.cond.wait(0.5)
            return self.entradas[k]

    def _signos(self, sig: torch.Tensor):
        if not self.cfg.signos:
            return None
        return torch.from_numpy(hashm.signos(sig.numpy().view(np.uint64), self.cfg.ancho)).reshape(-1, self.cfg.ancho)

    def _reunir_lote(self, k: int) -> None:
        ent = self._entrada(k)
        if ent is None:
            return
        ids, sig = ent
        t0 = time.perf_counter()
        if self.rapido is not None:
            e = rap.reunir_lote(self.rapido, ent, self.b, self.t)
        else:
            filas = self.tabla.leer(ids)
            sg = self._signos(sig)
            if sg is not None:
                filas = filas * sg
            e = filas.view(self.b, self.t, self.cfg.ancho_e).to(self.dtype)
        self.tiempos["gather"] += time.perf_counter() - t0
        with self.cond:
            self.listos[k] = e
            self.vistas[k] = self.aplicadas
            self.cond.notify_all()

    def _actualizar(self, e: int, grad, lr: float, lrm: float = 1.0) -> None:
        ent = self._entrada(e)
        if ent is None:
            raise RuntimeError(f"qf2_memhost: faltan los ids del lote {e} al actualizar")
        ids, sig = ent
        t0 = time.perf_counter()
        if grad is None:
            uid, g = torch.empty(0, dtype=torch.int64), torch.empty(0, self.cfg.ancho)
        elif self.rapido is not None:
            uid, g = rap.agregar(self.rapido, ent, grad)
        else:
            gf = torch.nan_to_num(grad.float().reshape(-1, self.cfg.ancho))
            sg = self._signos(sig)
            if sg is not None:
                gf = gf * sg
            uid, g = inter.agregar(ids, gf / self.mundo)
        t1 = time.perf_counter()
        rap.publicar(self.rapido, uid, self.limites) if self.rapido else self.inter.publicar(uid, g, self.limites)
        t2 = time.perf_counter()
        self.tabla.barrera.pasar(2 * e + 1, "A")
        t3 = time.perf_counter()
        pid, pg = rap.reunir(self.rapido) if self.rapido is not None else self.inter.reunir(self.rango)
        t4 = time.perf_counter()
        if grad is not None:
            self.pasos_opt += 1
            cabs = traza.activa(self.pasos_opt, self.rango, self.cfg, self.mundo) if self.modo == "rms" else ()
            est = traza.previa(self.tabla, pid, pg, self.cfg, cabs) if cabs else None
            rap.optimizar(self.rapido, pid, pg, lr, lrm) if self.rapido else optm.aplicar(self.tabla, self.modo, pid, pg, lr, self.pasos_opt, lrm)
            if est is not None:
                print("\n".join(traza.lineas(est, self.tabla, self.pasos_opt, lrm, optm.DECAY, self.rango)), flush=True)
        self.aplicadas += 1
        t5 = time.perf_counter()
        self.tabla.barrera.pasar(2 * e + 2, "B")
        t6 = time.perf_counter()
        for k, v in (("agregar", t1 - t0), ("publicar", t2 - t1), ("espera_A", t3 - t2), ("reunir", t4 - t3), ("optimizar", t5 - t4), ("espera_B", t6 - t5)):
            self.tiempos[k] += v
        with self.cond:
            self.entradas.pop(e, None)

    def _correr(self) -> None:
        try:
            nice.ceder("hilo de tablas")
            aten.fijar_hilo("hilo de tablas")
            for k in range(self.retraso + 1):
                self._reunir_lote(k)
            e = 0
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                self._actualizar(e, *job)
                self._reunir_lote(e + self.retraso + 1)
                e += 1
            self.pasos = e
        except BaseException as ex:
            self.error = ex
            with self.cond:
                self.cond.notify_all()

    def revisar(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"qf2_memhost: fallo del hilo de tabla: {self.error!r}") from self.error

    def anotar(self, clave: str, segundos: float) -> None:
        self.tiempos[clave] += segundos

    def postear(self, grad, lr: float, lrm: float = 1.0) -> None:
        self.revisar()
        self.jobs.put((grad, lr, lrm))

    def E(self, k: int) -> torch.Tensor:
        t0 = time.monotonic()
        with self.cond:
            if k not in self.listos:
                self.esperas += 1
            while k not in self.listos:
                self.revisar()
                if time.monotonic() - t0 > ESPERA_S:
                    raise TimeoutError(f"qf2_memhost: sin vectores del lote {k} en {ESPERA_S:.0f} s")
                self.cond.wait(0.5)
            self.servidos += 1
            return self.listos.pop(k)

    def fin(self) -> int:
        self.jobs.put(None)
        self.cierre.set()
        self.hilo.join(timeout=ESPERA_S)
        self.revisar()
        n = self.pasos
        self.tabla.barrera.pasar(2 * n + 3, "fin")
        return n

    def resumen(self) -> str:
        ms = " ".join(f"{k}={1000 * v / max(1, self.aplicadas if k != 'gather' else self.servidos):.1f}ms" for k, v in sorted(self.tiempos.items()))
        return (f"qf2_memhost: servidos={self.servidos} esperas={self.esperas} actualizaciones={self.aplicadas} retraso={self.retraso} "
                f"modo={self.modo} por_paso: {ms}")
''',
    "qf2_memhost_hojas.py": r'''"""QF2_MEMHOST: one gradient leaf per micro-step.

Slicing a single leaf E[B, T, S*W] per micro-step makes autograd run SliceBackward and AccumulateGrad over the whole E
outside the compiled graph, once per micro-step. Here each micro-step gets its own contiguous leaf, filled from E
without gradient, so its gradient comes straight out of the compiled backward. The step gradient is their
concatenation, padded with zeros up to the rows of E: the same values as the gradient of the sliced leaf.
"""
from __future__ import annotations

import torch


class Hojas:
    def __init__(self):
        self.hojas: list[torch.Tensor] = []

    def repartir(self, e: torch.Tensor, n_micro: int, filas: int) -> list[torch.Tensor]:
        forma = (filas,) + tuple(e.shape[1:])
        if len(self.hojas) != n_micro or tuple(self.hojas[0].shape) != forma or self.hojas[0].dtype != e.dtype:
            self.hojas = [torch.zeros(forma, dtype=e.dtype, device=e.device).requires_grad_(True) for _ in range(n_micro)]
        with torch.no_grad():
            for i, h in enumerate(self.hojas):
                h.copy_(e[i * filas:(i + 1) * filas])
        for h in self.hojas:
            h.grad = None
        return self.hojas

    def gradiente(self, filas_e: int) -> torch.Tensor | None:
        if not self.hojas or any(h.grad is None for h in self.hojas):
            return None
        g = torch.cat([h.grad for h in self.hojas])
        if g.shape[0] < filas_e:
            g = torch.cat([g, g.new_zeros((filas_e - g.shape[0],) + tuple(g.shape[1:]))])
        return g
''',
    "qf2_memhost_intercambio.py": r'''"""QF2_MEMHOST: gradient exchange between the ranks of a host through shared memory.

Each rank aggregates its own row gradients (unique ids sorted ascending, fp32 sums), publishes them in its region and
the offsets of every owner's slice; after barrier A each owner reads its slice from all origins in rank order and sums
per row. The region holds as many rows as references per rank, so it cannot overflow.
"""
from __future__ import annotations

from pathlib import Path

import torch

import qf2_memhost_sincro as sincro


def agregar(ids: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    uid, inv = torch.unique(ids, return_inverse=True)
    out = torch.zeros(uid.numel(), g.size(1), dtype=torch.float32)
    out.index_add_(0, inv, g)
    return uid, out


class Intercambio:
    def __init__(self, ruta: Path, rango: int, mundo: int, capacidad: int, ancho: int, crear: bool):
        self.rango, self.mundo, self.cap, self.w = rango, mundo, capacidad, ancho
        self.cab = sincro.mapear(ruta / "cab.i64", torch.int64, mundo * (mundo + 2), crear).view(mundo, mundo + 2)
        self.ids = sincro.mapear(ruta / "ids.i64", torch.int64, mundo * capacidad, crear).view(mundo, capacidad)
        self.gr = sincro.mapear(ruta / "gr.f32", torch.float32, mundo * capacidad * ancho, crear).view(mundo, capacidad, ancho)

    def publicar(self, uid: torch.Tensor, g: torch.Tensor, limites: torch.Tensor) -> None:
        n = uid.numel()
        if n > self.cap:
            raise RuntimeError(f"qf2_memhost: {n} filas unicas superan la capacidad {self.cap}")
        self.ids[self.rango, :n] = uid
        self.gr[self.rango, :n] = g
        self.cab[self.rango, 0] = n
        self.cab[self.rango, 1:] = torch.searchsorted(uid, limites)

    def reunir(self, propietario: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, gs = [], []
        for s in range(self.mundo):
            a, b = int(self.cab[s, 1 + propietario]), int(self.cab[s, 2 + propietario])
            if b > a:
                ids.append(self.ids[s, a:b].clone())
                gs.append(self.gr[s, a:b].clone())
        if not ids:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, self.w)
        return agregar(torch.cat(ids), torch.cat(gs))
''',
    "qf2_memhost_opt.py": r'''"""QF2_MEMHOST: optimiser for the host tables, applied only to the rows touched in the step.

rms (default): one fp32 second moment per row, bias corrected by the number of touches of that row, and a decay per
touch scaled by the lr multiplier of the schedule: acc = b*acc + (1-b)*mean(g^2); ahat = acc / (1 - b^n);
w = w*(1-decay*lrm) - lr*g/(sqrt(ahat)+eps). The first touch moves every row by an RMS of exactly lr: eps sits far below
the real per-row gradient (~1.9e-10 at n_embd 1536 over 64 ranks).
adam: lazy Adam per element with the exact arithmetic of torch.optim.SparseAdam (the control arm, three times the
master size in RAM).
"""
from __future__ import annotations

import math
import os

import torch

import qf2_memhost_tabla as tab

BETA = float(os.environ.get("QF2_MEMHOST_BETA", "0.99"))
DECAY = float(os.environ.get("QF2_MEMHOST_DECAY", "1e-3"))
EPS = float(os.environ.get("QF2_MEMHOST_EPS", "1e-24"))


@torch.no_grad()
def paso_rms(t: tab.Tabla, ids: torch.Tensor, g: torch.Tensor, lr: float, beta: float = BETA, decay: float | None = None, eps: float = EPS,
             lrm: float = 1.0) -> None:
    if ids.numel() == 0:
        return
    decay = DECAY if decay is None else decay
    w = t.q.index_select(0, ids)
    acc = t.acc.index_select(0, ids)
    n = t.cnt.index_select(0, ids).float() + 1.0
    acc = beta * acc + (1.0 - beta) * g.pow(2).mean(dim=-1)
    ahat = acc / (1.0 - beta ** n)
    w = w * (1.0 - decay * lrm) - lr * g / (ahat.sqrt() + eps).unsqueeze(-1)
    t.q.index_copy_(0, ids, w)
    t.acc.index_copy_(0, ids, acc)
    t.cnt.index_copy_(0, ids, n.clamp_(max=tab.CNT_TOPE).to(torch.int16))


@torch.no_grad()
def paso_adam(t: tab.Tabla, ids: torch.Tensor, g: torch.Tensor, lr: float, paso: int, b1: float = 0.8, b2: float = 0.995, eps: float = 1e-10) -> None:
    if ids.numel() == 0:
        return
    m = t.m.index_select(0, ids)
    v = t.v.index_select(0, ids)
    m = m + (g - m) * (1.0 - b1)
    v = v + (g.pow(2) - v) * (1.0 - b2)
    denom = v.sqrt() + eps
    paso_tam = lr * math.sqrt(1.0 - b2 ** paso) / (1.0 - b1 ** paso)
    t.q.index_copy_(0, ids, t.q.index_select(0, ids) - paso_tam * (m / denom))
    t.m.index_copy_(0, ids, m)
    t.v.index_copy_(0, ids, v)


def aplicar(t: tab.Tabla, modo: str, ids: torch.Tensor, g: torch.Tensor, lr: float, paso: int, lrm: float = 1.0) -> None:
    if modo == "adam":
        paso_adam(t, ids, g, lr, paso)
    else:
        paso_rms(t, ids, g, lr, lrm=lrm)
''',
    "qf2_memhost_rapido.py": r'''"""QF2_MEMHOST: fast table thread, bit for bit the same arithmetic as the reference path of qf2_memhost_hilo.

Work is done in blocks of about 1 MiB with fixed buffers (no large temporaries, so no page faults of fresh memory), signs
come from the unpacked bits of the 64-bit hash word, and the rows of a step are sorted once to aggregate duplicates in the
same order (stable, rank order for the exchange) as torch.unique + index_add_. Only the rms optimiser with signs, an
ancho multiple of 64 and bf16 vectors is covered: anything else must use the reference path (aplica()).
"""
from __future__ import annotations

import numpy as np
import torch

import qf2_memhost_cuantiza as cuant
import qf2_memhost_opt as optm
import qf2_memhost_tabla as tab

BLOQUE_BYTES = 1 << 20
ANILLO_EXTRA = 4


def aplica(cfg, modo: str, dtype: torch.dtype) -> bool:
    return modo == "rms" and cfg.signos and cfg.ancho % 64 == 0 and dtype == torch.bfloat16


class Estado:
    def __init__(self, cfg, tabla, intercambio, rango: int, retraso: int):
        self.cfg, self.tabla, self.inter, self.rango = cfg, tabla, intercambio, rango
        self.mundo = tabla.mundo
        w = cfg.ancho
        self.rb = max(256, BLOQUE_BYTES // (4 * w))
        self.cap = intercambio.cap
        self.bufE = [torch.empty(self.cap, w, dtype=torch.bfloat16) for _ in range(retraso + ANILLO_EXTRA)]
        self.k = 0
        self.wb = torch.empty(self.rb, w)
        self.tb = torch.empty(self.rb, w)
        self.pg = torch.empty(int(self.cap * 1.5), w)


def signo(sigb: np.ndarray) -> torch.Tensor:
    bits = np.unpackbits(sigb, axis=1, bitorder="little")
    return torch.from_numpy(bits).to(torch.float32).mul_(2.0).sub_(1.0)


def reunir_lote(est: Estado, ent: tuple, b: int, t: int) -> torch.Tensor:
    ids, sig = ent
    w = est.cfg.ancho
    n = ids.numel()
    E = est.bufE[est.k % len(est.bufE)]
    est.k += 1
    sigb = sig.numpy().view(np.uint8).reshape(n, 8)
    rb = est.rb
    for a in range(0, n, rb):
        z = min(n, a + rb)
        m = z - a
        wb, tb = est.wb[:m], est.tb[:m]
        torch.index_select(est.tabla.q, 0, ids[a:z], out=wb)
        torch.abs(wb, out=tb)
        s16 = (tb.amax(dim=-1) / 127.0).clamp_min(cuant.SUELO).to(torch.float16)
        s32 = s16.float().unsqueeze(-1)
        torch.div(wb, s32, out=tb)
        torch.round(tb, out=tb)
        tb.clamp_(-127, 127).add_(0.0).mul_(s32)
        tb.view(m, w // 64, 64).mul_(signo(sigb[a:z]).unsqueeze(1))
        E[a:z].copy_(tb)
    return E[:n].view(b, t, est.cfg.ancho_e)


def _segmentos(idn: np.ndarray) -> tuple:
    n = idn.size
    order = np.argsort(idn, kind="stable")
    sid = idn[order]
    es_ini = np.empty(n, dtype=bool)
    es_ini[0] = True
    np.not_equal(sid[1:], sid[:-1], out=es_ini[1:])
    starts = np.flatnonzero(es_ini)
    segid = np.cumsum(es_ini) - 1
    ends = np.append(starts, n)
    return order, sid[starts], es_ini, starts, segid, ends


def _sumar(est: Estado, nu: int, es_ini, starts, segid, ends, out: torch.Tensor, fuente) -> None:
    ub = max(64, int(est.rb * 0.85))
    for u0 in range(0, nu, ub):
        u1 = min(nu, u0 + ub)
        a, z = int(ends[u0]), int(ends[u1])
        gf = fuente(a, z)
        ob = out[u0:u1]
        torch.index_select(gf, 0, torch.from_numpy(starts[u0:u1] - a), out=ob)
        dp = np.flatnonzero(~es_ini[a:z])
        if dp.size:
            ob.index_add_(0, torch.from_numpy(segid[a:z][dp] - u0), gf.index_select(0, torch.from_numpy(dp)))


def agregar(est: Estado, ent: tuple, grad: torch.Tensor) -> tuple:
    ids, sig = ent
    w = est.cfg.ancho
    n = ids.numel()
    order, uid, es_ini, starts, segid, ends = _segmentos(ids.numpy())
    nu = uid.size
    gref = grad.reshape(-1, w)
    sigb = sig.numpy().view(np.uint8).reshape(n, 8)
    orderT = torch.from_numpy(order)

    def fuente(a: int, z: int) -> torch.Tensor:
        gf = gref.index_select(0, orderT[a:z]).float()
        torch.nan_to_num_(gf)
        gf.view(z - a, w // 64, 64).mul_(signo(sigb[order[a:z]]).unsqueeze(1))
        return gf.div_(est.mundo)

    out = est.inter.gr[est.rango]
    _sumar(est, nu, es_ini, starts, segid, ends, out, fuente)
    return torch.from_numpy(uid), out[:nu]


def publicar(est: Estado, uid: torch.Tensor, lim: torch.Tensor) -> None:
    i = est.inter
    n = uid.numel()
    if n > i.cap:
        raise RuntimeError(f"qf2_memhost: {n} filas unicas superan la capacidad {i.cap}")
    i.ids[i.rango, :n] = uid
    i.cab[i.rango, 0] = n
    i.cab[i.rango, 1:] = torch.searchsorted(uid, lim)


def reunir(est: Estado) -> tuple:
    i, cap, w = est.inter, est.inter.cap, est.cfg.ancho
    cab, idsn = i.cab.numpy(), i.ids.numpy()
    idl, fl = [], []
    for s in range(est.mundo):
        a, b = int(cab[s, 1 + est.rango]), int(cab[s, 2 + est.rango])
        if b > a:
            idl.append(idsn[s, a:b])
            fl.append(np.arange(s * cap + a, s * cap + b))
    if not idl:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, w)
    order, uid, es_ini, starts, segid, ends = _segmentos(np.concatenate(idl))
    nu = uid.size
    if nu > est.pg.size(0):
        est.pg = torch.empty(int(nu * 1.3), w)
    grflat = i.gr.view(-1, w)
    flo = torch.from_numpy(np.concatenate(fl)[order])
    _sumar(est, nu, es_ini, starts, segid, ends, est.pg, lambda a, z: grflat.index_select(0, flo[a:z]))
    return torch.from_numpy(uid), est.pg[:nu]


@torch.no_grad()
def optimizar(est: Estado, pid: torch.Tensor, pg: torch.Tensor, lr: float, lrm: float) -> None:
    t = est.tabla
    beta, eps = optm.BETA, optm.EPS
    c = 1.0 - optm.DECAY * lrm
    nu = pid.numel()
    for a in range(0, nu, est.rb):
        z = min(nu, a + est.rb)
        ib, g = pid[a:z], pg[a:z]
        w = t.q.index_select(0, ib)
        acc = t.acc.index_select(0, ib)
        n = t.cnt.index_select(0, ib).float() + 1.0
        acc = beta * acc + (1.0 - beta) * g.pow(2).mean(dim=-1)
        ahat = acc / (1.0 - beta ** n)
        tmp = g * lr
        tmp.div_((ahat.sqrt() + eps).unsqueeze(-1))
        w.mul_(c).sub_(tmp)
        t.q.index_copy_(0, ib, w)
        t.acc.index_copy_(0, ib, acc)
        t.cnt.index_copy_(0, ib, n.clamp_(max=tab.CNT_TOPE).to(torch.int16))
''',
    "qf2_memhost_sincro.py": r'''"""QF2_MEMHOST: epoch barriers between the ranks of one host, kept in a shared file.

Every rank owns one int64 slot, written only by it. wait() polls until all slots reach an epoch and raises after the
deadline, so a dead rank stops the others with a visible error instead of leaving them blocked. Nothing here uses
torch.distributed: the default process group on the instance is neuron and no gloo group runs next to it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

PLAZO_S = float(os.environ.get("QF2_MEMHOST_PLAZO_S", "300"))
SONDEO_S = 1e-4
SONDEO_MAX_S = 5e-3
INICIO_PROCESO = time.time() - 60.0


def mapear(ruta: Path, dtype: torch.dtype, n: int, crear: bool) -> torch.Tensor:
    if crear:
        with open(ruta, "wb") as fh:
            fh.truncate(n * torch.empty((), dtype=dtype).element_size())
    return torch.from_file(str(ruta), shared=True, size=n, dtype=dtype)


def marca_valida(ruta: Path, contenido: str | None) -> bool:
    try:
        return ruta.stat().st_mtime >= INICIO_PROCESO and (contenido is None or ruta.read_text() == contenido)
    except (OSError, ValueError):
        return False


def esperar_marca(ruta: Path, contenido: str | None = None, plazo: float = PLAZO_S) -> None:
    t0 = time.monotonic()
    while not marca_valida(ruta, contenido):
        if time.monotonic() - t0 > plazo:
            raise TimeoutError(f"qf2_memhost: sin {ruta} tras {plazo:.0f} s")
        time.sleep(0.01)


class Barrera:
    def __init__(self, ruta: Path, rango: int, mundo: int, crear: bool, plazo: float = PLAZO_S):
        self.rango, self.mundo, self.plazo = rango, mundo, plazo
        self.ind = mapear(ruta / "indicadores.i64", torch.int64, mundo, crear)
        if crear:
            self.ind.zero_()

    def subir(self, valor: int) -> None:
        self.ind[self.rango] = valor

    def esperar(self, valor: int, que: str = "") -> None:
        t0 = time.monotonic()
        while int(self.ind.min()) < valor:
            if time.monotonic() - t0 > self.plazo:
                raise TimeoutError(f"qf2_memhost: barrera {que} epoca {valor} tras {self.plazo:.0f} s, indicadores={self.ind.tolist()}")
            time.sleep(SONDEO_S if time.monotonic() - t0 < 2e-3 else SONDEO_MAX_S)

    def pasar(self, valor: int, que: str = "") -> None:
        self.subir(valor)
        self.esperar(valor, que)
''',
    "qf2_memhost_sitios.py": r'''"""QF2_MEMHOST: where the recovered vector enters the network.

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
''',
    "qf2_memhost_tabla.py": r'''"""QF2_MEMHOST: the master table, one fp32 file mapped by every rank of the host.

q [R, W] fp32 masters (rows start at zero: the files are sparse and pages are paid on first touch), acc [R] fp32 second
moment per row and cnt [R] int16 touches per row. For the Adam arm m and v [R, W] are added. Rows are owned in contiguous
ranges of ceil(R / world), so each row has one writer per step. leer() returns the rows as evaluation will read them
(quantised to int8 with an fp16 scale); signs are applied by the caller.

Each run lives in its own directory under the base path, named by MASTER_PORT and the launcher pid shared by its ranks.
Rank 0 removes directories whose owner died, refuses a directory whose owner is alive, and writes the run id into the
ready mark that the other ranks wait for, so a stale mark or a neighbour run never gets mapped.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import qf2_memhost_cfg as cfgm
import qf2_memhost_cuantiza as cuant
import qf2_memhost_sincro as sincro

CNT_TOPE = 30000


def id_corrida() -> str:
    return f"{os.environ.get('MASTER_PORT', '0')}-{os.getppid()}"


def borrar(d: Path) -> None:
    if not d.is_dir():
        return
    for f in d.iterdir():
        if f.is_dir():
            borrar(f)
        else:
            f.unlink(missing_ok=True)
    try:
        d.rmdir()
    except OSError:
        pass


def vivo(d: Path) -> bool:
    try:
        os.kill(int((d / "duenio").read_text()), 0)
    except (OSError, ValueError):
        return False
    return True


def preparar(base: Path, ruta: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for d in base.iterdir():
        if d.is_dir() and d != ruta and not vivo(d):
            borrar(d)
    if ruta.exists() and vivo(ruta):
        raise RuntimeError(f"qf2_memhost: {ruta} pertenece a una corrida viva (pid {(ruta / 'duenio').read_text()})")
    borrar(ruta)
    ruta.mkdir()
    (ruta / "duenio").write_text(str(os.getpid()))


class Tabla:
    def __init__(self, cfg: cfgm.Cfg, ruta: str, rango: int, mundo: int, adam: bool = False):
        self.cfg, self.rango, self.mundo, self.adam = cfg, rango, mundo, adam
        self.id = id_corrida()
        self.base = Path(ruta)
        self.ruta = self.base / self.id
        self.r, self.w = cfg.filas, cfg.ancho
        self.ro = -(-self.r // mundo)
        crear = rango == 0
        marca = self.ruta / "listo"
        if crear:
            preparar(Path(ruta), self.ruta)
        else:
            sincro.esperar_marca(marca, self.id)
        self.q = sincro.mapear(self.ruta / "q.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
        self.acc = sincro.mapear(self.ruta / "acc.f32", torch.float32, self.r, crear)
        self.cnt = sincro.mapear(self.ruta / "cnt.i16", torch.int16, self.r, crear)
        self.m = self.v = None
        if adam:
            self.m = sincro.mapear(self.ruta / "m.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
            self.v = sincro.mapear(self.ruta / "v.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
        self.barrera = sincro.Barrera(self.ruta, rango, mundo, crear)
        if crear:
            marca.write_text(self.id)

    @property
    def propias(self) -> tuple:
        return self.rango * self.ro, min(self.r, (self.rango + 1) * self.ro)

    def limites(self) -> torch.Tensor:
        return torch.tensor([min(self.r, o * self.ro) for o in range(self.mundo + 1)], dtype=torch.int64)

    def leer(self, ids: torch.Tensor) -> torch.Tensor:
        return cuant.ida_vuelta(self.q.index_select(0, ids))

    def leer_exacto(self, ids: torch.Tensor) -> torch.Tensor:
        return self.q.index_select(0, ids)

    def cerrar(self) -> None:
        borrar(self.ruta)
        try:
            self.base.rmdir()
        except OSError:
            pass
''',
    "qf2_memhost_traza.py": r'''"""QF2_MEMHOST_TRAZA: read-only trace of the host-table optimiser, to tell a badly scaled update from a lack of signal.

Every N optimiser steps (N = TRAZA, 0 = off: nothing is read, nothing is printed) the rank that owns the central row of
a hash head takes up to MAX_FILAS touched rows of that head before the update and the same rows after it, and prints one
line per head (tagged rango=R, so every head is observed on the rank that holds its middle rows at any world size): percentiles 10/50/90 of
|dw|/|w| over the rows with a non-zero w, the fraction of sampled rows still at zero, the median share of the decay term
in the update, and the mean gradient norm per bucket of touches (1, 2-3, 4-15, 16-63, 64+). Only copies are made: no value
of the table or of the optimiser state is written. train.py sets TRAZA from QF2_MEMHOST_TRAZA.
"""
from __future__ import annotations

import time

import torch

TRAZA = 0
MAX_FILAS = 4096
CUBOS = ("1", "2_3", "4_15", "16_63", "64mas")
BORDES = (2, 4, 16, 64)


def activa(paso: int, rango: int, cfg, mundo: int) -> tuple:
    if TRAZA <= 0 or paso % TRAZA:
        return ()
    ro = -(-cfg.filas // mundo)
    return tuple(k for k, (b, (_o, f)) in enumerate(zip(cfg.bases, cfg.cabezas)) if (b + f // 2) // ro == rango)


def _elegir(n: int) -> torch.Tensor:
    if n <= MAX_FILAS:
        return torch.arange(n)
    return torch.linspace(0, n - 1, MAX_FILAS).long()


@torch.no_grad()
def previa(t, ids: torch.Tensor, g: torch.Tensor, cfg, cabs: tuple) -> list:
    t0 = time.perf_counter()
    cab = torch.bucketize(ids, torch.tensor(cfg.bases[1:], dtype=ids.dtype), right=True)
    out = []
    for k in cabs:
        pos = (cab == k).nonzero().reshape(-1)
        if pos.numel() == 0:
            continue
        pos = pos[_elegir(pos.numel())]
        sel = ids.index_select(0, pos)
        out.append((k, sel, t.q.index_select(0, sel), t.cnt.index_select(0, sel).float() + 1.0, g.index_select(0, pos).float().norm(dim=-1)))
    return [out, time.perf_counter() - t0]


def _cuantiles(x: torch.Tensor) -> tuple:
    if x.numel() == 0:
        return 0.0, 0.0, 0.0
    return tuple(float(v) for v in torch.quantile(x, torch.tensor([0.1, 0.5, 0.9])))


@torch.no_grad()
def lineas(est: list, t, paso: int, lrm: float, decay: float, rango: int) -> list:
    t0 = time.perf_counter()
    out = []
    for k, sel, w0, n, gn in est[0]:
        dw = (t.q.index_select(0, sel) - w0).norm(dim=-1)
        nw = w0.norm(dim=-1)
        nz = nw > 0
        p10, p50, p90 = _cuantiles(dw[nz] / nw[nz])
        mov = dw > 0
        _a, dec, _b = _cuantiles(decay * lrm * nw[mov] / dw[mov])
        cubo = torch.bucketize(n, torch.tensor(BORDES, dtype=n.dtype), right=True)
        gcub = " ".join(f"gn_{c}={float(gn[cubo == i].mean()) if (cubo == i).any() else 0.0:.3e} c_{c}={int((cubo == i).sum())}" for i, c in enumerate(CUBOS))
        ms = 1000.0 * (est[1] + time.perf_counter() - t0)
        out.append(f"qf2_memhost_traza paso={paso} rango={rango} cabeza={k} filas={sel.numel()} cero={float((~nz).float().mean()):.4f} "
                   f"dw_p10={p10:.3e} dw_p50={p50:.3e} dw_p90={p90:.3e} decay_parte={dec:.3e} {gcub} ms={ms:.2f}")
    return out
''',
    "qf2_memoria.py": r'''"""QF2_MEM_FILAS: row-sharded hashed bigram/trigram memory (two tables of FILAS x ANCHO), added to the input embedding.

Same hash and sentinel rules as qf2_ngramas.indices. In training every rank owns a contiguous block of FILAS/world rows of each
table (plain nn.Parameter outside the model and outside DistMuonAdamW). A lookup is: all_gather of the int32 row ids of all
ranks, a masked local gather of the owned rows, and a reduce_scatter (SUM) of the outputs, which gives each rank back its own
queries exactly (one non-zero term plus zeros). The backward is an all_gather of the output gradient and a local scatter-add
into the shard, scaled by 1/world to match the AVG gradient average of the replicated parameters. Evaluation uses the same
tables replicated in the model (combinar is shared by both paths). QF2_MEM_TRANSPORTE=a2a replaces the lookup transport by
an all_to_all_single to the owner and back (qf2_memoria_a2a); "b" is the one described above.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import qf2_memoria_a2a
import qf2_ngramas

BLOQUE_INICIO = 64
STD_INICIO = 0.8
CLAVES = ("2", "3")
TROZO = 65536
TOPE_BYTES = 8 * 10 ** 9
TRANSPORTES = ("b", "a2a")


def comprobar(filas: int, ancho: int, world: int = 1) -> None:
    qf2_ngramas.comprobar(filas)
    if ancho < 1:
        raise ValueError(f"QF2_MEM_ANCHO={ancho} no es valido")
    if filas % (world * BLOQUE_INICIO) != 0:
        raise ValueError(f"QF2_MEM_FILAS={filas} debe ser multiplo de world*{BLOQUE_INICIO}={world * BLOQUE_INICIO}")


def iniciar_filas(semilla: int, tabla: int, fila0: int, n: int, ancho: int) -> torch.Tensor:
    out = torch.empty(n, ancho)
    for b in range(n // BLOQUE_INICIO):
        g = torch.Generator().manual_seed(((semilla * 2 + tabla) << 32) + fila0 // BLOQUE_INICIO + b)
        out[b * BLOQUE_INICIO:(b + 1) * BLOQUE_INICIO] = torch.randn(BLOQUE_INICIO, ancho, generator=g) * STD_INICIO
    return out


def tablas(filas: int, ancho: int, dtype: torch.dtype = torch.float32, device=None) -> nn.ModuleDict:
    comprobar(filas, ancho)
    return nn.ModuleDict({k: nn.Embedding(filas, ancho, _weight=torch.empty(filas, ancho, dtype=dtype, device=device)) for k in CLAVES})


@torch.no_grad()
def iniciar(mods: nn.ModuleDict, semilla: int) -> None:
    for i, k in enumerate(CLAVES):
        w = mods[k].weight
        w.copy_(iniciar_filas(semilla, i, 0, w.size(0), w.size(1)))


def mundo() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def trozos(n: int, world: int) -> list[tuple[int, int]]:
    c = max(1, TROZO // world)
    return [(s, min(n, s + c)) for s in range(0, n, c)]


class Consulta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, filas_g, rank, world, dtype):
        r = shard.size(0)
        n = filas_g.numel() // world
        ids = filas_g.view(world, n)
        copia = shard.detach().to(dtype)
        out = copia.new_empty(n, copia.size(1))
        for s, e in trozos(n, world):
            f = ids[:, s:e].reshape(-1)
            propia = (f // r) == rank
            loc = (f - rank * r).clamp(0, r - 1)
            local = copia[loc] * propia.unsqueeze(1).to(dtype)
            if world > 1:
                dist.reduce_scatter_tensor(out[s:e], local, op=dist.ReduceOp.SUM)
            else:
                out[s:e] = local
        ctx.save_for_backward(filas_g)
        ctx.meta = (r, rank, world)
        return out

    @staticmethod
    def backward(ctx, g):
        (filas_g,) = ctx.saved_tensors
        r, rank, world = ctx.meta
        g = g.contiguous()
        n = g.size(0)
        ids = filas_g.view(world, n)
        dw = torch.zeros(r + 1, g.size(1), dtype=torch.float32, device=g.device)
        for s, e in trozos(n, world):
            f = ids[:, s:e].reshape(-1)
            if world > 1:
                todo = g.new_empty(world * (e - s), g.size(1))
                dist.all_gather_into_tensor(todo, g[s:e].contiguous())
            else:
                todo = g[s:e]
            propia = (f // r) == rank
            destino = torch.where(propia, f - rank * r, torch.full_like(f, r))
            dw.index_add_(0, destino, todo.float())
        return dw[:r] / world, None, None, None, None


class Memoria:
    def __init__(self, filas: int, ancho: int, device, semilla: int, bos: int, transporte: str = "b"):
        if transporte not in TRANSPORTES:
            raise ValueError(f"QF2_MEM_TRANSPORTE={transporte} no es valido, debe ser uno de {TRANSPORTES}")
        self.transporte = transporte
        self.desbordes = torch.zeros((), dtype=torch.int64, device=device)
        self.rank, self.world = mundo()
        comprobar(filas, ancho, self.world)
        por_rango = filas * ancho * len(CLAVES) * 16 // self.world
        if por_rango > TOPE_BYTES:
            raise ValueError(f"memoria repartida: {filas} filas x {ancho} x {len(CLAVES)} tablas x 16 B / {self.world} rangos = {por_rango} B por rango, supera {TOPE_BYTES}; hace falta mas rangos")
        self.filas, self.ancho, self.bos = filas, ancho, bos
        self.r = filas // self.world
        self.shards = [nn.Parameter(iniciar_filas(semilla, i, self.rank * self.r, self.r, ancho).to(device)) for i in range(len(CLAVES))]

    def parametros(self) -> list[nn.Parameter]:
        return list(self.shards)

    def zero_grad(self) -> None:
        for p in self.shards:
            p.grad = None

    def reunir(self, h: torch.Tensor) -> torch.Tensor:
        h = h.reshape(-1).to(torch.int32).contiguous()
        if self.world > 1:
            todo = h.new_empty(self.world * h.numel())
            dist.all_gather_into_tensor(todo, h)
            h = todo
        return h.to(torch.int64)

    def consultar(self, idx: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.shape
        hs = qf2_ngramas.indices(idx, self.filas, self.bos)
        if self.transporte == "a2a":
            return tuple(qf2_memoria_a2a.consultar(s, h, self.rank, self.world, self.desbordes, dtype).view(b, t, self.ancho) for s, h in zip(self.shards, hs))
        return tuple(Consulta.apply(s, self.reunir(h), self.rank, self.world, dtype).view(b, t, self.ancho) for s, h in zip(self.shards, hs))


def consulta_replicada(mods: nn.ModuleDict, idx: torch.Tensor, filas: int, bos: int) -> tuple[torch.Tensor, torch.Tensor]:
    h2, h3 = qf2_ngramas.indices(idx, filas, bos)
    return mods["2"](h2), mods["3"](h3)


def combinar(m2: torch.Tensor, m3: torch.Tensor, proy: nn.Module, esc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = torch.cat([F.rms_norm(m2.to(x.dtype), (m2.size(-1),)), F.rms_norm(m3.to(x.dtype), (m3.size(-1),))], dim=-1)
    return x + esc.to(x.dtype) * proy(m)
''',
    "qf2_memoria_a2a.py": r'''"""QF2_MEM_TRANSPORTE=a2a: each query row travels only to the rank that owns it and back (all_to_all_single), instead of every
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
''',
    "qf2_memoria_anfitrion.py": r'''from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

import qf2_memoria
import qf2_ngramas

TROZO_LECTURA = 1 << 26


def claves() -> list[str]:
    return [f"memoria.{k}.weight" for k in qf2_memoria.CLAVES]


def abrir(checkpoint_path: str) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)


def bytes_tablas(state: dict) -> int:
    return sum(state[k].numel() * state[k].element_size() for k in claves() if k in state)


def usa_anfitrion(checkpoint_path: str, umbral: int) -> bool:
    return bytes_tablas(abrir(checkpoint_path)["model_state"]) > umbral


def precargar(checkpoint_path: str) -> None:
    rank, world = qf2_memoria.mundo()
    total = os.path.getsize(checkpoint_path)
    ini, fin = total * rank // world, total * (rank + 1) // world
    fd = os.open(checkpoint_path, os.O_RDONLY)
    try:
        for pos in range(ini, fin, TROZO_LECTURA):
            os.pread(fd, min(TROZO_LECTURA, fin - pos), pos)
    finally:
        os.close(fd)
    if world > 1:
        dist.barrier()


def cargar_resto(model: nn.Module, state: dict) -> None:
    propias = set(claves())
    res = model.load_state_dict({k: v for k, v in state.items() if k not in propias}, strict=False)
    if res.unexpected_keys or set(res.missing_keys) != propias:
        raise RuntimeError(f"claves inesperadas al cargar sin tablas: faltan={res.missing_keys} sobran={res.unexpected_keys}")


def asignar(model: nn.Module, state: dict) -> None:
    for k, clave in zip(qf2_memoria.CLAVES, claves()):
        tabla = state[clave]
        if tuple(tabla.shape) != (model.config.memoria, model.config.memoria_ancho):
            raise RuntimeError(f"{clave} tiene forma {tuple(tabla.shape)}, el modelo espera ({model.config.memoria}, {model.config.memoria_ancho})")
        model.memoria[k].weight = nn.Parameter(tabla, requires_grad=False)


@torch.compiler.disable
def consultar(mods: nn.ModuleDict, idx: torch.Tensor, filas: int, bos: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, t = idx.shape
    hs = qf2_ngramas.indices(idx.detach().cpu(), filas, bos)
    salida = []
    for k, h in zip(qf2_memoria.CLAVES, hs):
        w = mods[k].weight
        salida.append(torch.index_select(w, 0, h.reshape(-1)).view(b, t, w.size(1)).to(idx.device))
    return tuple(salida)
''',
    "qf2_memoria_ckpt.py": r'''"""Consolidation of the row-sharded memory tables into full tables for the checkpoint (QF2_MEM_FILAS). Collective: every rank
must call it; only rank 0 returns tensors (keys of the replicated evaluation model, memoria.<k>.weight)."""
from __future__ import annotations

import torch
import torch.distributed as dist

import qf2_memoria

TROZO = 1024


def consolidar(mem: qf2_memoria.Memoria, dtype: torch.dtype, trozo: int = TROZO) -> dict:
    salida = {}
    r, w = mem.r, mem.world
    for k, shard in zip(qf2_memoria.CLAVES, mem.shards):
        total = torch.empty(mem.filas, mem.ancho, dtype=dtype) if mem.rank == 0 else None
        for c in range(0, r, trozo):
            parte = shard.detach()[c:c + trozo].to(dtype).contiguous()
            n = parte.size(0)
            if w > 1:
                todo = parte.new_empty(w * n, parte.size(1))
                dist.all_gather_into_tensor(todo, parte)
            else:
                todo = parte
            if mem.rank == 0:
                todo = todo.cpu()
                for q in range(w):
                    total[q * r + c:q * r + c + n] = todo[q * n:(q + 1) * n]
        if mem.rank == 0:
            salida[f"memoria.{k}.weight"] = total
    return salida


def cargar_trozo(mem: qf2_memoria.Memoria, tablas: dict) -> None:
    with torch.no_grad():
        for k, shard in zip(qf2_memoria.CLAVES, mem.shards):
            shard.copy_(tablas[f"memoria.{k}.weight"][mem.rank * mem.r:(mem.rank + 1) * mem.r].to(shard.dtype))
''',
    "qf2_memoria_opt.py": r'''"""Adam over the row shards of the memory tables (QF2_MEM_FILAS). The shards are not in DistMuonAdamW: their gradient is
already the global average (see qf2_memoria.Consulta.backward). The update is MuonAdamW._adamw_update, passed in by train."""
from __future__ import annotations

import torch

BETAS = (0.8, 0.995)
EPS = 1e-10


class AdamShard:
    def __init__(self, params, lr: float, actualizar):
        self.actualizar = actualizar
        self.state: dict = {}
        self.param_groups = [dict(kind="adamw", params=list(params), lr=lr, initial_lr=lr, betas=BETAS, eps=EPS, weight_decay=0.0)]
        for g in self.param_groups:
            g["_lr_t"] = torch.tensor(float(g["lr"]), dtype=torch.float32)
            g["_wd_t"] = torch.tensor(0.0, dtype=torch.float32)
            g["_one_minus_beta1_t"] = torch.tensor(1 - BETAS[0], dtype=torch.float32)
            g["_one_minus_beta2_t"] = torch.tensor(1 - BETAS[1], dtype=torch.float32)
            g["_eps_t"] = torch.tensor(EPS, dtype=torch.float32)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    self.actualizar(p, p.grad, self.state.setdefault(p, {}), g)
''',
    "qf2_mos.py": r'''"""QF2_MOS: mixture of two softmax distributions over the shared output head (astra r37, proposal C).

From the final hidden state h (after the final norm) two representations are built: h1 = h and h2 = norm(h + W2 h). Both go through
the same lm_head and the same logit softcap of the kit, and a gate pi(h) = sigmoid(w_pi . h + b_pi) mixes the two distributions:
log p = logaddexp(log pi + log_softmax(logits_1), log(1 - pi) + log_softmax(logits_2)). Everything is per position, so causality and
batch independence are those of the rest of the model.

Training loss: -mean(log p[target]) is computed from the per-branch logsumexp and the target logit only,
log p[target] = logaddexp_i(log pi_i + z_i[target] - lse_i), so no mixed [N, V] tensor is built. With cefa=True the lse comes from
cefa_perdida.cef_lse (bf16 residual per branch); otherwise it is the explicit softcap + logsumexp route, which keeps two fp32
[N, V] tensors per branch for the backward pass (tanh output and softcapped logits).

Evaluation (targets=None): the model returns log p over the full vocabulary as its logits. exp(log p) sums to 1 per position, so the
log_softmax / cross-entropy applied by prepare.py and core_eval returns exactly -log p[target] without altering the scoring procedure.

W2 and w_pi/b_pi are plain Parameters created empty (no RNG consumed) and initialised last, so with QF2_MOS=0 the model and its
initialisation stream are untouched. W2 starts at W2_ESCALA times the usual uniform init so the two branches already differ at step 0
(an exact copy h2 = h would give a mixture identical to the single softmax and a symmetric start); w_pi starts at 0 and b_pi at 0, so
pi = 0.5.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RAMAS = 2
W2_ESCALA = 0.5
PI_LR_BASE = 0.2


def comprobar(ramas: int) -> None:
    if ramas != RAMAS:
        raise ValueError(f"QF2_MOS={ramas} debe ser 0 (apagado) o {RAMAS} (dos ramas)")


def pi_lr(n_embd: int) -> float:
    return PI_LR_BASE / n_embd


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def logits_rama(h: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float) -> torch.Tensor:
    z = lm_head(h)[..., :vocab].float()
    return cap * torch.tanh(z / cap)


def lse_y_objetivo(h: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, ids: torch.Tensor, cefa: bool):
    if cefa:
        from cefa_perdida import cef_lse
        w = lm_head.weight
        lse = cef_lse(F.linear(h, w[:vocab].to(dtype=h.dtype), None))
        zt = (h * F.embedding(ids, w).to(dtype=h.dtype)).float().sum(-1)
        return lse, cap * torch.tanh(zt / cap)
    z = logits_rama(h, lm_head, vocab, cap)
    return torch.logsumexp(z, dim=-1), z.gather(1, ids.unsqueeze(1)).squeeze(1)


class Mezcla(nn.Module):
    def __init__(self, n_embd: int, ramas: int = RAMAS):
        super().__init__()
        comprobar(ramas)
        self.w2 = nn.Parameter(torch.empty(n_embd, n_embd))
        self.pi_w = nn.Parameter(torch.empty(n_embd, 1))
        self.pi_b = nn.Parameter(torch.empty(1))

    @torch.no_grad()
    def iniciar(self) -> None:
        n_embd = self.w2.size(1)
        s = W2_ESCALA * math.sqrt(3.0) * n_embd ** -0.5
        nn.init.uniform_(self.w2, -s, s)
        nn.init.zeros_(self.pi_w)
        nn.init.zeros_(self.pi_b)

    def ramas(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return h, norm(h + F.linear(h, self.w2.to(h.dtype)))

    def log_puertas(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = h.float() @ self.pi_w.float() + self.pi_b.float()
        return F.logsigmoid(g), F.logsigmoid(-g)

    def log_prob(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float) -> torch.Tensor:
        (h1, h2), (l1, l2) = self.ramas(x), self.log_puertas(x)
        a = l1 + F.log_softmax(logits_rama(h1, lm_head, vocab, cap), dim=-1)
        b = l2 + F.log_softmax(logits_rama(h2, lm_head, vocab, cap), dim=-1)
        return torch.logaddexp(a, b)

    def nll_tokens(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, targets: torch.Tensor, cefa: bool):
        n_embd = x.size(-1)
        xf, ids = x.reshape(-1, n_embd), targets.reshape(-1).clamp_min(0)
        (h1, h2), (l1, l2) = self.ramas(xf), self.log_puertas(xf)
        lse1, zt1 = lse_y_objetivo(h1, lm_head, vocab, cap, ids, cefa)
        lse2, zt2 = lse_y_objetivo(h2, lm_head, vocab, cap, ids, cefa)
        return -torch.logaddexp(l1.squeeze(1) + zt1 - lse1, l2.squeeze(1) + zt2 - lse2)

    def forward(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, targets: torch.Tensor | None = None,
                loss_reduction: str = "mean", cefa: bool = False) -> torch.Tensor:
        if targets is None:
            return self.log_prob(x, lm_head, vocab, cap)
        nll = self.nll_tokens(x, lm_head, vocab, cap, targets, cefa)
        valido = targets.reshape(-1) >= 0
        nll = torch.where(valido, nll, torch.zeros_like(nll))
        if loss_reduction == "none":
            return nll.view_as(targets)
        return nll.sum() / valido.sum()
''',
    "qf2_ngramas.py": r'''"""QF2_NGRAM_FILAS: causal hashed bigram and trigram memory tables added to the input embedding.

Two tables of FILAS rows (the same for every rank, like any embedding) are indexed by a deterministic hash of the last two
(bigram) or three (trigram) tokens ending at the current one. The hash is computed in int64 inside the forward from the
input ids only: it never reads the next token, the target, or other rows of the batch. Positions without enough history
since the start of the row or the most recent BOS token map to a fixed sentinel row; the (BOS, t) bigram is kept. Both tables are normalised, summed and scaled by one learned scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MULT_BI = (10007,)
MULT_TRI = (10009, 12979)
ESCALA_INICIAL = 0.1
MINIMO = 64


def primo_bajo(filas: int) -> int:
    n = filas - 3
    while n > 2 and any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n -= 1
    return n


def comprobar(filas: int) -> None:
    if filas < MINIMO:
        raise ValueError(f"QF2_NGRAM_FILAS={filas} es menor que {MINIMO}")


def desplazar(idx: torch.Tensor, k: int) -> torch.Tensor:
    return F.pad(idx, (k, 0))[:, :idx.size(1)]


def indices(idx: torch.Tensor, filas: int, bos: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = primo_bajo(filas)
    i0 = idx.to(torch.int64)
    i1, i2 = desplazar(i0, 1), desplazar(i0, 2)
    pos = torch.arange(i0.size(1), device=i0.device).unsqueeze(0)
    h2 = (i1 * MULT_BI[0] + i0) % p
    h3 = (((i2 * MULT_TRI[0] + i1) % p) * MULT_TRI[1] + i0) % p
    vale2 = (i0 != bos) & (pos >= 1)
    vale3 = vale2 & (i1 != bos) & (pos >= 2)
    centinela = torch.full_like(h2, p)
    return torch.where(vale2, h2, centinela), torch.where(vale3, h3, centinela)


def tablas(filas: int, n_embd: int) -> nn.ModuleDict:
    comprobar(filas)
    with torch.random.fork_rng(devices=[]):
        return nn.ModuleDict({"2": nn.Embedding(filas, n_embd), "3": nn.Embedding(filas, n_embd)})


def escala() -> nn.Parameter:
    return nn.Parameter(torch.full((1,), ESCALA_INICIAL))


@torch.no_grad()
def iniciar(mods: nn.ModuleDict, n_embd: int) -> None:
    for mod in mods.values():
        torch.nn.init.normal_(mod.weight, mean=0.0, std=0.8)


def sumar(mods: nn.ModuleDict, esc: torch.Tensor, idx: torch.Tensor, filas: int, bos: int, tabla, x: torch.Tensor) -> torch.Tensor:
    h2, h3 = indices(idx, filas, bos)
    m = F.rms_norm(tabla(mods["2"], h2).to(x.dtype), (x.size(-1),)) + F.rms_norm(tabla(mods["3"], h3).to(x.dtype), (x.size(-1),))
    return x + esc.to(x.dtype) * m
''',
    "qf2_objetivos.py": r'''"""QF2_CE_DIFICILES, QF2_CE_TRAMO y QF2_RUIDO_ENTRADA: objetivo y regularización del banco CORE bifurcado, ronda 2.

Los dos objetivos reciben la NLL por token de la CE por trozos (ce_trozos.ce_trozos_tokens, 0 donde el objetivo es -1)
con forma [B, T] y la máscara de objetivos válidos, y devuelven la pérdida escalar del micro-lote.

Dificiles(factor): ventanas fijas de VENTANA posiciones de cada fila. En cada ventana pesan factor los floor(v / 4)
tokens válidos de mayor NLL (v = válidos de la ventana: 16 con la ventana llena) y 1 el resto de válidos; los -1 pesan
0 y no cuentan en v. El orden se decide por comparaciones por pares dentro de la ventana (empates: gana la posición
anterior), sin sort ni topk. Los pesos se normalizan para que su media sobre los válidos del micro-lote sea 1 y la
pérdida es la media ponderada sum(w * nll) / válidos. Los pesos salen de comparaciones y van sin gradiente: d pérdida /
d nll_i = w_i / válidos. La normalización depende solo del número de válidos por ventana, no de las NLL; lo que no es
causal es el orden dentro de la ventana: el peso de un token puede cambiar con la NLL de tokens posteriores de su misma
ventana. Es una ponderación del entreno: la NLL que predice el modelo sigue siendo causal.

Tramo(f, tau, k): pérdida = (1 - f) * CE media + f * media, sobre las ventanas fijas de k posiciones con sus k objetivos
válidos, de tau * logsumexp(nll / tau) - tau * log k. Sin ninguna ventana entera el segundo término vale 0.

Ruido(r): en entreno, U(-a, a) con a = r * sqrt(3) (RMS r) y forma [B, 1, d], nuevo en cada micro-paso, de un
generador de CPU propio sembrado con la semilla y el rango (no toca el RNG global). train.forward_mem lo pide solo con
model.training y GPT.forward lo suma tras la norma del embedding de entrada solo con self.training; la evaluación no lo
ve. Vacío o '0' apaga cada mando.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

VENTANA = 64
SAL_RUIDO = 7_340_033


def _numero(texto: str, nombre: str) -> float:
    try:
        valor = float(texto)
    except ValueError:
        raise ValueError(f"{nombre}={texto!r} no es un número") from None
    if not math.isfinite(valor):
        raise ValueError(f"{nombre}={texto!r} no es finito")
    return valor


def apagado(texto: str) -> bool:
    return texto.strip() in ("", "0")


def pesos_dificiles(nll: torch.Tensor, vale: torch.Tensor, factor: float) -> torch.Tensor:
    b, t = nll.shape
    if t % VENTANA:
        raise ValueError(f"QF2_CE_DIFICILES: T={t} no es múltiplo de la ventana {VENTANA}")
    n = nll.detach().float().reshape(b, t // VENTANA, VENTANA)
    v = vale.reshape(b, t // VENTANA, VENTANA).float()
    pos = torch.arange(VENTANA, device=nll.device)
    antes = (pos.unsqueeze(0) < pos.unsqueeze(1)).float()
    mayor = (n.unsqueeze(-2) > n.unsqueeze(-1)).float()
    empate = (n.unsqueeze(-2) == n.unsqueeze(-1)).float() * antes
    delante = ((mayor + empate) * v.unsqueeze(-2)).sum(dim=-1)
    cupo = torch.floor(v.sum(dim=-1, keepdim=True) / 4.0)
    alto = (delante < cupo).float() * v
    w = v + (factor - 1.0) * alto
    w = w * (v.sum() / w.sum().clamp_min(1.0))
    return w.reshape(b, t).detach()


def perdida_dificiles(nll: torch.Tensor, vale: torch.Tensor, factor: float) -> torch.Tensor:
    w = pesos_dificiles(nll, vale, factor)
    return (w * nll).sum() / vale.sum()


def perdida_tramo(nll: torch.Tensor, vale: torch.Tensor, f: float, tau: float, k: int) -> torch.Tensor:
    b, t = nll.shape
    if t % k:
        raise ValueError(f"QF2_CE_TRAMO: T={t} no es múltiplo de k={k}")
    ce = torch.where(vale, nll, 0.0).sum() / vale.sum()
    entera = vale.reshape(b, t // k, k).float().sum(dim=-1) == k
    peor = tau * torch.logsumexp(nll.float().reshape(b, t // k, k) / tau, dim=-1) - tau * math.log(k)
    tramo = torch.where(entera, peor, 0.0).sum() / entera.float().sum().clamp_min(1.0)
    return (1.0 - f) * ce + f * tramo


@dataclass(frozen=True)
class Dificiles:
    factor: float

    def __call__(self, nll: torch.Tensor, vale: torch.Tensor) -> torch.Tensor:
        return perdida_dificiles(nll, vale, self.factor)

    def descripcion(self) -> str:
        return f"dificiles factor={self.factor} ventana={VENTANA} cuartil=floor(validos/4)"


@dataclass(frozen=True)
class Tramo:
    f: float
    tau: float
    k: int

    def __call__(self, nll: torch.Tensor, vale: torch.Tensor) -> torch.Tensor:
        return perdida_tramo(nll, vale, self.f, self.tau, self.k)

    def descripcion(self) -> str:
        return f"tramo f={self.f} tau={self.tau} k={self.k}"


def leer_objetivo(dificiles: str, tramo: str) -> Dificiles | Tramo | None:
    if not apagado(dificiles) and not apagado(tramo):
        raise ValueError("QF2_CE_DIFICILES y QF2_CE_TRAMO no se combinan: un brazo por mando")
    if not apagado(dificiles):
        factor = _numero(dificiles, "QF2_CE_DIFICILES")
        if factor <= 0:
            raise ValueError(f"QF2_CE_DIFICILES={dificiles!r} debe ser positivo")
        return Dificiles(factor)
    if not apagado(tramo):
        partes = tramo.strip().split(":")
        if len(partes) != 3:
            raise ValueError(f"QF2_CE_TRAMO={tramo!r} no es 'f:tau:k'")
        f, tau = _numero(partes[0], "QF2_CE_TRAMO f"), _numero(partes[1], "QF2_CE_TRAMO tau")
        if not partes[2].isdigit() or int(partes[2]) < 1 or not 0.0 <= f <= 1.0 or tau <= 0:
            raise ValueError(f"QF2_CE_TRAMO={tramo!r}: f en [0, 1], tau > 0 y k entero >= 1")
        return Tramo(f, tau, int(partes[2]))
    return None


class Ruido:
    def __init__(self, texto: str, semilla: int):
        self.r = 0.0 if apagado(texto) else _numero(texto, "QF2_RUIDO_ENTRADA")
        if self.r < 0:
            raise ValueError(f"QF2_RUIDO_ENTRADA={texto!r} debe ser >= 0")
        self.activo = self.r > 0
        self.semilla = semilla
        self.generador: torch.Generator | None = None

    def muestra(self, filas: int, ancho: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.generador is None:
            rango = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.generador = torch.Generator().manual_seed(SAL_RUIDO * (self.semilla + 1) + rango)
        a = self.r * math.sqrt(3.0)
        u = torch.rand((filas, 1, ancho), generator=self.generador, dtype=torch.float32)
        return ((2.0 * u - 1.0) * a).to(device=device, dtype=dtype)

    def descripcion(self) -> str:
        return f"ruido_entrada rms={self.r} a={self.r * math.sqrt(3.0):.6f} forma=[B,1,d] solo entreno" if self.activo else "ruido_entrada apagado"


def descripcion(objetivo: Dificiles | Tramo | None, ruido: Ruido) -> str:
    return f"qf2_objetivos: {objetivo.descripcion() if objetivo is not None else 'ce ordinaria'}; {ruido.descripcion()}"
''',
    "qf2_opt_compilado.py": r'''"""Compiled optimizer update (QF2_OPT_COMPILADO=1; default OFF).

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
''',
    "qf2_peso_contexto.py": r'''"""QF2_PESO_CTX: pesos por token segun el beneficio del contexto, solo en entreno (brazos B y C de f2-peso-beneficio-contexto).

Beneficio de un objetivo t = nll con contexto corto recalculado - nll con contexto completo, medido por el modelo que se
entrena (sin gradiente, sin estado). Contexto corto: dos particiones de la fila en trozos de K tokens, una desde 0 y otra
desde K/2; cada objetivo toma el trozo que le da mas contexto previo j en [K/2, K-1] (objetivos con posicion < K: beneficio 0,
su contexto completo es corto). Perdida: sum(w * nll_completo) / n_validos con w = u + (1-u) * w_norm, w_norm = w_bruto /
media(w_bruto), w_bruto = min(1 + alfa * max(beneficio, 0) / media(max(beneficio, 0)), tope). Los pesos tienen media 1 sobre
los validos de las filas con senal, son > 0 y <= u + (1-u) * tope; las demas filas pesan 1. El brazo C reparte esos mismos
pesos con una permutacion de todas las posiciones de las filas con senal (el cargador de entreno no emite objetivos -1;
con -1 la media sobre validos de C deja de ser 1). Evaluacion e inferencia no pasan por aqui.
QF2_PESO_CTX_PRED=1: todos los tokens pesan segun una tabla cubo-de-rasgos -> beneficio medio (qf2_peso_rasgos), alimentada con el
beneficio medido de las filas con senal de la llamada anterior; cubos sin datos pesan 1; C permuta esos pesos en todo el micro-lote.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

import qf2_peso_rasgos as rasgos

_MASCARAS = {}


@dataclass(frozen=True)
class Cfg:
    modo: str = ""
    k: int = 16
    fraccion: float = 0.125
    alfa: float = 1.0
    tope: float = 4.0
    uniforme: float = 0.5
    semilla: int = 0
    pred: bool = False
    decaimiento: float = 0.99
    minc: float = 8.0

    @property
    def activo(self) -> bool:
        return self.modo != ""


def desde_entorno() -> Cfg:
    modo = os.environ.get("QF2_PESO_CTX", "").strip().lower()
    if modo not in ("", "b", "c"):
        raise ValueError(f"QF2_PESO_CTX debe ser '', 'b' o 'c': {modo!r}")
    cfg = Cfg(
        modo=modo,
        k=int(os.environ.get("QF2_PESO_CTX_K", "16")),
        fraccion=float(os.environ.get("QF2_PESO_CTX_FRACCION", "0.125")),
        alfa=float(os.environ.get("QF2_PESO_CTX_ALFA", "1.0")),
        tope=float(os.environ.get("QF2_PESO_CTX_TOPE", "4.0")),
        uniforme=float(os.environ.get("QF2_PESO_CTX_UNIFORME", "0.5")),
        semilla=int(os.environ.get("QF2_PESO_CTX_SEMILLA", "0")),
        pred=os.environ.get("QF2_PESO_CTX_PRED", "0").strip() == "1",
        decaimiento=float(os.environ.get("QF2_PESO_CTX_PRED_DECAIMIENTO", "0.99")),
        minc=float(os.environ.get("QF2_PESO_CTX_PRED_MINC", "8")),
    )
    if cfg.activo:
        if not (math.isfinite(cfg.alfa) and math.isfinite(cfg.tope)) or cfg.k < 2 or cfg.k % 2 or not 0 < cfg.fraccion <= 1 or cfg.alfa < 0 or cfg.tope < 1 or not 0 <= cfg.uniforme < 1 or not 0 < cfg.decaimiento <= 1 or cfg.minc <= 0:
            raise ValueError(f"QF2_PESO_CTX con parametros fuera de rango: {cfg}")
    return cfg


def filas_con_senal(cfg: Cfg, filas: int, paso: int, micro: int) -> list[int]:
    objetivo = cfg.fraccion * filas
    turno = paso + micro
    if objetivo >= 1:
        ns = min(filas, round(objetivo))
        return sorted((turno * ns + i) % filas for i in range(ns))
    periodo = round(1.0 / objetivo)
    return [(turno // periodo) % filas] if turno % periodo == 0 else []


def mascaras(t: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    clave = (t, k, str(device))
    if clave not in _MASCARAS:
        p = torch.arange(t)
        j0 = p % k
        usa_desfase = (p >= k // 2) & (p < t - k // 2) & (j0 < k // 2)
        elegible = (p >= k).float()
        _MASCARAS[clave] = (usa_desfase.float().to(device), elegible.to(device))
    return _MASCARAS[clave]


def contexto_elegido(t: int, k: int) -> torch.Tensor:
    p = torch.arange(t)
    j0 = p % k
    j1 = (p - k // 2) % k
    desfase = (p >= k // 2) & (p < t - k // 2) & (j0 < k // 2)
    return torch.where(desfase, j1, j0)


def nll_corto(nll_fn, x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    n, t = x.shape
    if t % k:
        raise ValueError(f"QF2_PESO_CTX: la longitud {t} no es multiplo de K={k}")
    h = k // 2
    xc = torch.cat([x.reshape(-1, k), x[:, h:t - h].reshape(-1, k)], dim=0)
    yc = torch.cat([y.reshape(-1, k), y[:, h:t - h].reshape(-1, k)], dim=0)
    with torch.no_grad():
        sal = nll_fn(xc, yc).float()
    n0 = n * (t // k)
    a = sal[:n0].reshape(n, t)
    b = torch.nn.functional.pad(sal[n0:].reshape(n, t - k), (h, h))
    desfase, _ = mascaras(t, k, x.device)
    return a * (1.0 - desfase) + b * desfase


def pesos_b(nll_lleno: torch.Tensor, nll_c: torch.Tensor, y: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    with torch.no_grad():
        vale = (y != -1).float()
        _, elegible = mascaras(y.shape[1], cfg.k, y.device)
        ben = torch.clamp(nll_c - nll_lleno.detach().float(), min=0.0) * elegible * vale
        media_ben = ben.sum() / (elegible * vale).sum().clamp_min(1.0)
        bruto = torch.clamp(1.0 + cfg.alfa * ben / (media_ben + 1e-6), max=cfg.tope)
        media_bruto = (bruto * vale).sum() / vale.sum().clamp_min(1.0)
        return cfg.uniforme + (1.0 - cfg.uniforme) * bruto / media_bruto


def permutacion(n: int, semilla: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(semilla)
    return torch.randperm(n, generator=g)


def pesos_filas(cfg: Cfg, nll_lleno, nll_c, y, semilla: int) -> torch.Tensor:
    w = pesos_b(nll_lleno, nll_c, y, cfg)
    if cfg.modo == "c":
        w = w.reshape(-1)[permutacion(w.numel(), semilla).to(w.device)].reshape(w.shape)
    return w


def ponderar(cfg: Cfg, nll_fn, x: torch.Tensor, y: torch.Tensor, nll: torch.Tensor, paso: int, micro: int) -> torch.Tensor:
    if cfg.pred:
        return ponderar_pred(cfg, nll_fn, x, y, nll, paso, micro)
    vale = y != -1
    sel = filas_con_senal(cfg, x.shape[0], paso, micro)
    if not sel:
        return nll.sum() / vale.sum()
    nc = nll_corto(nll_fn, x[sel], y[sel], cfg.k)
    semilla = cfg.semilla * 1000003 + paso * 1009 + micro
    w = pesos_filas(cfg, nll[sel], nc, y[sel], semilla)
    filas = [w[sel.index(r)] if r in sel else torch.ones_like(nll[r], dtype=w.dtype) for r in range(x.shape[0])]
    return (torch.stack(filas) * nll).sum() / vale.sum()


_PRED = {}


def reiniciar() -> None:
    _PRED.clear()


def predictor(cfg: Cfg, paso: int, micro: int) -> rasgos.Predictor:
    p = _PRED.get(cfg)
    if p is None or (paso, micro) <= p.ultimo:
        p = _PRED[cfg] = rasgos.Predictor(cfg.decaimiento, cfg.minc)
    p.ultimo = (paso, micro)
    return p


def pesos_pred(cfg: Cfg, p: rasgos.Predictor, xh: np.ndarray, yh: np.ndarray, semilla: int):
    p.consumir()
    cubo = p.cubos(xh, yh)
    elegible = np.broadcast_to(np.arange(xh.shape[1]) >= cfg.k, xh.shape)
    bruto = np.where(elegible, p.tabla.bruto(cubo, cfg.alfa, cfg.tope), 1.0)
    vale = yh != -1
    w = rasgos.normalizar(bruto, vale, cfg.uniforme)
    if cfg.modo == "c":
        w = w.reshape(-1)[permutacion(w.size, semilla).numpy()].reshape(w.shape)
    return w, cubo, elegible & vale


def ponderar_pred(cfg: Cfg, nll_fn, x: torch.Tensor, y: torch.Tensor, nll: torch.Tensor, paso: int, micro: int) -> torch.Tensor:
    p = predictor(cfg, paso, micro)
    w, cubo, medible = pesos_pred(cfg, p, x.detach().cpu().numpy(), y.detach().cpu().numpy(), cfg.semilla * 1000003 + paso * 1009 + micro)
    sel = filas_con_senal(cfg, x.shape[0], paso, micro)
    if sel:
        nc = nll_corto(nll_fn, x[sel], y[sel], cfg.k)
        p.pendiente = (torch.clamp(nc - nll[sel].detach().float(), min=0.0), cubo[sel], medible[sel])
    return (torch.from_numpy(w).to(x.device) * nll).sum() / (y != -1).sum()
''',
    "qf2_peso_rasgos.py": r'''"""Rasgos de host y tabla para QF2_PESO_CTX_PRED=1: predice el beneficio del contexto de cada token sin medirlo.

Rasgos por posicion i (x = entrada, y = objetivo, contexto = x[0..i]), todos en numpy con np.unique por fila, sin sorteos:
rareza del objetivo (-log2 de su frecuencia en los objetivos ya vistos en la corrida, cubos de 2 bits), distancia a la
primera aparicion del objetivo en el contexto (0 ninguna, 1 <=16, 2 <=64, 3 <=256, 4 mas) y repeticion lejana: 1 si el
bigrama (x[i], y[i]) ya aparecio como par (x[j], y[j]) con j <= i-17, 2 si ademas el trigrama (x[i-1], x[i], y[i]). Cubo =
(rareza * 5 + distancia) * 3 + repeticion, NB cubos. La tabla guarda sumas y cuentas con decaimiento del beneficio medido
(nll corto - nll completo, >= 0) por cubo; un cubo con menos de MINC de cuenta pesa bruto 1.
"""
from __future__ import annotations

import numpy as np

NB = 16 * 5 * 3
M = 1 << 20
V0 = 32768
LEJOS = 16


def rasgos_fila(x: np.ndarray, y: np.ndarray, rareza: np.ndarray) -> np.ndarray:
    t = x.shape[0]
    i = np.arange(t)
    uniq, primero = np.unique(x, return_index=True)
    pos = np.minimum(np.searchsorted(uniq, y), len(uniq) - 1)
    s = primero[pos]
    esta = (uniq[pos] == y) & (s <= i)
    d = i + 1 - s
    dist = np.where(~esta, 0, np.where(d <= 16, 1, np.where(d <= 64, 2, np.where(d <= 256, 3, 4))))
    k2 = (x + 1) * M + (y + 1)
    _, f2, inv2 = np.unique(k2, return_index=True, return_inverse=True)
    primero2 = f2[inv2.reshape(-1)]
    rep = (primero2 <= i - LEJOS - 1).astype(np.int64)
    k3 = np.empty(t, dtype=np.int64)
    k3[0] = -1
    k3[1:] = (x[:-1] + 1) * (M * M) + k2[1:]
    _, f3, inv3 = np.unique(k3, return_index=True, return_inverse=True)
    primero3 = f3[inv3.reshape(-1)]
    rep += ((primero3 <= i - LEJOS - 1) & (i >= 1)).astype(np.int64)
    return (rareza * 5 + dist) * 3 + rep


class Tabla:
    def __init__(self, decaimiento: float, minc: float):
        self.decaimiento, self.minc = decaimiento, minc
        self.sumas = np.zeros(NB)
        self.cuentas = np.zeros(NB)

    def actualizar(self, cubo: np.ndarray, ben: np.ndarray) -> None:
        cubo, ben = cubo.reshape(-1), ben.reshape(-1).astype(np.float64)
        self.sumas = self.decaimiento * self.sumas + np.bincount(cubo, weights=ben, minlength=NB)
        self.cuentas = self.decaimiento * self.cuentas + np.bincount(cubo, minlength=NB)

    def bruto(self, cubo: np.ndarray, alfa: float, tope: float) -> np.ndarray:
        total = self.cuentas.sum()
        media = self.sumas.sum() / total if total > 0 else 0.0
        if media <= 0.0:
            return np.ones(cubo.shape)
        por_cubo = self.sumas / np.maximum(self.cuentas, 1e-12)
        r = np.where(self.cuentas >= self.minc, np.minimum(1.0 + alfa * por_cubo / media, tope), 1.0)
        return r[cubo]


class Predictor:
    def __init__(self, decaimiento: float, minc: float):
        self.tabla = Tabla(decaimiento, minc)
        self.conteo = np.zeros(M, dtype=np.int64)
        self.total = 0
        self.pendiente = None
        self.ultimo = (-1, -1)

    def cubos(self, xh: np.ndarray, yh: np.ndarray) -> np.ndarray:
        xh, yh = xh.astype(np.int64, copy=False), yh.astype(np.int64, copy=False)
        if int(xh.max()) >= M - 1 or int(yh.max()) >= M - 1:
            raise ValueError("QF2_PESO_CTX_PRED: id de token fuera de 2^20 - 1")
        yv = np.maximum(yh, 0)
        r = -np.log2((self.conteo[yv] + 1.0) / (self.total + V0))
        rareza = np.clip(np.floor(r / 2.0), 0, 15).astype(np.int64)
        out = np.stack([rasgos_fila(xh[f], yh[f], rareza[f]) for f in range(xh.shape[0])])
        v = yh[yh != -1]
        np.add.at(self.conteo, v, 1)
        self.total += v.size
        return out

    def consumir(self) -> None:
        if self.pendiente is None:
            return
        ben, cubo, mascara = self.pendiente
        self.pendiente = None
        b = ben.detach().float().cpu().numpy()
        self.tabla.actualizar(cubo[mascara], b[mascara])


def normalizar(bruto: np.ndarray, vale: np.ndarray, uniforme: float) -> np.ndarray:
    n = max(int(vale.sum()), 1)
    media = (bruto * vale).sum() / n if vale.any() else 1.0
    return (uniforme + (1.0 - uniforme) * bruto / media).astype(np.float32)
''',
    "qf2_pkm.py": r'''"""QF2_PKM_FILAS: product-key memory (PKM) residual branch in one central block.

After block n_layer // 2 the normalised residual stream of each position is projected to two query halves of dk dims. Each half is
scored against its own bank of sqrt(FILAS) sub-keys, the top k of each half are combined into k * k candidate rows (row = i * sqrt(FILAS) + j),
the best k of those are kept and weighted by a softmax over their scores. The weighted sum of the selected value rows (a FILAS x VALOR
table read by index) is projected back to n_embd and added to the residual stream through one learned scalar. Everything is computed
per position from that position's state only: no mixing across positions or batch rows. The module is built with a forked RNG and
initialised last, so with FILAS = 0 the model and its initialisation stream are untouched.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ESCALA_INICIAL = 0.1
VALOR_STD = 0.5


def raiz_exacta(filas: int) -> int:
    r = math.isqrt(filas)
    if r * r != filas:
        raise ValueError(f"QF2_PKM_FILAS={filas} debe ser un cuadrado perfecto")
    return r


def comprobar(filas: int, k: int, valor: int, dk: int) -> None:
    r = raiz_exacta(filas)
    if not 1 <= k <= r:
        raise ValueError(f"QF2_PKM_K={k} debe estar entre 1 y sqrt(filas)={r}")
    if valor < 1 or dk < 1:
        raise ValueError(f"QF2_PKM_VALOR={valor} y QF2_PKM_DK={dk} deben ser positivos")


def seleccionar(s1: torch.Tensor, s2: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    r = s1.size(-1)
    v1, i1 = s1.topk(k, dim=-1)
    v2, i2 = s2.topk(k, dim=-1)
    comb = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
    puntos, pos = comb.topk(k, dim=-1)
    a = torch.div(pos, k, rounding_mode="floor")
    b = pos - a * k
    filas = i1.gather(-1, a) * r + i2.gather(-1, b)
    return puntos, filas


class PKM(nn.Module):
    def __init__(self, filas: int, n_embd: int, k: int, valor: int, dk: int):
        super().__init__()
        comprobar(filas, k, valor, dk)
        self.filas, self.k, self.dk, self.r = filas, k, dk, math.isqrt(filas)
        with torch.random.fork_rng(devices=[]):
            self.valores = nn.Embedding(filas, valor)
        self.consulta = nn.Parameter(torch.empty(2 * dk, n_embd))
        self.subclaves = nn.Parameter(torch.empty(2 * self.r, dk))
        self.salida = nn.Parameter(torch.empty(n_embd, valor))
        self.escala = nn.Parameter(torch.full((1,), ESCALA_INICIAL))

    @torch.no_grad()
    def iniciar(self) -> None:
        n_embd = self.consulta.size(1)
        s = math.sqrt(3.0) * n_embd ** -0.5
        nn.init.uniform_(self.consulta, -s, s)
        nn.init.normal_(self.subclaves, mean=0.0, std=self.dk ** -0.5)
        nn.init.normal_(self.valores.weight, mean=0.0, std=VALOR_STD)
        nn.init.uniform_(self.salida, -s, s)

    def puntuaciones(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.linear(x, self.consulta.to(x.dtype))
        claves = self.subclaves.to(x.dtype)
        ancho = torch.promote_types(x.dtype, torch.float32)
        s1 = F.linear(q[..., :self.dk], claves[:self.r]).to(ancho)
        s2 = F.linear(q[..., self.dk:], claves[self.r:]).to(ancho)
        return s1, s2

    def forward(self, x: torch.Tensor, tabla) -> torch.Tensor:
        s1, s2 = self.puntuaciones(x)
        puntos, filas = seleccionar(s1, s2, self.k)
        peso = F.softmax(puntos, dim=-1).to(x.dtype)
        v = tabla(self.valores, filas).to(x.dtype)
        mezcla = (v * peso.unsqueeze(-1)).sum(dim=-2)
        return self.escala.to(x.dtype) * F.linear(mezcla, self.salida.to(x.dtype))
''',
    "qf2_plano.py": r'''"""Transient Muon slab (QF2_PLANO_TRANSITORIO, default OFF).

The flat slab of RepartoMuon (world * chunk elements, 3.11 GB per rank at the d24 shape) is only used inside the
optimizer step and restaurar(). With the switch ON it is allocated at zero when the step begins and dropped when it
ends, so the allocator can hand the same bytes to activations and gradients during forward and backward.
"""

from __future__ import annotations

import torch


class Losa:
    def __init__(self, numel: int, dtype: torch.dtype, device: torch.device):
        self.numel, self.dtype, self.device = numel, dtype, device
        self.aperturas = 0

    def abrir(self) -> torch.Tensor:
        self.aperturas += 1
        return torch.zeros(self.numel, dtype=self.dtype, device=self.device)

    def cerrar(self) -> None:
        return None

    def bytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()
''',
    "qf2_precarga.py": r'''"""QF2_PRECARGA: the training loader's best-fit packing and tokenization run in a thread while the device computes.

The kit calls next(train_loader) after synchronize(device), inside dt, so on every step the device waits for the CPU
packing of the next batch (and, every few steps, for a tokenizer refill); with 64 ranks the step pays the slowest one.
Here a daemon thread drives prepare.make_dataloader on the CPU (same generator, same order, so the batches are the
same), clones each yielded batch (the generator reuses its buffers) into a queue of PROFUNDIDAD, and the main thread
only copies the ready batch into two persistent device buffers, as the kit's loader does. No device work runs in the
thread. Phase 1: ARCH_PREFETCH (P62, ../trainium-frontier/experiments/log/20260908T032945Z.md) took data loading from
0.007-0.037 s per step to 0.000.
"""
from __future__ import annotations

import queue
import threading

import torch

import qf2_hilos_nice as nice

PROFUNDIDAD = 2
ESPERA_S = 120.0


class Precarga:
    def __init__(self, fabrica, device, profundidad: int = PROFUNDIDAD, memhost=None, forma_mem=None, corpus=None):
        self.device = torch.device(device)
        self.memhost = memhost
        self.forma_mem = forma_mem
        self.corpus = corpus
        self.e = self.c = self.k = None
        self.producidos = 0
        self.cola: queue.Queue = queue.Queue(maxsize=profundidad)
        self.parar = threading.Event()
        self.error: BaseException | None = None
        self.x = self.y = None
        self.servidos = 0
        self.esperas = 0
        self.hilo = threading.Thread(target=self._correr, args=(fabrica,), name="qf2_precarga", daemon=True)
        self.hilo.start()

    def _poner(self, item) -> bool:
        while not self.parar.is_set():
            try:
                self.cola.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _correr(self, fabrica) -> None:
        try:
            nice.ceder("precarga")
            for x, y, estado in fabrica():
                item = (x.clone(), y.clone(), dict(estado))
                if self.corpus is not None:
                    c, k = self.corpus(item[0] if self.forma_mem is None else item[0].reshape(self.forma_mem))
                    item = item + (c.reshape(x.shape), k.reshape(x.shape))
                if self.memhost is not None:
                    self.memhost.registrar(self.producidos, item[0] if self.forma_mem is None else item[0].reshape(self.forma_mem))
                    self.producidos += 1
                if not self._poner(item):
                    return
        except BaseException as e:
            self.error = e
        self._poner(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cola.empty():
            self.esperas += 1
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            raise RuntimeError(f"qf2_precarga: sin lote en {ESPERA_S:.0f} s (hilo vivo={self.hilo.is_alive()})") from self.error
        if item is None:
            raise RuntimeError("qf2_precarga: el hilo del cargador terminó") from self.error
        x, y, estado = item[:3]
        if self.corpus is not None:
            if self.c is None:
                self.c = torch.empty(item[3].shape, dtype=item[3].dtype, device=self.device)
                self.k = torch.empty(item[4].shape, dtype=item[4].dtype, device=self.device)
            self.c.copy_(item[3])
            self.k.copy_(item[4])
        if self.x is None:
            self.x = torch.empty(x.shape, dtype=x.dtype, device=self.device)
            self.y = torch.empty(y.shape, dtype=y.dtype, device=self.device)
        self.x.copy_(x)
        self.y.copy_(y)
        if self.memhost is not None:
            e = self.memhost.E(self.servidos)
            if self.e is None:
                self.e = torch.empty(e.shape, dtype=e.dtype, device=self.device)
            self.e.copy_(e)
        self.servidos += 1
        return self.x, self.y, estado

    def cerrar(self) -> None:
        self.parar.set()
        while True:
            try:
                self.cola.get_nowait()
            except queue.Empty:
                break
        self.hilo.join(timeout=10.0)

    def resumen(self) -> str:
        return f"qf2_precarga: servidos={self.servidos} esperas={self.esperas} profundidad={self.cola.maxsize}"
''',
    "qf2_promedio.py": r'''from __future__ import annotations

import torch

FRACCIONES = (0.9, 0.925, 0.95, 0.975)
ESTADOS = len(FRACCIONES) + 1


class Promedio:
    def __init__(self, params, en_host: bool = False, antes=None):
        self.params = list(params)
        self.antes = antes
        self.en_host = en_host
        self.suma = [torch.zeros(p.shape, dtype=torch.float32, device="cpu" if en_host else p.device) for p in self.params]
        with torch.no_grad():
            for s, p in zip(self.suma, self.params):
                s.copy_(p)
                s.add_(p.to(s.device))
                s.div_(float(ESTADOS))
                s.zero_()
        self.pendientes = list(FRACCIONES)
        self.pasos: list[int] = []

    def descripcion(self) -> str:
        n = sum(s.numel() for s in self.suma)
        return f"qf2_promedio: fracciones={','.join(map(str, FRACCIONES))},1.0 tensores={len(self.suma)} destino={'host' if self.en_host else 'dispositivo'} acumulador_fp32_MB={4 * n / 2**20:.1f}"

    @torch.no_grad()
    def _sumar(self, pasos: int, veces: int) -> None:
        if self.antes is not None:
            self.antes()
        for s, p in zip(self.suma, self.params):
            q = p.to(s.device)
            for _ in range(veces):
                s.add_(q)
        self.pasos.extend([pasos] * veces)

    def tras_paso(self, pasos: int, progreso: float, final: bool) -> None:
        cruzadas = sum(1 for f in self.pendientes if progreso >= f)
        self.pendientes = self.pendientes[cruzadas:]
        veces = cruzadas
        if final:
            veces += len(self.pendientes) + 1
            self.pendientes = []
        if veces:
            self._sumar(pasos, veces)

    @torch.no_grad()
    def instalar(self) -> list[int]:
        if len(self.pasos) != ESTADOS:
            raise RuntimeError(f"qf2_promedio: {len(self.pasos)} estados sumados, se esperaban {ESTADOS}")
        for s, p in zip(self.suma, self.params):
            s.div_(float(ESTADOS))
            p.copy_(s)
        self.suma = []
        return self.pasos
''',
    "qf2_puntos.py": r'''"""Resumable training points (QF2_PUNTOS_S=N, QF2_REANUDA=1; both default OFF).

QF2_PUNTOS_S=N: every N seconds of training (the run's own budget clock, reduced with MAX across ranks inside the existing
control all_reduce, so every rank decides the same step) the run writes <out-dir>/puntos/p<step>/: rank<r>.pt per rank
(optimizer shard, loader position and pending batch, RNG, loop counters, guard/queue/averaging counters, own bf16-inexact
parameter ranges with QF2_COM_GATHER_BF16) and modelo.pt from rank 0. Each file goes to a temporary name, is fsynced and
renamed; after a barrier rank 0 writes HECHO the same way, so a point without HECHO never counts. Only the two newest
complete points are kept. The write time is taken off the budget clock.
QF2_REANUDA=1: every rank loads the newest complete point of the same world size and the loop continues from it with the
trained seconds already charged. The first TRAIN_STARTUP_STEPS_EXCLUDED resumed steps (recompilation) are left out of the
clock and of the recent step times, as the runner does with startup. Without a point the run starts from zero and says so.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import qf2_puntos_estado as est
from prepare import TRAIN_STARTUP_STEPS_EXCLUDED

LIBRES = ("QF2_PUNTOS_S", "QF2_REANUDA", "QF2_RELOJ_LOG", "QF2_RELOJ_TRAMOS")
ARGS = ("depth", "aspect_ratio", "head_dim", "seq_len", "device_batch_size", "total_batch_size", "num_steps",
        "target_param_data_ratio", "compile")


def borrar(d: Path) -> None:
    try:
        for f in d.iterdir():
            f.unlink(missing_ok=True)
        d.rmdir()
    except OSError:
        pass


def escribir(objeto, ruta: Path) -> None:
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "wb") as fh:
        if isinstance(objeto, (bytes, bytearray)):
            fh.write(objeto)
        else:
            torch.save(objeto, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, ruta)


def configuracion(args) -> dict:
    banderas = {k: v for k, v in os.environ.items() if k.startswith("QF2_") and k not in LIBRES}
    return {"env": dict(sorted(banderas.items())), "args": {k: getattr(args, k) for k in ARGS}}


def distribuido() -> bool:
    return dist.is_available() and dist.is_initialized()


class Puntos:
    def __init__(self, cada: str, reanuda: bool, out_dir: str, rank: int, mundo: int, args, avisar=print):
        self.cada_ms = round(float(cada) * 1000) if cada.strip() not in ("", "0") else 0
        if self.cada_ms < 0:
            raise ValueError(f"QF2_PUNTOS_S must be >= 0, got {cada!r}")
        self.reanuda, self.rank, self.mundo, self.avisar = reanuda, rank, mundo, avisar
        self.dir = Path(out_dir) / "puntos"
        self.config = configuracion(args)
        self.max_train_seconds = args.max_train_seconds
        self.proximo_ms = self.cada_ms
        self.ultimo_ms = 0
        self.calentar = 0
        self.entrenado_s = 0.0

    def descripcion(self) -> str:
        return f"qf2_puntos: cada_s={self.cada_ms / 1000 if self.cada_ms else '-'} reanuda={int(self.reanuda)} dir={self.dir}"

    def completos(self) -> list:
        salida = []
        for d in self.dir.glob("p*"):
            try:
                meta = json.loads((d / "HECHO").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("mundo") == self.mundo:
                salida.append((int(meta["paso"]), d))
        return sorted(salida)

    def rotar(self) -> None:
        guardar = {d for _paso, d in self.completos()[-2:]}
        for d in self.dir.glob("p*"):
            if d not in guardar:
                borrar(d)

    def toca(self, control) -> bool:
        if not self.cada_ms:
            return False
        ms = int(control[2].item())
        if ms < self.ultimo_ms:
            self.proximo_ms = (ms // self.cada_ms + 1) * self.cada_ms
        self.ultimo_ms = ms
        return ms >= self.proximo_ms

    def calienta(self) -> bool:
        if self.calentar <= 0:
            return False
        self.calentar -= 1
        return True

    def guardar(self, paso: int, bucle: dict, modelo, opt, cargador, extras: dict, bf16: bool, device) -> float:
        inicio = time.monotonic()
        entrenado_ms = bucle["entrenado_ms"]
        d = self.dir / f"p{paso:09d}"
        d.mkdir(parents=True, exist_ok=True)
        propio = {"paso": paso, "mundo": self.mundo, "rango": self.rank, "bf16": bf16, "config": self.config,
                  "bucle": {**bucle, "entrenado_s": entrenado_ms / 1000.0}, "rng": est.estado_rng(device),
                  "opt": est.estado_opt(opt), "datos": cargador.estado(), "extras": est.estado_extras(extras),
                  "modelo": est.modelo_propio(modelo) if bf16 else None}
        escribir(propio, d / f"rank{self.rank}.pt")
        if self.rank == 0:
            promedio = extras.get("promedio")
            escribir({"modelo": est.modelo_base(modelo, bf16), "bf16": bf16,
                      "promedio": [s.detach().to("cpu", copy=True) for s in promedio.suma] if promedio else None}, d / "modelo.pt")
        if distribuido():
            dist.barrier()
        if self.rank == 0:
            escribir(json.dumps({"paso": paso, "mundo": self.mundo, "entrenado_s": entrenado_ms / 1000.0,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode(), d / "HECHO")
            self.rotar()
        self.proximo_ms = (entrenado_ms // self.cada_ms + 1) * self.cada_ms
        segundos = time.monotonic() - inicio
        mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 2**20 if self.rank == 0 else 0.0
        self.avisar(f"qf2_puntos: saved step={paso} trained_s={entrenado_ms / 1000:.1f} write_s={segundos:.2f} "
                    f"dir_MB_seen_by_rank0={mb:.1f} next_at_s={self.proximo_ms / 1000:.1f}")
        return segundos

    def reanudar(self, modelo, opt, cargador, extras: dict, bf16: bool, device) -> dict | None:
        if not self.reanuda:
            return None
        lista = self.completos()
        paso, d = lista[-1] if lista else (-1, None)
        if distribuido():
            t = torch.tensor([paso, -paso], dtype=torch.int64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if int(t[0].item()) != -int(t[1].item()):
                raise RuntimeError(f"qf2_puntos: ranks see different newest points (max {int(t[0])}, min {-int(t[1])})")
        if d is None:
            self.avisar(f"qf2_puntos: no complete point in {self.dir} for world={self.mundo}: starting from zero")
            return None
        propio = torch.load(d / f"rank{self.rank}.pt", map_location="cpu", weights_only=True)
        comun = torch.load(d / "modelo.pt", map_location="cpu", weights_only=True, mmap=True)
        self.comprobar(propio, comun, bf16)
        est.poner_modelo(modelo, comun["modelo"], propio["modelo"])
        est.poner_opt(opt, propio["opt"])
        est.poner_extras(extras, propio["extras"])
        if extras.get("promedio") is not None:
            est.poner_promedio(extras["promedio"], comun["promedio"])
        est.poner_rng(propio["rng"], device)
        lote = cargador.restaurar(propio["datos"])
        bucle = propio["bucle"]
        self.entrenado_s = bucle["entrenado_s"]
        if self.cada_ms:
            self.proximo_ms = (bucle["entrenado_ms"] // self.cada_ms + 1) * self.cada_ms
        self.ultimo_ms = bucle["entrenado_ms"]
        self.calentar = TRAIN_STARTUP_STEPS_EXCLUDED if bucle["step"] >= TRAIN_STARTUP_STEPS_EXCLUDED else 0
        if distribuido():
            dist.barrier()
        if self.rank == 0:
            self.rotar()
        self.avisar(f"qf2_puntos: resumed from {d} step={bucle['step']} trained_s={self.entrenado_s:.1f} "
                    f"tokens={bucle['total_train_tokens']} excluded_steps={self.calentar}")
        return {**bucle, "lote": lote}

    def comprobar(self, propio: dict, comun: dict, bf16: bool) -> None:
        if propio["mundo"] != self.mundo or propio["rango"] != self.rank or propio["bf16"] != bf16 or comun["bf16"] != bf16:
            raise ValueError(f"qf2_puntos: point world={propio['mundo']} rank={propio['rango']} bf16={propio['bf16']} "
                             f"does not match world={self.mundo} rank={self.rank} bf16={bf16}")
        guardada, ahora = propio["config"], self.config
        if guardada != ahora:
            raise ValueError(f"qf2_puntos: point was saved with another configuration: saved={guardada} now={ahora}")
        if self.max_train_seconds != propio["bucle"]["max_train_seconds"]:
            self.avisar(f"qf2_puntos: max_train_seconds {propio['bucle']['max_train_seconds']} -> {self.max_train_seconds}")
''',
    "qf2_puntos_estado.py": r'''"""What a resumable point holds, besides the loader: model, optimizer, RNG and the small QF2 objects (QF2_PUNTOS_S).

Model. Without QF2_COM_GATHER_BF16 every rank holds the same fp32 parameters, so rank 0 alone writes them. With it, each
rank keeps the fp32 master of what it owns and the bf16 rounding of the rest. Rank 0 then writes the bf16 rounding of its
parameters as a shared base, and every rank writes its own values on ranges that cover every element that is not bf16
exact. bf16(base) is what any rank holds wherever its value is bf16 exact, so base plus own ranges rebuilds each rank's
parameters bit for bit, whoever owns what. Ranges merge gaps below HUECO elements: an own value is always right.
Optimizer: every rank writes the state of its shard, keyed by the position of the parameter in param_groups.
"""
from __future__ import annotations

import torch

HUECO = 4096
CAMPOS = {"guarda": ("saltos", "seguidos"), "cola": ("forma",), "promedio": ("pendientes", "pasos")}


def tramos(plano: torch.Tensor) -> list:
    malos = (plano != plano.bfloat16().float()).nonzero().flatten()
    if malos.numel() == 0:
        return []
    cortes = ((malos[1:] - malos[:-1]) > HUECO).nonzero().flatten()
    inicios = [int(malos[0])] + (malos[cortes + 1]).tolist()
    fines = (malos[cortes]).tolist() + [int(malos[-1])]
    return [[a, plano[a:b + 1].clone()] for a, b in zip(inicios, fines)]


@torch.no_grad()
def modelo_base(modelo, bf16: bool) -> dict:
    return {k: v.detach().to("cpu", copy=True).to(torch.bfloat16 if bf16 else v.dtype) for k, v in modelo.state_dict().items()}


@torch.no_grad()
def modelo_propio(modelo) -> dict:
    return {k: tramos(v.detach().to("cpu", copy=True).float().reshape(-1)) for k, v in modelo.state_dict().items()}


@torch.no_grad()
def poner_modelo(modelo, base: dict, propio: dict | None) -> None:
    estado = {}
    for k, t in modelo.state_dict().items():
        v = base[k].to(t.dtype)
        if propio is not None:
            plano = v.reshape(-1).clone()
            for a, valores in propio[k]:
                plano[a:a + valores.numel()] = valores
            v = plano.view(t.shape)
        estado[k] = v
    modelo.load_state_dict(estado, strict=True)


def parametros(opt) -> list:
    return [p for g in opt.param_groups for p in g["params"]]


def estado_opt(opt) -> dict:
    salida = {}
    for i, p in enumerate(parametros(opt)):
        s = opt.state.get(p)
        if s:
            salida[i] = {k: {"t": v.detach().to("cpu", copy=True), "cpu": v.device.type == "cpu"} if torch.is_tensor(v) else v
                         for k, v in s.items()}
    return salida


def poner_opt(opt, estado: dict) -> None:
    params = parametros(opt)
    if estado and max(estado) >= len(params):
        raise ValueError(f"qf2_puntos: optimizer state for parameter {max(estado)} but only {len(params)} parameters")
    for i, s in estado.items():
        p = params[int(i)]
        opt.state[p] = {k: (v["t"] if v["cpu"] else v["t"].to(p.device)) if isinstance(v, dict) else v for k, v in s.items()}


def estado_rng(device) -> dict:
    salida = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        salida["cuda"] = torch.cuda.get_rng_state(device)
    return salida


def poner_rng(estado: dict, device) -> None:
    torch.set_rng_state(estado["cpu"])
    if "cuda" in estado:
        torch.cuda.set_rng_state(estado["cuda"], device)


def estado_extras(extras: dict) -> dict:
    return {k: {c: getattr(o, c) for c in CAMPOS[k]} for k, o in extras.items() if o is not None}


def poner_extras(extras: dict, estado: dict) -> None:
    activos = {k for k, o in extras.items() if o is not None}
    if activos != set(estado):
        raise ValueError(f"qf2_puntos: point saved with {sorted(estado)} but this run has {sorted(activos)}")
    for k, campos in estado.items():
        for c, v in campos.items():
            setattr(extras[k], c, v)


@torch.no_grad()
def poner_promedio(promedio, suma: list) -> None:
    if len(suma) != len(promedio.suma):
        raise ValueError(f"qf2_puntos: {len(suma)} saved averaging tensors, run has {len(promedio.suma)}")
    for s, t in zip(promedio.suma, suma):
        s.copy_(t.to(s.device))
''',
    "qf2_recolocado.py": r'''"""QF2_RECOLOCADO: una fraccion fija de filas de cada lote de entreno se sustituye por filas de ejemplos recolocados.

Un hilo propio lee documentos de los shards de entreno (00000-06499, los que lista prepare.list_shard_paths con
PUBLIC_TRAIN_SHARDS, con el lector de parquet de prepare), construye ejemplos (qf2_recolocado_ejemplos), los tokeniza
con el tokenizador del kit (cada ejemplo empieza en BOS, como los documentos) y los empaqueta en filas del ancho del
cargador. envolver() sustituye en cada lote, antes de que lo clone la precarga, n filas: n sale de un acumulador
(error difundido) para que la fraccion sea exacta a la larga y el numero de filas, y por tanto de tokens, no cambie;
cual de ellas se decide por (semilla, rango, paso). La evaluacion no se toca. Con fraccion 0 cargador() devuelve el
mismo generador del kit.

QF2_RECOLOCADO_CE: con "todo" los objetivos son los del texto completo. Con "continuacion" los objetivos (-1, el
ignore_index de la CE del kit) de los tokens del BOS, del enunciado y de las dos etiquetas valen -1 y solo cuenta la
continuacion; la entrada x no cambia. El corte en tokens es el prefijo comun entre la tokenizacion del ejemplo entero
y la del tramo hasta la segunda etiqueta: si un token cruza la frontera queda fuera de la mascara.

QF2_RECOLOCADO_RAFAGA: vacio deja el comportamiento anterior (un ejemplo por documento recolocado). Con "a:b" cada
documento recolocado es BOS + k ejemplos del mismo tipo sobre tramos distintos del documento fuente (k uniforme en
[a, b] con el mismo rng, qf2_recolocado_ejemplos.documentos_de). Con "continuacion" la mascara se extiende a cada ejemplo
de la rafaga: objetivo -1 desde el final de una continuacion (incluido un token que cruce esa frontera) hasta la segunda
etiqueta del ejemplo siguiente, con el mismo prefijo comun de tokens en cada frontera (qf2_recolocado_ejemplos.fronteras).

QF2_RECOLOCADO_FORMATO: "ejemplos" (por defecto) es todo lo anterior. Con "pares" cada documento del flujo se escribe entero
y en orden como pares etiquetados (qf2_recolocado_pares, tramos de QF2_PARES_TRAMO palabras, etiquetas de QF2_PARES_ETIQUETAS);
exige QF2_RECOLOCADO_CE=todo y QF2_RECOLOCADO_RAFAGA vacia. Se valida aunque la fraccion sea 0.

ocupadas (paquete 302, qf2_filas): si se da, envolver() elige sus filas solo entre ocupadas.libres(paso, filas), las que
otro sustituidor no tomo en ese paso; n sale del mismo acumulador sobre el total de filas, recortado a las libres, y lo
que no cabe queda en el acumulador. Sin ocupadas (por defecto) la eleccion es la de antes.
"""
from __future__ import annotations

import queue
import random
import threading
import time

import torch

import qf2_hilos_nice as nice
import qf2_recolocado_ejemplos as ejemplos
import qf2_recolocado_pares as pares_m

ESPERA_S = 120.0
TOKENIZAR_DE_A = 64
BUFFER_MIN = 64
CE_MODOS = ("todo", "continuacion")


def leer_rafaga(texto: str) -> tuple[int, int] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes) or not 2 <= int(partes[0]) <= int(partes[1]):
        raise ValueError(f"QF2_RECOLOCADO_RAFAGA={texto!r}: se espera 'a:b' con 2 <= a <= b")
    return int(partes[0]), int(partes[1])


def documentos_de_shards(rank: int, world: int, semilla: int, rutas=None):
    import pyarrow.parquet as pq
    if rutas is None:
        import prepare
        import qf2_inicio
        rutas = prepare.list_shard_paths(qf2_inicio.shards_banco() or prepare.PUBLIC_TRAIN_SHARDS, require_all=False)
    rutas = sorted(rutas)
    j = 0
    while True:
        pf = pq.ParquetFile(rutas[(rank + world * j + 7919 * semilla) % len(rutas)])
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g).column("text").to_pylist()
        j += 1


def flujo_ejemplos(documentos, rng, rafaga: tuple[int, int] | None = None):
    for doc in documentos:
        yield from ejemplos.documentos_de(doc, rng, rafaga)


def prefijo_comun(a: list[int], b: list[int]) -> int:
    n = 0
    for u, v in zip(a, b):
        if u != v:
            break
        n += 1
    return n


def tramos_de(tokenizador, lote, ids, bos: int) -> list[list[tuple[int, int]]]:
    fronteras = [ejemplos.fronteras(e) for e in lote]
    previos = iter(tokenizador.encode([f for fs in fronteras for f in fs], prepend=bos, num_threads=1))
    salida = []
    for t, fs in zip(ids, fronteras):
        cortes = [prefijo_comun(t, next(previos)) for _ in fs]
        inicios = [0] + cortes[1::2]
        salida.append([(a, max(a, b)) for a, b in zip(inicios, cortes[0::2])])
    return salida


def empaquetar(tokenizador, flujo, capacidad: int, ce: str = "todo"):
    if ce not in CE_MODOS:
        raise ValueError(f"QF2_RECOLOCADO_CE={ce!r} fuera de {CE_MODOS}")
    bos = tokenizador.get_bos_token_id()
    buffer: list[tuple[list[int], str, list[tuple[int, int]]]] = []
    while True:
        fila: list[int] = []
        tipos: list[str] = []
        ignorar = torch.zeros(capacidad - 1, dtype=torch.bool) if ce == "continuacion" else None
        while len(fila) < capacidad:
            while len(buffer) < BUFFER_MIN:
                lote = [next(flujo) for _ in range(TOKENIZAR_DE_A)]
                ids = tokenizador.encode([e["texto"] for e in lote], prepend=bos, num_threads=1)
                tramos = tramos_de(tokenizador, lote, ids, bos) if ce == "continuacion" else [[]] * len(lote)
                buffer.extend((t, e["tipo"], c) for t, e, c in zip(ids, lote, tramos))
            resto = capacidad - len(fila)
            caben = [k for k, (t, _, _) in enumerate(buffer) if len(t) <= resto]
            k = max(caben, key=lambda q: len(buffer[q][0])) if caben else min(range(len(buffer)), key=lambda q: len(buffer[q][0]))
            t, tipo, tramos = buffer.pop(k)
            if ignorar is not None:
                for ini, fin in tramos:
                    ignorar[max(0, len(fila) + ini - 1):max(0, min(len(fila) + fin - 1, capacidad - 1))] = True
            fila.extend(t[:resto])
            tipos.append(tipo)
        yield torch.tensor(fila, dtype=torch.long), tipos, ignorar


class Fuente:
    def __init__(self, tokenizador, documentos, semilla: int, rank: int, capacidad: int, ce: str = "todo", profundidad: int = 4,
                 rafaga: tuple[int, int] | None = None, pares: tuple | None = None):
        self.cola: queue.Queue = queue.Queue(maxsize=profundidad)
        self.error: BaseException | None = None
        self.cpu_s = 0.0
        self.filas = 0
        rng = random.Random(f"recolocado-ejemplos:{semilla}:{rank}")
        self.hilo = threading.Thread(target=self._correr, args=(tokenizador, documentos, rng, capacidad, ce, rafaga, pares), name="qf2_recolocado", daemon=True)
        self.hilo.start()

    def _correr(self, tokenizador, documentos, rng, capacidad, ce, rafaga, pares=None) -> None:
        try:
            nice.ceder("recolocado")
            t0 = time.thread_time()
            for fila in empaquetar(tokenizador, pares_m.flujo(documentos, rng, *pares) if pares else flujo_ejemplos(documentos, rng, rafaga), capacidad, ce):
                self.cpu_s = time.thread_time() - t0
                self.filas += 1
                self.cola.put(fila)
        except BaseException as e:
            self.error = e
            self.cola.put(None)

    def siguiente(self):
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            raise RuntimeError(f"qf2_recolocado: sin fila en {ESPERA_S:.0f} s") from self.error
        if item is None:
            raise RuntimeError("qf2_recolocado: el hilo de ejemplos terminó") from self.error
        return item


def envolver(base, tokenizador, rank: int, world: int, semilla: int, fraccion: float, documentos=None, registro=None, ce: str = "todo",
             rafaga: tuple[int, int] | None = None, pares: tuple | None = None, ocupadas=None):
    fuente = None
    acumulado = 0.0
    for paso, (x, y, estado) in enumerate(base):
        if fuente is None:
            docs = documentos if documentos is not None else documentos_de_shards(rank, world, semilla)
            fuente = Fuente(tokenizador, docs, semilla, rank, x.shape[1] + 1, ce, rafaga=rafaga, pares=pares)
        acumulado += fraccion * x.shape[0]
        libres = range(x.shape[0]) if ocupadas is None else ocupadas.libres(paso, x.shape[0])
        n = min(int(acumulado + 1e-9), len(libres))
        acumulado -= n
        if n:
            for r in random.Random(f"recolocado-filas:{semilla}:{rank}:{paso}").sample(libres, n):
                fila, tipos, ignorar = fuente.siguiente()
                x[r].copy_(fila[:-1])
                y[r].copy_(fila[1:])
                if ignorar is not None:
                    y[r].masked_fill_(ignorar, -1)
                if registro is not None:
                    registro.append((paso, r, tipos))
        yield x, y, estado


def cargador(base, tokenizador, rank: int, world: int, semilla: int, fraccion: float, ce: str = "todo", rafaga: str = "", formato: str = "ejemplos",
             pares_tramo: str = "8:40", pares_etiquetas: str = "Input/Output", **extra):
    if ce not in CE_MODOS:
        raise ValueError(f"QF2_RECOLOCADO_CE={ce!r} fuera de {CE_MODOS}")
    tramos_rafaga = leer_rafaga(rafaga)
    pares = pares_m.configurar(formato, ce, tramos_rafaga, pares_tramo, pares_etiquetas)
    if fraccion <= 0:
        return base
    return envolver(base, tokenizador, rank, world, semilla, fraccion, ce=ce, rafaga=tramos_rafaga, pares=pares, **extra)
''',
    "qf2_recolocado_ejemplos.py": r'''"""QF2_RECOLOCADO: ejemplos de texto recolocado, construidos solo con texto del propio documento de entreno.

Cada ejemplo tiene dos partes unidas por etiquetas fijas y genericas. La primera parte es texto del documento (o,
para Brackets, sus caracteres de corchete en el orden original) y la segunda es la continuacion literal del
documento justo despues. Nada se calcula: no se invierte, ordena ni cuenta. Regla de la organizacion (guangtai,
26 oct 2026): pares de sustantivos genericos admitidos cuando la primera parte es texto del corpus y la segunda su
continuacion. No se usa ninguna etiqueta de plantilla de tarea.

QF2_RECOLOCADO_RAFAGA (documentos_de con rafaga=(minimo, maximo)): por cada documento fuente y cada tipo se sortea k
uniforme en [minimo, maximo] con el mismo rng y se buscan k ejemplos de ese tipo sobre tramos [i, k) que no se solapen,
como mucho INTENTOS_POR_EJEMPLO * k llamadas al constructor. Los hallados se ordenan por su posicion en el documento y se
unen con SEPARADOR en un solo documento recolocado; con uno solo queda el ejemplo suelto de siempre y con ninguno, nada.
fronteras() da los prefijos de texto que delimitan, en cada ejemplo, el contexto (hasta la segunda etiqueta) y el final
de su continuacion: el empaquetador los convierte en tramos de tokens con objetivo -1.
"""
from __future__ import annotations

import re

ETIQUETAS = {
    "corchetes": ("Brackets: ", "\nClosers: "),
    "serie": ("Series: ", "\nRemainder:"),
    "tramo": ("String: ", "\nTail:"),
}
TIPOS = tuple(ETIQUETAS)
CORCHETES = "()[]{}<>"
RE_CORCHETE = re.compile(r"[()\[\]{}<>]")
_ELEM = r"[\w$%.\-']{1,24}"
RE_SERIE = re.compile(rf"(?<![\w$%.\-']){_ELEM}(?:, {_ELEM}){{5,59}}(?![\w$%.\-'])")
RE_INICIO_PALABRA = re.compile(r"(?<=\s)\S")
LARGO_MIN_DOC = 600
SEPARADOR = "\n"
INTENTOS_POR_EJEMPLO = 3


def texto(tipo: str, a: str, b: str) -> str:
    izq, der = ETIQUETAS[tipo]
    return izq + a + der + b


def hasta_segunda_etiqueta(e: dict) -> str:
    return texto(e["tipo"], e["a"], "")


def corchetes(doc: str, rng) -> dict | None:
    pos = [m.start() for m in RE_CORCHETE.finditer(doc)]
    n1, n2 = rng.randint(5, 32), rng.randint(4, 16)
    if len(pos) < n1 + n2:
        return None
    for _ in range(8):
        t = rng.randrange(0, len(pos) - n1 - n2 + 1)
        if pos[t + n1 + n2 - 1] - pos[t + n1 - 1] <= 1500:
            i, j = pos[t], pos[t + n1 - 1] + 1
            a = "".join(c for c in doc[i:j] if c in CORCHETES)
            b = "".join(doc[p] for p in pos[t + n1:t + n1 + n2])
            return {"tipo": "corchetes", "i": i, "j": j, "k": pos[t + n1 + n2 - 1] + 1, "a": a, "b": b}
    return None


def serie(doc: str, rng) -> dict | None:
    corridas = [m for m in RE_SERIE.finditer(doc) if m.end() - m.start() <= 1200]
    if not corridas:
        return None
    m = rng.choice(corridas)
    elementos = m.group(0).split(", ")
    t = rng.randint(3, len(elementos) - 3)
    p = m.start() + sum(len(e) + 2 for e in elementos[:t]) - 1
    return {"tipo": "serie", "i": m.start(), "j": p, "k": m.end(), "a": doc[m.start():p], "b": doc[p:m.end()]}


def tramo(doc: str, rng) -> dict | None:
    l1, l2 = rng.randint(150, 500), rng.randint(100, 400)
    if len(doc) < max(LARGO_MIN_DOC, l1 + l2 + 200):
        return None
    m = RE_INICIO_PALABRA.search(doc, rng.randrange(0, len(doc) - l1 - l2 - 100))
    if m is None:
        return None
    i = m.start()
    j = doc.find(" ", i + l1)
    if j < 0 or j - i > l1 + 60:
        return None
    fin = re.compile(r"\s").search(doc, j + l2)
    k = fin.start() if fin else len(doc)
    if k - j > l2 + 60:
        return None
    return {"tipo": "tramo", "i": i, "j": j, "k": k, "a": doc[i:j], "b": doc[j:k]}


CONSTRUCTORES = {"corchetes": corchetes, "serie": serie, "tramo": tramo}


def ejemplos_de(doc: str, rng) -> list[dict]:
    salida = []
    for tipo in TIPOS:
        e = CONSTRUCTORES[tipo](doc, rng)
        if e is not None:
            e["texto"] = texto(tipo, e["a"], e["b"])
            salida.append(e)
    return salida


def rafaga_de(doc: str, tipo: str, rng, minimo: int, maximo: int) -> dict | None:
    k = rng.randint(minimo, maximo)
    elegidos: list[dict] = []
    for _ in range(INTENTOS_POR_EJEMPLO * k):
        if len(elegidos) == k:
            break
        e = CONSTRUCTORES[tipo](doc, rng)
        if e is not None and all(e["k"] <= o["i"] or o["k"] <= e["i"] for o in elegidos):
            elegidos.append(e)
    elegidos.sort(key=lambda e: e["i"])
    for e in elegidos:
        e["texto"] = texto(tipo, e["a"], e["b"])
    if len(elegidos) < 2:
        return elegidos[0] if elegidos else None
    return {"tipo": tipo, "ejemplos": elegidos, "texto": SEPARADOR.join(e["texto"] for e in elegidos)}


def documentos_de(doc: str, rng, rafaga: tuple[int, int] | None = None) -> list[dict]:
    if rafaga is None:
        return ejemplos_de(doc, rng)
    salida = []
    for tipo in TIPOS:
        d = rafaga_de(doc, tipo, rng, *rafaga)
        if d is not None:
            salida.append(d)
    return salida


def fronteras(d: dict) -> list[str]:
    if "ejemplos" not in d:
        return [hasta_segunda_etiqueta(d)]
    salida: list[str] = []
    previo = ""
    for n, e in enumerate(d["ejemplos"]):
        if n:
            salida.append(previo)
            previo += SEPARADOR
        salida.append(previo + hasta_segunda_etiqueta(e))
        previo += e["texto"]
    return salida
''',
    "qf2_recolocado_pares.py": r'''"""QF2_RECOLOCADO_FORMATO=pares: cada documento del flujo de recolocado se escribe entero y en orden como pares etiquetados.

El documento se parte en tramos consecutivos c1 c2 c3 ... de k palabras (k uniforme en [minimo, maximo] por tramo, con
el rng del hilo de recolocado; el último tramo se queda con las palabras que sobren). Solo se corta en el blanco que hay
entre dos palabras: cada tramo es el texto del documento byte a byte, con sus espacios y saltos internos, y solo pierde
el blanco de los bordes de corte. Se escribe 'Input: c1\\nOutput: c2\\n\\nInput: c3\\nOutput: c4...' (o 'Q: c1\\nA: c2...');
con un número impar de tramos el último par lleva solo la primera etiqueta. QF2_PARES_ETIQUETAS='ambas' alterna
Input/Output y Q/A por documento escrito, empezando por Input/Output. Un documento sin palabras no da nada.

Etiquetas genéricas fijas, admitidas por la organización el 23 oct (docs/bitacora/20261026T032641Z y 20261026T032911Z):
la primera parte es texto del corpus y la segunda su continuación literal. Nada se calcula y nada viene del bundle de CORE.
"""
from __future__ import annotations

import re

ETIQUETAS = {"Input/Output": ("Input: ", "\nOutput: "), "Q/A": ("Q: ", "\nA: ")}
AMBAS = "ambas"
SEPARADOR_PARES = "\n\n"
FORMATOS = ("ejemplos", "pares")
RE_PALABRA = re.compile(r"\S+")


def leer_tramo(texto: str) -> tuple[int, int]:
    partes = (texto or "").strip().split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes) or not 1 <= int(partes[0]) <= int(partes[1]):
        raise ValueError(f"QF2_PARES_TRAMO={texto!r}: se espera 'min:max' en palabras con 1 <= min <= max")
    return int(partes[0]), int(partes[1])


def leer_etiquetas(texto: str) -> tuple[str, ...]:
    texto = (texto or "").strip()
    if texto == AMBAS:
        return tuple(ETIQUETAS)
    if texto not in ETIQUETAS:
        raise ValueError(f"QF2_PARES_ETIQUETAS={texto!r} fuera de {(*ETIQUETAS, AMBAS)}")
    return (texto,)


def configurar(formato: str, ce: str, rafaga, tramo: str, etiquetas: str) -> tuple | None:
    if formato not in FORMATOS:
        raise ValueError(f"QF2_RECOLOCADO_FORMATO={formato!r} fuera de {FORMATOS}")
    if formato != "pares":
        return None
    if ce != "todo" or rafaga is not None:
        raise ValueError(f"QF2_RECOLOCADO_FORMATO=pares necesita QF2_RECOLOCADO_CE=todo y QF2_RECOLOCADO_RAFAGA vacía (ce={ce!r}, rafaga={rafaga!r})")
    return leer_tramo(tramo), leer_etiquetas(etiquetas)


def tramos(doc: str, rng, minimo: int, maximo: int) -> list[str]:
    palabras = [(m.start(), m.end()) for m in RE_PALABRA.finditer(doc)]
    salida: list[str] = []
    i = 0
    while i < len(palabras):
        j = min(len(palabras), i + rng.randint(minimo, maximo))
        salida.append(doc[palabras[i][0]:palabras[j - 1][1]])
        i = j
    return salida


def texto(partes: list[str], etiquetas: str) -> str:
    izq, der = ETIQUETAS[etiquetas]
    pares = [izq + partes[n] + (der + partes[n + 1] if n + 1 < len(partes) else "") for n in range(0, len(partes), 2)]
    return SEPARADOR_PARES.join(pares)


def flujo(documentos, rng, tramo: tuple[int, int], etiquetas: tuple[str, ...]):
    escritos = 0
    for doc in documentos:
        partes = tramos(doc, rng, *tramo)
        if not partes:
            continue
        nombre = etiquetas[escritos % len(etiquetas)]
        escritos += 1
        yield {"tipo": f"pares:{nombre}", "texto": texto(partes, nombre)}
''',
    "qf2_reparto.py": r'''"""Optimizer communication at many ranks (QF2_COM_ESCALARES, QF2_COM_MUON, QF2_COM_GATHER_BF16; all default OFF).

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
''',
    "qf2_resumen.py": r'''"""QF2_RESUMEN='entrada:salida': causal summary of the current document, read after one block and added before a later one.

After block `entrada` (0-based index of the block loop of GPT.forward) the normalised residual stream h = norm(x) is
averaged over the positions of the current document up to and including each position: m_t is the mean of h_s for s in
[start of the document of t, t]. A document starts at every BOS token, and at the start of the row. Before block `salida`
the residual stream receives x + W_o(norm(m)), where W_o is one D x D matrix without bias initialised to zero, so at
initialisation the model is exactly the one without the branch.

The mean uses matrix products only, with no cumsum, cummax or scan along the sequence (Neuron does not lower them in the
compiled forward): c = L @ [idx == BOS] numbers the documents of each row (L is the T x T lower triangle of ones, in
float32 so the count stays exact), M[b, i, j] = (j <= i) & (c[b, j] == c[b, i]) selects the earlier positions of the same
document, S = M @ h and the count is M.sum(-1). Every output position reads only its own row and positions <= itself.
W_o is built under a forked RNG and zeroed in iniciar(), so the initialisation stream of the rest of the model is untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def capas(texto: str, n_layer: int) -> tuple[int, int]:
    partes = texto.strip().split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes):
        raise ValueError(f"QF2_RESUMEN={texto!r} debe ser 'entrada:salida' con dos enteros")
    entrada, salida = int(partes[0]), int(partes[1])
    if not 0 <= entrada < salida < n_layer:
        raise ValueError(f"QF2_RESUMEN={texto!r} exige 0 <= entrada < salida < n_layer={n_layer}")
    return entrada, salida


def mascara(idx: torch.Tensor, bos: int, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(idx.size(1), device=idx.device)
    inferior = pos.view(1, -1) <= pos.view(-1, 1)
    es_bos = (idx == bos).to(torch.float32)
    c = F.linear(es_bos, inferior.to(torch.float32))
    return (inferior.unsqueeze(0) & (c.unsqueeze(1) == c.unsqueeze(2))).to(dtype)


def media(h: torch.Tensor, idx: torch.Tensor, bos: int) -> torch.Tensor:
    m = mascara(idx, bos, h.dtype)
    suma = torch.matmul(m, h)
    cuenta = m.sum(-1, keepdim=True)
    return suma / cuenta


class Resumen(nn.Module):
    def __init__(self, texto: str, n_layer: int, n_embd: int, bos: int, lineal):
        super().__init__()
        self.capa_entrada, self.capa_salida = capas(texto, n_layer)
        self.bos = bos
        with torch.random.fork_rng(devices=[]):
            self.proy = lineal(n_embd, n_embd, bias=False)

    @torch.no_grad()
    def iniciar(self) -> None:
        nn.init.zeros_(self.proy.weight)

    def media(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return media(h, idx, self.bos)

    def inyectar(self, m: torch.Tensor) -> torch.Tensor:
        return self.proy(F.rms_norm(m, (m.size(-1),)))
''',
    "qf2_union.py": r'''"""Pérdida de la unión del banco CORE (paquete 302): QF2_CE_DIFICILES, QF2_CE_TRAMO, QF2_EQUILIBRIO y QF2_BRIER a la vez.

leer() devuelve None con menos de dos de esos mandos encendidos: entonces train.py va por la rama de su paquete de origen
(qf2_brier.ce_brier_trozos del 298, OBJETIVO del 295, EQUILIBRIO del 300 o ce_trozos), bit a bit. Con dos o más, Union
recibe los estados que entran en la CE por trozos, los objetivos [B, T] y es_bos = (idx == BOS) y devuelve

    w        = w_dificiles * w_equilibrio en los válidos (1 el que esté apagado), renormalizado: w * V / sum(w)
    pérdida  = (1 - f) * sum(w * nll) / V + f * tramo(nll) + lam * sum_válidos(brier) / V

con V los objetivos válidos del micro-lote, nll y brier por token de la CE por trozos (0 en los -1), w_dificiles de
qf2_objetivos.pesos_dificiles y w_equilibrio de qf2_equilibrio (cada uno ya con media 1 sobre los válidos; el producto se
vuelve a normalizar), tramo(nll) el segundo término de qf2_objetivos.perdida_tramo sobre la NLL sin ponderar (f = 0 sin
QF2_CE_TRAMO) y el Brier sin ponderar por w (lam = 0 sin QF2_BRIER). Los pesos no llevan gradiente. Con un solo mando la
fórmula es la de su paquete de origen; la rama de ese paquete la calcula bit a bit.

nll y brier salen juntos de _CETrozosBrierTokens (trozos fijos de filas como ce_trozos, sin one-hot ni probabilidades de
toda la secuencia): con a = d pérdida / d nll y b = d pérdida / d brier por token, dL/dz_j = p_j (a + 2 b (p_j - S2 + p_y))
- [j = y] (a + 2 b p_y) por la derivada del softcap; con a = 1 / V y b = lam / V es el gradiente de qf2_brier. Sin
QF2_BRIER la NLL sale de ce_trozos.ce_trozos_tokens.

Enganche para el 301 (QF2_MAESTRO, sin fusionar): la pérdida mezclada sería (1 - alfa) * (lo que devuelve Union, o la rama
de un solo mando) + alfa * KL(maestro || alumno). Enganche para el 297 (QF2_MARGEN, sin fusionar): su término de margen se
sumaría a la pérdida como otro término aditivo, sobre sus filas, que los sustituidores de qf2_filas evitan.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import qf2_objetivos
from ce_trozos import _trozo, ce_trozos_tokens

MINIMO = 1e-30


class _CETrozosBrierTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo):
        vale = y != -1
        nlls, briers, lses, s2s, pys = [], [], [], [], []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            yc, vc = y[i:i + trozo].clamp_min(0).unsqueeze(1), vale[i:i + trozo]
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            p = torch.exp(zc - lse.unsqueeze(1))
            s2 = (p * p).sum(dim=-1)
            py = p.gather(1, yc).squeeze(1)
            nlls.append(torch.where(vc, lse - zc.gather(1, yc).squeeze(1), 0.0))
            briers.append(torch.where(vc, s2 - 2.0 * py + 1.0, 0.0))
            lses.append(lse)
            s2s.append(s2)
            pys.append(py)
        ctx.save_for_backward(x, w, y, torch.cat(lses), torch.cat(s2s), torch.cat(pys))
        ctx.cap, ctx.vocab, ctx.trozo = cap, vocab, trozo
        return torch.cat(nlls), torch.cat(briers)

    @staticmethod
    def backward(ctx, g_nll, g_brier):
        x, w, y, lse, s2, py = ctx.saved_tensors
        cap, vocab, trozo = ctx.cap, ctx.vocab, ctx.trozo
        vale = (y != -1).float()
        a, b = g_nll.float() * vale, g_brier.float() * vale
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc = x[i:i + trozo], y[i:i + trozo].clamp_min(0).unsqueeze(1)
            ac, bc, pyc = a[i:i + trozo], b[i:i + trozo], py[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d = d * (ac.unsqueeze(1) + 2.0 * bc.unsqueeze(1) * (d + (pyc - s2[i:i + trozo]).unsqueeze(1)))
            d.scatter_add_(1, yc, -(ac + 2.0 * bc * pyc).unsqueeze(1))
            dz = (d * (1.0 - t * t)).to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None


def ce_brier_tokens(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int):
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozosBrierTokens.apply(x, w, targets, cap, vocab, trozo)


@dataclass(frozen=True)
class Union:
    dificiles: object = None
    tramo: object = None
    equilibrio: object = None
    brier: object = None

    def pesos(self, nll: torch.Tensor, vale: torch.Tensor, es_bos: torch.Tensor) -> torch.Tensor:
        w = vale.to(torch.float32)
        if self.dificiles is not None:
            w = w * qf2_objetivos.pesos_dificiles(nll, vale, self.dificiles.factor)
        if self.equilibrio is not None:
            w = w * self.equilibrio.pesos(vale, es_bos)
        return w * (vale.sum() / w.sum().clamp_min(MINIMO))

    def combinar(self, nll: torch.Tensor, brier: torch.Tensor | None, vale: torch.Tensor, es_bos: torch.Tensor) -> torch.Tensor:
        validos = vale.sum()
        perdida = (self.pesos(nll, vale, es_bos) * nll).sum() / validos
        if self.tramo is not None:
            tramo = qf2_objetivos.perdida_tramo(nll, vale, 1.0, self.tramo.tau, self.tramo.k)
            perdida = (1.0 - self.tramo.f) * perdida + self.tramo.f * tramo
        if self.brier is not None:
            perdida = perdida + self.brier.lam * brier.sum() / validos
        return perdida

    def __call__(self, x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, es_bos: torch.Tensor, cap: float, vocab: int,
                 trozo: int) -> torch.Tensor:
        if self.brier is None:
            nll, brier = ce_trozos_tokens(x, w, targets.reshape(-1), cap, vocab, trozo), None
        else:
            nll, brier = ce_brier_tokens(x, w, targets.reshape(-1), cap, vocab, trozo)
            brier = brier.view_as(targets)
        return self.combinar(nll.view_as(targets), brier, targets != -1, es_bos)

    def partes(self) -> list:
        return [p for p in (self.dificiles, self.tramo, self.equilibrio, self.brier) if p is not None]

    def descripcion(self, ruido=None) -> str:
        texto = "; ".join(p.descripcion() for p in self.partes())
        extra = f"; {ruido.descripcion()}" if ruido is not None else ""
        return (f"qf2_union: pérdida = (1 - f) * sum(w * nll) / V + f * tramo(nll) + lam * sum(brier) / V, "
                f"w = w_dificiles * w_equilibrio renormalizado a media 1; {texto}{extra}")


def leer(dificiles, tramo, equilibrio, brier) -> Union | None:
    union = Union(dificiles, tramo, equilibrio, brier)
    return union if len(union.partes()) >= 2 else None
''',
}


def _qf2_completa(destino, modulos):
    import os
    try:
        if hasattr(os, "getuid") and os.stat(destino).st_uid != os.getuid():
            return False
        for nombre, texto in modulos.items():
            with open(os.path.join(destino, nombre), "rb") as fh:
                if fh.read() != texto.encode("utf-8"):
                    return False
    except OSError:
        return False
    return True


def _qf2_escribir(base, destino, modulos):
    import os
    import tempfile
    try:
        os.makedirs(base, exist_ok=True)
        borrador = tempfile.mkdtemp(prefix=os.path.basename(destino) + ".", dir=base)
    except OSError:
        return
    try:
        for nombre, texto in sorted(modulos.items()):
            ruta = os.path.join(borrador, nombre)
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, "wb") as fh:
                fh.write(texto.encode("utf-8"))
        os.rename(borrador, destino)
    except OSError:
        _qf2_descartar(borrador)


def _qf2_descartar(borrador):
    import os
    try:
        for raiz, carpetas, ficheros in os.walk(borrador, topdown=False):
            for f in ficheros:
                os.remove(os.path.join(raiz, f))
            for c in carpetas:
                os.rmdir(os.path.join(raiz, c))
        os.rmdir(borrador)
    except OSError:
        pass


def _qf2_carpeta(modulos):
    import hashlib
    huella = hashlib.sha256()
    for nombre in sorted(modulos):
        huella.update(nombre.encode("utf-8") + b"\0" + modulos[nombre].encode("utf-8") + b"\0")
    return "qf2_modulos_" + huella.hexdigest()[:24]


def _qf2_instalar(modulos):
    import os
    import sys
    import tempfile
    carpeta = _qf2_carpeta(modulos)
    bases = [tempfile.gettempdir(), os.path.join(os.path.expanduser("~"), ".cache")]
    if globals().get("__file__"):
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    for base in bases:
        destino = os.path.join(base, carpeta)
        if not _qf2_completa(destino, modulos):
            _qf2_escribir(base, destino, modulos)
        if _qf2_completa(destino, modulos):
            if destino not in sys.path:
                sys.path.insert(0, destino)
            return destino
    raise ImportError(f"could not materialize the participant modules under any of {bases}")


_QF2_CARPETA = _qf2_instalar(_QF2_MODULOS)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import qf2_diff
import qf2_memoria
import qf2_memoria_anfitrion
import qf2_memoria_ckpt
import qf2_corpus_constructor
import qf2_corpus_mezcla
import qf2_guardado_anticipado
import qf2_memhost_canon
import qf2_memhost_ckpt
import qf2_memhost_cfg
import qf2_hilos_aten
import qf2_hilos_nice
import qf2_memhost_hilo
import qf2_memhost_opt
import qf2_memhost_hojas
import qf2_memhost_sitios
import qf2_memoria_opt
import qf2_mos
import qf2_ngramas
import qf2_peso_contexto
import qf2_pkm
import qf2_resumen

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    PUBLIC_TRAIN_SHARDS,
    Tokenizer,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    list_shard_paths,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 2048
# Best measured baseline package at MBS1/T2048 on all 64 LNC2 ranks.
DEPTH = int(os.environ.get("QF2_DEPTH", "24"))
ASPECT_RATIO = int(os.environ.get("QF2_ASPECT_RATIO", "64"))
HEAD_DIM = 128

DEVICE_BATCH_SIZE = int(os.environ.get("QF2_DEVICE_BATCH_SIZE", "4"))
TOTAL_BATCH_SIZE = int(os.environ.get("QF2_TOTAL_BATCH_SIZE", "1048576"))
NUM_STEPS = -1
TARGET_PARAM_DATA_RATIO = 10.30
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = float(os.environ.get("QF2_MATRIX_LR", "0.02"))
SCALAR_LR = 0.20
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 40
WARMDOWN_RATIO = float(os.environ.get("QF2_WARMDOWN_RATIO", "").strip() or 0.65)
FINAL_LR_FRAC = float(os.environ.get("QF2_FINAL_LR_FRAC", "").strip() or 0.05)
LR_SHAPE = os.environ.get("QF2_LR_SHAPE", "lin").strip()
LR_SCHEDULE = os.environ.get("QF2_LR_SCHEDULE", "reloj")
RELOJ_LOG = os.environ.get("QF2_RELOJ_LOG", "")
RELOJ_TRAMOS = os.environ.get("QF2_RELOJ_TRAMOS", "0") != "0"
RELOJ_STEP_CEILING = 1_000_000
CHECKPOINT_RESERVE_SECONDS = 5.0
MUON_MOMENTUM = 0.95
GRAD_CLIP = 0.0
SEED = int(os.environ.get("QF2_SEED", "1").strip() or 42)
WD_COS = os.environ.get("QF2_WD_COS", "1") != "0"
WD_PICO = float(os.environ.get("QF2_WD_PICO", "0.06").strip() or WEIGHT_DECAY)
SUELO_A = os.environ.get("QF2_SUELO_A", "1") != "0"
SUELO_MOM = SUELO_A or os.environ.get("QF2_SUELO_MOM", "0") != "0"
SUELO_ESC = SUELO_A or os.environ.get("QF2_SUELO_ESC", "0") != "0"
SUELO_LR = os.environ.get("QF2_SUELO_LR", "0") != "0"
SUELO_INIT = SUELO_A or os.environ.get("QF2_SUELO_INIT", "0") != "0"
OPT = os.environ.get("QF2_OPT", "")
OPT_BETA2 = float(os.environ.get("QF2_OPT_BETA2", "0.95"))
OPT_EPS = float(os.environ.get("QF2_OPT_EPS", "1e-8"))
OPT_VALORES = ("", "m2")
VALUE_EMBEDS = os.environ.get("QF2_VE", "1") != "0"
VE_PASO = int(os.environ.get("QF2_VE_PASO", "8"))
MLP_EXP = int(os.environ.get("QF2_MLP_EXP", "4"))
NGRAM_FILAS = int(os.environ.get("QF2_NGRAM_FILAS", "0"))
MEM_FILAS = int(os.environ.get("QF2_MEM_FILAS", "0"))
MEMHOST_VIA_SITIO11 = os.environ.get("QF2_MEMHOST_VIA_SITIO11", "").strip()
MEMHOST = qf2_memhost_cfg.desde_banderas(
    os.environ.get("QF2_MEMHOST", "1"), os.environ.get("QF2_MEMHOST_ANCHO", "128"),
    os.environ.get("QF2_MEMHOST_CABEZAS", "2x2:64000000,3x4:64000000"), os.environ.get("QF2_MEMHOST_SITIOS", "in,1,11"),
    os.environ.get("QF2_MEMHOST_SIGNOS", "1"), os.environ.get("QF2_MEMHOST_CANON", "1"),
    os.environ.get("QF2_MEMHOST_PUERTA", "sig"),
    qf2_memhost_cfg.via_sitio11(os.environ.get("QF2_MEMHOST_VIA", "res"), MEMHOST_VIA_SITIO11))
if MEMHOST_VIA_SITIO11 and not MEMHOST:
    raise RuntimeError("QF2_MEMHOST_VIA_SITIO11 necesita QF2_MEMHOST=1")
MEMHOST_RETRASO = int(os.environ.get("QF2_MEMHOST_RETRASO", "1"))
MEMHOST_MASCARA = int(os.environ.get("QF2_MEMHOST_MASCARA", "0"))
qf2_hilos_nice.NICE = int(os.environ.get("QF2_HILOS_NICE", "0"))
qf2_memhost_hilo.traza.TRAZA = int(os.environ.get("QF2_MEMHOST_TRAZA", "0"))
qf2_hilos_aten.ATEN = int(os.environ.get("QF2_HILOS_ATEN", "1"))
qf2_hilos_aten.HILO = int(os.environ.get("QF2_HILOS_HILO", "0"))
MEMHOST_X0_ESCALA = float(os.environ.get("QF2_MEMHOST_X0_ESCALA", "1"))
GIL_MS = float(os.environ.get("QF2_GIL_MS", "0"))
if GIL_MS > 0:
    import sys
    sys.setswitchinterval(GIL_MS / 1e3)
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_gil: switchinterval={sys.getswitchinterval()}", flush=True)
MEMHOST_OPT = os.environ.get("QF2_MEMHOST_OPT", "rms")
MEMHOST_LR = float(os.environ.get("QF2_MEMHOST_LR", "0"))
MEMHOST_RUTA = os.environ.get("QF2_MEMHOST_RUTA", "/dev/shm/memhost")
MEMHOST_VAL_ALEATORIO = os.environ.get("QF2_MEMHOST_VAL", "1") == "1"
MEMHOST_RESERVA_S = float(os.environ.get("QF2_MEMHOST_RESERVA_S", "250"))
PESO_CTX = qf2_peso_contexto.desde_entorno()
MEMCORPUS = os.environ.get("QF2_MEMCORPUS", "").strip()
MEMCORPUS_RUTA = os.environ.get("QF2_MEMCORPUS_RUTA", "/dev/shm/memcorpus")
MEMCORPUS_PROCESOS = int(os.environ.get("QF2_MEMCORPUS_PROCESOS", "96"))
MEMCORPUS_EXCLUIR = int(os.environ.get("QF2_MEMCORPUS_EXCLUIR", "256"))
MEMCORPUS_RESERVA_S = float(os.environ.get("QF2_MEMCORPUS_RESERVA_S", "300"))
MEMCORPUS_ANTICIPADO_S = float(os.environ.get("QF2_MEMCORPUS_ANTICIPADO_S", "0"))
MEMCORPUS_TRONCO = float(os.environ.get("QF2_MEMCORPUS_TRONCO", "0"))
MEMHOST_DECAY = os.environ.get("QF2_MEMHOST_DECAY", "0")
MEM_ANCHO = int(os.environ.get("QF2_MEM_ANCHO", "0"))
MEM_TRANSPORTE = os.environ.get("QF2_MEM_TRANSPORTE", "b")
PKM_FILAS = int(os.environ.get("QF2_PKM_FILAS", "0"))
PKM_K = int(os.environ.get("QF2_PKM_K", "4"))
PKM_VALOR = int(os.environ.get("QF2_PKM_VALOR", "256"))
PKM_DK = int(os.environ.get("QF2_PKM_DK", "128"))
PKM_LR = float(os.environ.get("QF2_PKM_LR", "").strip() or EMBEDDING_LR)
RESUMEN = os.environ.get("QF2_RESUMEN", "6:12").strip()
DIFF_CAPAS = os.environ.get("QF2_DIFF_CAPAS", "")
DIFF_SIGNO = int(os.environ.get("QF2_DIFF_SIGNO", "-1"))
MOS = int(os.environ.get("QF2_MOS", "0"))
MEM_EVAL_ANFITRION_BYTES = int(float(os.environ.get("QF2_MEM_EVAL_ANFITRION_BYTES", "8e9")))
ROPE_MM = os.environ.get("QF2_ROPE_MM", "1") != "0"
ROPE_ROLES = os.environ.get("QF2_ROPE_ROLES", "").strip()
QKNORM_MM = os.environ.get("QF2_QKNORM_MM", "1") != "0"
QK_ESCALA = float(os.environ.get("QF2_QK_ESCALA", "1.2"))
FP8_MLP = os.environ.get("QF2_FP8_MLP", "1") != "0"
os.environ["QF2_FP8_DYSCALE"] = os.environ.get("QF2_FP8_DYSCALE", "18")
os.environ["QF2_FP8_RECORTE_GRAD"] = os.environ.get("QF2_FP8_RECORTE_GRAD", "1")
os.environ["QF2_FP8_BWDK"] = os.environ.get("QF2_FP8_BWDK", "0")
if FP8_MLP:
    from kernels.fp8_mlp_op import RESUMEN as FP8_RESUMEN, mlp_fp8
    from kernels.fp8_mlp_comun import _H_CLAMP, _SCALE_H
EVAL_TOPE = os.environ.get("QF2_EVAL_TOPE", "t61").strip()
EMB_SCATTER =os.environ.get("QF2_EMB_SCATTER", "1") != "0"
EMB_SCATTER_GATE = os.environ.get("QF2_EMB_SCATTER_GATE", "0") != "0"
EMB_DETACH = os.environ.get("QF2_EMB_DETACH", "0") != "0"
EMB_RUTA = EMB_SCATTER or EMB_DETACH
if EMB_RUTA:
    from kernels.emb_op import emb_descripcion, emb_preparar, emb_tabla
if EMB_SCATTER_GATE:
    from kernels.emb_puerta import emb_puerta
CEFA = os.environ.get("QF2_CEFA", "0") != "0"
if CEFA:
    from cefa_perdida import cefa_descripcion, cefa_perdida
CE_LSE = os.environ.get("QF2_CE_LSE", "0") != "0"
CE_TROZOS = int(os.environ.get("QF2_CE_TROZOS", "1024"))
if CE_TROZOS:
    from ce_trozos import ce_trozos, ce_trozos_tokens
PROMEDIO_MODO = os.environ.get("QF2_PROMEDIO", "0").strip()
PROMEDIO = PROMEDIO_MODO not in ("", "0")
if PROMEDIO:
    from qf2_promedio import Promedio
GUARDA = os.environ.get("QF2_GUARDA", "1") != "0"
if GUARDA:
    from qf2_guarda import Guarda
REGISTRO = os.environ.get("QF2_REGISTRO", "0") != "0"
PREFIJO_BOS = os.environ.get("QF2_PREFIJO_BOS", "0") != "0"
COM_ESCALARES = os.environ.get("QF2_COM_ESCALARES", "1") != "0"
COM_MUON = os.environ.get("QF2_COM_MUON", "1") != "0"
COM_GATHER_BF16 = os.environ.get("QF2_COM_GATHER_BF16", "1") != "0"
COM_RS_BF16 = os.environ.get("QF2_COM_RS_BF16", "0") != "0"
PLANO_TRANSITORIO = os.environ.get("QF2_PLANO_TRANSITORIO", "0") != "0"
COM_FRANJAS = int(os.environ.get("QF2_COM_FRANJAS", "8").strip() or 0)
OPT_COMPILADO = os.environ.get("QF2_OPT_COMPILADO", "1") != "0"
if OPT_COMPILADO:
    import qf2_opt_compilado
if COM_ESCALARES or COM_MUON or COM_GATHER_BF16 or COM_RS_BF16:
    import qf2_reparto
if COM_FRANJAS:
    import qf2_franjas
if COM_GATHER_BF16:
    CHECKPOINT_RESERVE_SECONDS += 25.0
COLA_BANDERAS = (
    os.environ.get("QF2_SEQ_MAX", "").strip(),
    os.environ.get("QF2_CARGADOR_LARGO", "1").strip(),
    os.environ.get("QF2_COLA_FRAC", "").strip(),
    os.environ.get("QF2_CALIENTA_LARGO", "0").strip(),
)
COLA4096 = any(v not in ("", "0") for v in COLA_BANDERAS)
COLA_LARGA = any(v not in ("", "0") for v in (COLA_BANDERAS[0], COLA_BANDERAS[2], COLA_BANDERAS[3]))
if COLA4096:
    import qf2_cola
    qf2_cola.configurar(*COLA_BANDERAS)
FLUJO_FRAC = os.environ.get("QF2_FLUJO_FRAC", "").strip()
if FLUJO_FRAC not in ("", "0"):
    import qf2_flujo
    qf2_flujo.configurar(FLUJO_FRAC, COLA4096)
FLUJO = FLUJO_FRAC not in ("", "0") and qf2_flujo.activo()
PUNTOS_S = os.environ.get("QF2_PUNTOS_S", "").strip()
REANUDA = os.environ.get("QF2_REANUDA", "0") != "0"
PUNTOS = PUNTOS_S not in ("", "0") or REANUDA
if PUNTOS:
    import qf2_flujo
    import qf2_puntos
PRECARGA = os.environ.get("QF2_PRECARGA", "1") != "0" and not PUNTOS and not FLUJO
if PRECARGA:
    import qf2_precarga
RECOLOCADO = float(os.environ.get("QF2_RECOLOCADO", "0.08").strip() or 0)
RECOLOCADO_CE = os.environ.get("QF2_RECOLOCADO_CE", "continuacion").strip()
RECOLOCADO_RAFAGA = os.environ.get("QF2_RECOLOCADO_RAFAGA", "3:6").strip()
if RECOLOCADO > 0 and not PRECARGA:
    raise RuntimeError("QF2_RECOLOCADO necesita QF2_PRECARGA=1")
if PRECARGA:
    import qf2_recolocado
RECOLOCADO_FORMATO = os.environ.get("QF2_RECOLOCADO_FORMATO", "ejemplos").strip()
PARES_TRAMO = os.environ.get("QF2_PARES_TRAMO", "8:40").strip()
PARES_ETIQUETAS = os.environ.get("QF2_PARES_ETIQUETAS", "Input/Output").strip()
INICIO_DESDE = os.environ.get("QF2_INICIO_DESDE", "").strip()
BANCO_LRM = os.environ.get("QF2_BANCO_LRM", "").strip()
if INICIO_DESDE and not PRECARGA:
    raise RuntimeError("QF2_INICIO_DESDE necesita QF2_PRECARGA=1")
import qf2_inicio
BANCO_LRM_TRAMO = qf2_inicio.leer_lrm(BANCO_LRM)
qf2_inicio.SHARDS_DESDE = os.environ.get("QF2_BANCO_SHARDS_DESDE", "").strip()
import qf2_agrupa
AGRUPA = qf2_agrupa.leer(os.environ.get("QF2_AGRUPA", "").strip())
if AGRUPA is not None and not PRECARGA:
    raise RuntimeError("QF2_AGRUPA necesita QF2_PRECARGA=1")
import qf2_filas
qf2_filas.validar(AGRUPA, RECOLOCADO)
import qf2_objetivos
DIFICILES = qf2_objetivos.leer_objetivo(os.environ.get("QF2_CE_DIFICILES", "").strip(), "")
TRAMO = qf2_objetivos.leer_objetivo("", os.environ.get("QF2_CE_TRAMO", "").strip())
OBJETIVO = TRAMO if DIFICILES is None else DIFICILES
RUIDO = qf2_objetivos.Ruido(os.environ.get("QF2_RUIDO_ENTRADA", "").strip(), SEED)
if OBJETIVO is not None and (not CE_TROZOS or CEFA or MOS):
    raise RuntimeError("QF2_CE_DIFICILES y QF2_CE_TRAMO van en la CE por trozos: necesitan QF2_CE_TROZOS sin QF2_CEFA ni QF2_MOS")
if RUIDO.activo and PESO_CTX.activo:
    raise RuntimeError("QF2_RUIDO_ENTRADA no llega a las pasadas propias de QF2_PESO_CTX")
import qf2_brier
BRIER = qf2_brier.leer(os.environ.get("QF2_BRIER", "").strip())
if BRIER is not None and (not CE_TROZOS or CEFA or MOS or PESO_CTX.activo):
    raise RuntimeError("QF2_BRIER va en la CE por trozos: necesita QF2_CE_TROZOS, sin QF2_CEFA, QF2_MOS ni QF2_PESO_CTX")
import qf2_equilibrio
EQUILIBRIO = qf2_equilibrio.leer(os.environ.get("QF2_EQUILIBRIO", "").strip())
EQUILIBRIO = qf2_filas.equilibrio(EQUILIBRIO, AGRUPA)
if EQUILIBRIO is not None and (not CE_TROZOS or CEFA or MOS or REGISTRO or MEMCORPUS or PESO_CTX.activo):
    raise RuntimeError("QF2_EQUILIBRIO va solo en la CE por trozos: sin QF2_CEFA, QF2_MOS, QF2_REGISTRO, QF2_MEMCORPUS ni QF2_PESO_CTX")
import qf2_union
UNION = qf2_union.leer(DIFICILES, TRAMO, EQUILIBRIO, BRIER)
VE_GATE_CHANNELS = 12
SUELO_B_REF = 2**19
SUELO_UNEMBEDDING_LR = 0.008
SUELO_MATRIX_LR = 0.02
SUELO_RESID_LR = 0.005
SUELO_X0_LR = 0.5
SUELO_MOMENTUM_WARMUP_FRAC = 0.0625
SUELO_MOMENTUM_START = 0.85
SUELO_MOMENTUM_PEAK = 0.97
SUELO_MOMENTUM_END = 0.90


# -----------------------------------------------------------------------------
# Device / distributed setup

def neuron_available() -> bool:
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        return "neuron"
    return "cpu"


def init_runtime(device_type: str):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(SEED)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    elif ddp and device_type == "neuron":
        dist.init_process_group("neuron")
        device = torch.device("neuron")
    elif ddp:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return ddp, rank, local_rank, world_size, device


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def forward_mem(model, mem_obj, x, y, mem=None, corpus=None, reduccion="mean"):
    extra = {"corpus": corpus} if corpus is not None else {}
    if reduccion != "mean":
        extra["loss_reduction"] = reduccion
    if RUIDO.activo and model.training:
        extra["ruido"] = RUIDO.muestra(x.size(0), model.config.n_embd, x.device, COMPUTE_DTYPE)
    if mem is not None:
        return model(x, y, mem=mem, **extra)
    if mem_obj is None:
        return model(x, y, **extra)
    return model(x, y, mem=mem_obj.consultar(x, COMPUTE_DTYPE), **extra)


def perdida_micro(model, mem_obj, x, y, mem, corpus, paso, micro):
    if not PESO_CTX.activo:
        return forward_mem(model, mem_obj, x, y, mem, corpus)
    nll = forward_mem(model, mem_obj, x, y, mem, corpus, "none")
    return qf2_peso_contexto.ponderar(PESO_CTX, lambda a, b: model(a, b, loss_reduction="none"), x, y, nll, paso, micro)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "neuron":
        return torch.bfloat16
    return torch.float32


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    value_embeds: bool = False
    ve_paso: int = 2
    registro: bool = False
    prefijo_bos: bool = False
    ngramas: int = 0
    memoria: int = 0
    memoria_ancho: int = 0
    memhost: str = ""
    memcorpus: str = ""
    pkm: int = 0
    pkm_k: int = 4
    pkm_valor: int = 256
    pkm_dk: int = 128
    resumen: str = ""
    diff_capas: str = ""
    diff_signo: int = -1
    mos: int = 0
    mlp_exp: int = 4
    rope_roles: str = ""


def has_ve(layer_idx: int, n_layer: int, paso: int = 2) -> bool:
    return (n_layer - 1 - layer_idx) % paso == 0


def config_dict(config: GPTConfig) -> dict:
    return {k: v for k, v in asdict(config).items()
            if (config.pkm or not k.startswith("pkm")) and (config.diff_capas or not k.startswith("diff")) and (config.mos or k != "mos") and (k not in ("value_embeds", "registro", "prefijo_bos", "ngramas", "memoria", "memoria_ancho", "memhost", "memcorpus", "resumen") or v) and (k != "ve_paso" or (config.value_embeds and v != 2)) and (k != "mlp_exp" or v != 4) and (k != "rope_roles" or v)}


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


def tabla(mod: nn.Embedding, idx: torch.Tensor) -> torch.Tensor:
    return emb_tabla(idx, mod.weight, COMPUTE_DTYPE, EMB_DETACH) if EMB_RUTA else mod(idx)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


_BOS_ID: list[int] = []


def bos_token_id() -> int:
    if not _BOS_ID:
        _BOS_ID.append(ensure_tokenizer(build_if_missing=False).get_bos_token_id())
    return _BOS_ID[0]


def anteponer(primero: torch.Tensor, v: torch.Tensor, recorta: bool) -> torch.Tensor:
    cabeza = primero.to(v.dtype).view(1, 1, -1).expand(v.size(0), 1, v.size(-1))
    return torch.cat([cabeza, v[:, :-1] if recorta else v], dim=1)


def prefijo_bos(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bos = idx.new_full((idx.size(0), 1), bos_token_id())
    sin_bos = idx[:, :1] != bos
    return torch.where(sin_bos, torch.cat([bos, idx], dim=1), torch.cat([idx, bos], dim=1)), sin_bos


# -----------------------------------------------------------------------------
# Example: a custom NKI kernel (optional, default OFF)
#
# This is an ILLUSTRATION to get you started writing your own Trainium kernels.
# It implements the MLP activation `relu(x)**2` as a fused NKI kernel. The eager
# baseline does not use it; enable it with `--nki-relu2` (Neuron only).
#
# Three things worth copying from this example:
#   1. The kernel body: load a tile from HBM into SBUF, compute, store back.
#   2. Tiling over the 128-partition limit: a NeuronCore processes at most 128
#      rows (the "partition" dim) at once, so we loop over chunks of 128 rows.
#   3. Autograd: a raw NKI kernel is NOT differentiable. To use one inside the
#      training forward pass you must wrap it in a torch.autograd.Function and
#      provide the backward yourself, or gradients silently stop flowing.
#
# To write your own: change the per-tile math in `_relu_sq_fwd` (and the matching
# derivative in `_relu_sq_bwd`), or fuse a different elementwise op.

_NKI_PARTITION = 128  # NeuronCore SBUF partition limit (== nki.language.tile_size.pmax)

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _relu_sq_fwd(x):
        """Fused y = relu(x)**2 for a 2-D [M, N] tensor, tiled over rows."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)  # relu
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)     # square
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    @nki.jit
    def _relu_sq_bwd(x, grad_out):
        """Backward of relu(x)**2: grad_in = grad_out * 2 * relu(x)."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            gb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=grad_out[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)   # relu(x)
            two_r = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=two_r, data=rb, op0=nl.multiply, operand0=2.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=two_r, data2=gb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    class _ReluSquaredNKI(torch.autograd.Function):
        """Makes the NKI relu**2 kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x):
            y = _relu_sq_fwd(x.reshape(-1, x.shape[-1])).reshape(x.shape)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)

    _HAS_NKI = True
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) — eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the example NKI kernel when enabled, else eager."""
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def mascara_secuencias(idx: torch.Tensor, por_ciento: int) -> torch.Tensor:
    viva = torch.remainder(idx[:, 1:9].float().sum(dim=1), 100.0) >= por_ciento
    return viva.view(-1, 1, 1) | (torch.arange(idx.size(1), device=idx.device) < 8).view(1, -1, 1)


def qknorm(x: torch.Tensor, media: torch.Tensor | None = None) -> torch.Tensor:
    if media is None:
        return norm(x) * QK_ESCALA if QK_ESCALA != 1.0 else norm(x)
    return x * torch.rsqrt(torch.matmul(x * x, media.to(x.dtype)) + torch.finfo(torch.float32).eps)


def rot_matrix(d: int) -> torch.Tensor:
    h = d // 2
    z = torch.zeros(h, h, dtype=torch.float32)
    i = torch.eye(h, dtype=torch.float32)
    return torch.cat((torch.cat((z, -i), dim=1), torch.cat((i, z), dim=1)), dim=0)


def rope_roles_mask(roles: str, n_head: int, pairs: int) -> torch.Tensor | None:
    if not roles:
        return None
    tipo, sep, arg = roles.partition(":")
    if tipo not in ("nope", "pares") or not sep or not arg.isascii() or not arg.isdigit():
        raise ValueError(f"rope_roles {roles!r}: se esperaba '', 'nope:N' o 'pares:K'")
    n = int(arg)
    mask = torch.zeros(n_head, pairs, dtype=torch.bool)
    if tipo == "nope":
        if n > n_head:
            raise ValueError(f"rope_roles {roles!r}: N={n} supera n_head={n_head}")
        mask[:n] = True
        return mask
    if n < 1 or pairs % n:
        raise ValueError(f"rope_roles {roles!r}: K={n} debe dividir {pairs} pares")
    for h in range(n_head):
        s = (h % (pairs // n)) * n
        mask[h, s:s + n] = True
    return mask


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor | None = None) -> torch.Tensor:
    if rot is not None:
        return x * cos + torch.matmul(x, rot.to(x.dtype)) * sin
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        if ROPE_MM:
            self.register_buffer("rot", rot_matrix(self.head_dim).to(COMPUTE_DTYPE), persistent=False)
        else:
            self.rot = None
        if QKNORM_MM:
            self.register_buffer("media", torch.full((self.head_dim, self.head_dim), 1.0 / (self.head_dim * QK_ESCALA ** 2)).to(COMPUTE_DTYPE), persistent=False)
        else:
            self.media = None
        if config.value_embeds and has_ve(layer_idx, config.n_layer, config.ve_paso):
            with torch.random.fork_rng(devices=[]):
                self.ve_gate = Linear(VE_GATE_CHANNELS, config.n_head, bias=False)
        else:
            self.ve_gate = None
        self.diff = qf2_diff.RamaDiff(config.n_embd, config.n_head, config.diff_signo) if layer_idx in qf2_diff.indices(config.diff_capas, config.n_layer) else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None = None, mem_v: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve.view(b, t, self.n_head, self.head_dim)
        if mem_v is not None:
            v = v + mem_v.view(b, t, self.n_head, self.head_dim)
        q = qknorm(apply_rotary_emb(q, cos, sin, self.rot), self.media)
        k = qknorm(apply_rotary_emb(k, cos, sin, self.rot), self.media)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.diff is not None:
            q2 = qknorm(apply_rotary_emb(self.diff.c_q2(x).view(b, t, self.n_head, self.head_dim), cos, sin, self.rot), self.media).transpose(1, 2)
            y = self.diff.combinar(y, F.scaled_dot_product_attention(q2, k, v, is_causal=True), x)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


def oculta_mlp(config: GPTConfig) -> int:
    return config.mlp_exp if config.mlp_exp > 16 else config.mlp_exp * config.n_embd


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, oculta_mlp(config), bias=False)
        self.c_proj = Linear(oculta_mlp(config), config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FP8_MLP and self.training and torch.is_grad_enabled():
            return mlp_fp8(x, self.c_fc.weight, self.c_proj.weight)
        if FP8_MLP and EVAL_TOPE == "t61":
            return self.c_proj(torch.clamp(F.relu(self.c_fc(x)), max=_H_CLAMP / _SCALE_H).square())
        return self.c_proj(relu_squared(self.c_fc(x)))


def mix_residual(
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_lambda: torch.Tensor,
    x0_lambda: torch.Tensor,
) -> torch.Tensor:
    """Mix residual streams without promoting BF16 activations to FP32."""
    return resid_lambda.to(dtype=x.dtype) * x + x0_lambda.to(dtype=x.dtype) * x0


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None = None, mem_v: torch.Tensor | None = None) -> torch.Tensor:
        x = mix_residual(x, x0, self.resid_lambda, self.x0_lambda)
        x = x + self.attn(norm(x), cos, sin, ve, mem_v)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, repartida: bool = False, memoria_dtype: torch.dtype = torch.float32, memoria_device=None):
        super().__init__()
        self.config = config
        self.repartida = repartida
        self.memoria_anfitrion = False
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        with torch.random.fork_rng(devices=[]):
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded_vocab_size, config.n_embd)
                for i in range(config.n_layer) if config.value_embeds and has_ve(i, config.n_layer, config.ve_paso)
            })
        if config.ngramas and (config.registro or config.prefijo_bos):
            raise ValueError("ngramas es incompatible con registro y prefijo_bos")
        self.ngramas = qf2_ngramas.tablas(config.ngramas, config.n_embd) if config.ngramas else nn.ModuleDict()
        if config.ngramas:
            self.ngrama_escala = qf2_ngramas.escala()
            self.ngrama_bos = bos_token_id()
        self.memoria = nn.ModuleDict()
        if config.memoria:
            if config.ngramas or config.registro or config.prefijo_bos:
                raise ValueError("memoria es incompatible con ngramas, registro y prefijo_bos")
            if not repartida:
                self.memoria = qf2_memoria.tablas(config.memoria, config.memoria_ancho, memoria_dtype, memoria_device)
            with torch.random.fork_rng(devices=[]):
                self.memoria_proy = Linear(2 * config.memoria_ancho, config.n_embd, bias=False)
            self.memoria_escala = qf2_ngramas.escala()
            self.memoria_bos = bos_token_id()
        self.memhost_cfg = None
        if config.memhost:
            if config.ngramas or config.memoria or config.registro or config.prefijo_bos:
                raise ValueError("memhost es incompatible con ngramas, memoria, registro y prefijo_bos")
            self.memhost_cfg = qf2_memhost_cfg.desde_texto(config.memhost)
            self.memhost_bos = bos_token_id()
            self.memhost_sitios = qf2_memhost_sitios.Sitios(self.memhost_cfg, config.n_embd, Linear)
            self.memhost = qf2_memhost_sitios.Tablas(self.memhost_cfg, config.vocab_size)
        self.memcorpus_cfg = None
        if config.memcorpus:
            if config.mos or config.registro or config.prefijo_bos:
                raise ValueError("memcorpus es incompatible con mos, registro y prefijo_bos")
            self.memcorpus_cfg = qf2_corpus_mezcla.desde_texto(config.memcorpus)
            self.memcorpus_bos = bos_token_id()
            self.memcorpus_puerta = qf2_corpus_mezcla.Puerta(self.memcorpus_cfg, config.n_embd)
        if config.registro:
            self.registro = nn.Parameter(torch.zeros(config.n_embd))
        self.pkm = qf2_pkm.PKM(config.pkm, config.n_embd, config.pkm_k, config.pkm_valor, config.pkm_dk) if config.pkm else None
        if config.resumen and config.registro:
            raise ValueError("resumen es incompatible con registro")
        self.resumen = qf2_resumen.Resumen(config.resumen, config.n_layer, config.n_embd, bos_token_id(), Linear) if config.resumen else None
        self.mos = qf2_mos.Mezcla(config.n_embd, config.mos) if config.mos else None
        cos, sin = self._precompute_rotary_embeddings(self.rotary_len(), config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, memoria: bool = True) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        depth_span = max(self.config.n_layer - 1, 1)
        for i, block in enumerate(self.transformer.h):
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if SUELO_INIT:
                block.resid_lambda.fill_(1.15 - 0.10 * i / depth_span)
                block.x0_lambda.fill_(0.20 - 0.15 * i / depth_span)
            else:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        if self.config.ngramas:
            qf2_ngramas.iniciar(self.ngramas, self.config.n_embd)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_len(), self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        if self.config.registro:
            self.registro.copy_(self.transformer.wte.weight[bos_token_id()])
        if self.config.memhost:
            self.memhost_sitios.iniciar(MEMHOST_VAL_ALEATORIO)
        if self.config.memcorpus:
            self.memcorpus_puerta.w.fill_(qf2_corpus_mezcla.LAM_INICIAL)
            if self.memcorpus_puerta.v is not None:
                self.memcorpus_puerta.v.zero_()
        if self.config.memoria:
            torch.nn.init.uniform_(self.memoria_proy.weight, -s, s)
            if memoria and not self.repartida:
                qf2_memoria.iniciar(self.memoria, SEED)
        if self.pkm is not None:
            self.pkm.iniciar()
        if self.resumen is not None:
            self.resumen.iniciar()
        if self.mos is not None:
            self.mos.iniciar()
        for block in self.transformer.h:
            if block.attn.diff is not None:
                block.attn.diff.iniciar()

    def rotary_len(self) -> int:
        return self.config.sequence_len + int(self.config.registro) + int(self.config.prefijo_bos)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 100000):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        mask = rope_roles_mask(self.config.rope_roles, self.config.n_head, head_dim // 2)
        if mask is not None:
            mask = mask.to(device)[None, None]
            cos = torch.where(mask, torch.ones_like(cos), cos).contiguous()
            sin = torch.where(mask, torch.zeros_like(sin), sin).contiguous()
        if ROPE_MM:
            return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer  # per-block scalars
        non_matmul += sum(p.numel() for p in self.value_embeds.parameters())
        non_matmul += sum(p.numel() for p in self.ngramas.parameters())
        non_matmul += sum(p.numel() for p in self.memoria.parameters())
        if self.pkm is not None:
            non_matmul += self.pkm.valores.weight.numel() + self.pkm.escala.numel()
        non_matmul += self.registro.numel() if self.config.registro else 0
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T: 2*h*d*T per token, AV: 2*h*d*T per token, x3 for fwd+bwd
        attn_flops = (self.config.n_layer + len(qf2_diff.indices(self.config.diff_capas, self.config.n_layer))) * 12 * h * d * t
        mos_flops = 6 * self.lm_head.weight.numel() if self.mos is not None else 0
        return 6 * (nparams - non_matmul) + attn_flops + mos_flops

    def num_scaling_params(self) -> dict[str, int]:
        """Parameter count used for nanochat-style data:parameter scaling."""
        transformer_matrices = sum(
            p.numel()
            for block in self.transformer.h
            for name, p in block.named_parameters()
            if p.ndim == 2 and "ve_gate" not in name and "diff." not in name
        )
        lm_head = self.lm_head.weight.numel()
        return {
            "transformer_matrices": transformer_matrices,
            "lm_head": lm_head,
            "total": transformer_matrices + lm_head,
        }

    def memoria_lr(self, total_batch_size: int) -> float:
        batch_lr_scale = (total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0
        return EMBEDDING_LR * (self.config.n_embd / 768) ** -0.5 * 0.5 * batch_lr_scale

    def setup_optimizer(self, total_batch_size: int = TOTAL_BATCH_SIZE):
        scalar_params = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        if self.config.memoria:
            matrix_params.append(self.memoria_proy.weight)
        if self.config.memhost:
            matrix_params.extend(self.memhost_sitios.matrices())
        if self.pkm is not None:
            matrix_params.extend([self.pkm.consulta, self.pkm.salida])
        if self.resumen is not None:
            matrix_params.append(self.resumen.proy.weight)
        if self.mos is not None:
            matrix_params.append(self.mos.w2)
        embedding_params = list(self.transformer.wte.parameters()) + ([self.registro] if self.config.registro else [])
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        batch_lr_scale = (total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0
        unembedding_lr = SUELO_UNEMBEDDING_LR if SUELO_LR else UNEMBEDDING_LR
        matrix_lr = SUELO_MATRIX_LR if SUELO_LR else MATRIX_LR
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        ]
        if SUELO_ESC:
            param_groups.append(dict(kind="adamw", params=[b.resid_lambda for b in self.transformer.h], lr=SUELO_RESID_LR * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05))
            param_groups.append(dict(kind="adamw", params=[b.x0_lambda for b in self.transformer.h], lr=SUELO_X0_LR * batch_lr_scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        else:
            param_groups.append(dict(kind="adamw", params=scalar_params, lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        value_embeds_params = list(self.value_embeds.parameters())
        if value_embeds_params:
            param_groups.append(dict(kind="adamw", params=value_embeds_params, lr=EMBEDDING_LR * dmodel_lr_scale * 0.5 * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
        ngramas_params = list(self.ngramas.parameters())
        if ngramas_params:
            param_groups.append(dict(kind="adamw", params=ngramas_params, lr=EMBEDDING_LR * dmodel_lr_scale * 0.5 * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind="adamw", params=[self.ngrama_escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        memoria_params = list(self.memoria.parameters())
        if memoria_params:
            param_groups.append(dict(kind="adamw", params=memoria_params, lr=self.memoria_lr(total_batch_size), betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        if self.pkm is not None:
            param_groups.append(dict(kind="adamw", params=[self.pkm.valores.weight, self.pkm.subclaves], lr=PKM_LR * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind="adamw", params=[self.pkm.escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.mos is not None:
            param_groups.append(dict(kind="adamw", params=[self.mos.pi_w, self.mos.pi_b], lr=qf2_mos.pi_lr(self.config.n_embd) * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memhost and self.memhost_sitios.escalares():
            param_groups.append(dict(kind="adamw", params=self.memhost_sitios.escalares(), lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memoria:
            param_groups.append(dict(kind="adamw", params=[self.memoria_escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memcorpus:
            param_groups.append(dict(kind="adamw", params=[self.memcorpus_puerta.w], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
            if self.memcorpus_puerta.v is not None:
                param_groups.append(dict(kind="adamw", params=[self.memcorpus_puerta.v], lr=unembedding_lr * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=matrix_lr * batch_lr_scale, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=wd_muon(0.0)))
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        opt = DistMuonAdamW(param_groups) if distributed else MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def consulta_memoria(self, idx: torch.Tensor):
        if self.memoria_anfitrion:
            return qf2_memoria_anfitrion.consultar(self.memoria, idx, self.config.memoria, self.memoria_bos)
        return qf2_memoria.consulta_replicada(self.memoria, idx, self.config.memoria, self.memoria_bos)

    def memoria_externa(self, idx: torch.Tensor):
        if self.config.memhost:
            return qf2_memhost_ckpt.consultar(self, idx, COMPUTE_DTYPE)
        return self.consulta_memoria(idx) if self.memoria_anfitrion else None

    def salida_memcorpus(self, x, idx, targets, loss_reduction, corpus):
        if CEFA or CE_TROZOS or CE_LSE:
            raise ValueError("QF2_MEMCORPUS calcula su propia pérdida: incompatible con QF2_CEFA, QF2_CE_TROZOS y QF2_CE_LSE")
        cand, celda = corpus if corpus is not None else qf2_corpus_mezcla.consultar_modelo(self, idx)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        s = self.memcorpus_puerta.desplazamiento(x)
        if targets is None:
            return qf2_corpus_mezcla.log_probs(self.memcorpus_puerta, logits, cand, celda, s)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction="none")
        return qf2_corpus_mezcla.reducir(self.memcorpus_puerta, nll, targets, cand, celda, loss_reduction, s,
                                         MEMCORPUS_TRONCO if self.training else 0.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", mem=None, corpus=None, ruido=None):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        registro = self.config.registro
        entrena = targets is not None
        prefijo = self.config.prefijo_bos and not entrena
        if prefijo:
            idx, sin_bos = prefijo_bos(idx)
            t = idx.size(1)
        n = t + int(registro and not entrena)
        cos, sin = self.cos[:, :n], self.sin[:, :n]
        x = tabla(self.transformer.wte, idx).to(COMPUTE_DTYPE)
        if registro:
            x = anteponer(self.registro, x, entrena)
            if entrena:
                targets = torch.cat([targets.new_full((targets.size(0), 1), -1), targets[:, :-1]], dim=1)
        x = norm(x)
        if ruido is not None and self.training:
            x = x + ruido
        if self.config.ngramas:
            x = qf2_ngramas.sumar(self.ngramas, self.ngrama_escala, idx, self.config.ngramas, self.ngrama_bos, tabla, x)
        if self.config.memoria:
            if mem is None:
                if self.repartida:
                    raise ValueError("el modelo repartido necesita mem")
                mem = self.consulta_memoria(idx)
            x = qf2_memoria.combinar(mem[0], mem[1], self.memoria_proy, self.memoria_escala, x)
        if self.config.memhost:
            if mem is None:
                if not hasattr(self.memhost, "q"):
                    raise ValueError("el modelo memhost necesita mem")
                mem = qf2_memhost_ckpt.consultar(self, idx, COMPUTE_DTYPE)
            if MEMHOST_MASCARA and self.training:
                mem = mem * mascara_secuencias(idx, MEMHOST_MASCARA).to(mem.dtype)
            x_sin = x
            if self.memhost_cfg.entrada:
                x = self.memhost_sitios.entrada(mem, x)
        x0 = x
        if self.config.memhost and MEMHOST_X0_ESCALA != 1:
            x0 = x_sin + MEMHOST_X0_ESCALA * (x - x_sin)
        resumen_doc = None
        for i, block in enumerate(self.transformer.h):
            if self.resumen is not None and i == self.resumen.capa_salida:
                x = x + self.resumen.inyectar(resumen_doc)
            ve = tabla(self.value_embeds[str(i)], idx).to(x.dtype) if str(i) in self.value_embeds else None
            if ve is not None and registro:
                ve = anteponer(ve.new_zeros(ve.size(-1)), ve, entrena)
            gated = self.config.memhost and i in self.memhost_cfg.gated
            mem_v = self.memhost_sitios.delta(i, mem, x) if gated and self.memhost_cfg.via_de(i) == "v" else None
            x = block(x, x0, cos, sin, ve, mem_v)
            if gated and self.memhost_cfg.via_de(i) == "res":
                x = self.memhost_sitios.compuerta(i, mem, x)
            if self.pkm is not None and i == self.config.n_layer // 2:
                x = x + self.pkm(norm(x), tabla)
            if self.resumen is not None and i == self.resumen.capa_entrada:
                resumen_doc = self.resumen.media(norm(x), idx)
        x = norm(x)
        if registro and not entrena:
            x = x[:, 1:]
        if prefijo:
            x = torch.where(sin_bos.unsqueeze(-1), x[:, 1:], x[:, :-1])
        if self.mos is not None:
            return self.mos(x, self.lm_head, self.config.vocab_size, LOGIT_SOFTCAP, targets, loss_reduction, CEFA and self.training)
        if self.config.memcorpus:
            return self.salida_memcorpus(x, idx, targets, loss_reduction, corpus)
        if CEFA and self.training and targets is not None and loss_reduction == "mean":
            return cefa_perdida(self, x, targets, LOGIT_SOFTCAP)
        if CE_TROZOS and self.training and targets is not None and loss_reduction == "mean":
            if UNION is not None:
                return UNION(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets, idx == bos_token_id(), LOGIT_SOFTCAP,
                             self.config.vocab_size, CE_TROZOS)
            if BRIER is not None:
                return qf2_brier.ce_brier_trozos(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1),
                                                 LOGIT_SOFTCAP, self.config.vocab_size, CE_TROZOS, BRIER.lam)
            if OBJETIVO is not None:
                nll = ce_trozos_tokens(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP,
                                       self.config.vocab_size, CE_TROZOS)
                return OBJETIVO(nll.view_as(targets), targets != -1)
            if EQUILIBRIO is not None:
                nll = ce_trozos_tokens(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP,
                                       self.config.vocab_size, CE_TROZOS)
                return EQUILIBRIO(nll.view_as(targets), targets != -1, idx == bos_token_id())
            return ce_trozos(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP, self.config.vocab_size, CE_TROZOS)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        if CE_LSE and self.training and loss_reduction == "mean":
            z, y = logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            vale = y != -1
            nll = torch.logsumexp(z, dim=-1) - z.gather(1, y.clamp_min(0).unsqueeze(1)).squeeze(1)
            return torch.where(vale, nll, 0.0).sum() / vale.sum()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config(
    depth: int = DEPTH,
    aspect_ratio: int = ASPECT_RATIO,
    head_dim: int = HEAD_DIM,
    sequence_len: int = SEQ_LEN,
) -> GPTConfig:
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        value_embeds=VALUE_EMBEDS,
        ve_paso=VE_PASO,
        rope_roles=ROPE_ROLES,
        registro=REGISTRO,
        prefijo_bos=PREFIJO_BOS,
        ngramas=NGRAM_FILAS,
        memoria=MEM_FILAS,
        memoria_ancho=(MEM_ANCHO or n_embd // 2) if MEM_FILAS else 0,
        memhost=MEMHOST,
        memcorpus=MEMCORPUS,
        pkm=PKM_FILAS,
        pkm_k=PKM_K,
        pkm_valor=PKM_VALOR,
        pkm_dk=PKM_DK,
        resumen=RESUMEN,
        diff_capas=DIFF_CAPAS,
        diff_signo=DIFF_SIGNO,
        mos=MOS,
        mlp_exp=MLP_EXP,
    )


# -----------------------------------------------------------------------------
# Optimizer

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        if OPT not in OPT_VALORES:
            raise ValueError(f"QF2_OPT must be one of {OPT_VALORES}, got {OPT!r}")
        if OPT_COMPILADO and (OPT or type(self) is MuonAdamW):
            raise ValueError("QF2_OPT_COMPILADO=1 needs the distributed optimizer (world > 1) and QF2_OPT unset")
        super().__init__(param_groups, defaults={})
        # Changing Python scalar arguments retraces Neuron eager optimizer ops.
        # Keep the changing schedule values in persistent 0-D CPU tensors so the
        # same device graphs can be reused throughout warmup and training.
        for group in self.param_groups:
            group["_lr_t"] = torch.tensor(float(group["lr"]), dtype=torch.float32)
            group["_wd_t"] = torch.tensor(float(group["weight_decay"]), dtype=torch.float32)
            if group["kind"] == "adamw":
                beta1, beta2 = group["betas"]
                group["_one_minus_beta1_t"] = torch.tensor(1 - beta1, dtype=torch.float32)
                group["_one_minus_beta2_t"] = torch.tensor(1 - beta2, dtype=torch.float32)
                group["_eps_t"] = torch.tensor(float(group["eps"]), dtype=torch.float32)
            elif group["kind"] == "muon":
                group["_momentum_t"] = torch.tensor(float(group["momentum"]), dtype=torch.float32)
                group["_one_minus_momentum_t"] = torch.tensor(1 - float(group["momentum"]), dtype=torch.float32)
                group["_one_minus_beta2_t"] = torch.tensor(1 - float(group["beta2"]), dtype=torch.float32)
                if OPT == "m2":
                    group["_m2_one_minus_beta2_t"] = torch.tensor(1 - OPT_BETA2, dtype=torch.float32)
                    group["_m2_eps_t"] = torch.tensor(OPT_EPS, dtype=torch.float32)

    @staticmethod
    def _m2_denominator(g, state, group):
        if "m2_v" not in state:
            state["m2_v"] = torch.zeros_like(g)
            state["m2_step"] = 0
            state["m2_bias_t"] = torch.tensor(0.0, dtype=torch.float32)
        state["m2_step"] += 1
        state["m2_bias_t"].fill_(1 - OPT_BETA2 ** state["m2_step"])
        state["m2_v"].lerp_(g.square(), group["_m2_one_minus_beta2_t"])
        return (state["m2_v"] / state["m2_bias_t"]).sqrt_().add_(group["_m2_eps_t"])

    @staticmethod
    def _adamw_update(p, grad, state, group):
        beta1, beta2 = group["betas"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        one_minus_beta1_t = group["_one_minus_beta1_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        eps_t = group["_eps_t"]
        lr_t.fill_(group["lr"])
        wd_t.fill_(group["weight_decay"])
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["bias1_t"] = torch.tensor(0.0, dtype=torch.float32)
            state["bias2_t"] = torch.tensor(0.0, dtype=torch.float32)
        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        bias1_t = state["bias1_t"]
        bias2_t = state["bias2_t"]
        bias1_t.fill_(1 - beta1 ** state["step"])
        bias2_t.fill_(1 - beta2 ** state["step"])
        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, one_minus_beta1_t)
        exp_avg_sq.lerp_(grad.square(), one_minus_beta2_t)
        update = exp_avg / bias1_t / ((exp_avg_sq / bias2_t).sqrt() + eps_t)
        p.sub_(lr_t * update)

    @torch.no_grad()
    def _adamw_step(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            self._adamw_update(p, p.grad, self.state[p], group)

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if "stacked_grads" not in state:
            state["stacked_grads"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "stacked_params" not in state:
            state["stacked_params"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)

        g = state["stacked_grads"]
        w = state["stacked_params"]
        for i, p in enumerate(params):
            if p.grad is None:
                g[i].zero_()
            else:
                g[i].copy_(p.grad)
            w[i].copy_(p)
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]

        momentum_t = group["_momentum_t"]
        one_minus_momentum_t = group["_one_minus_momentum_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        momentum_t.fill_(group["momentum"])
        one_minus_momentum_t.fill_(1 - group["momentum"])
        one_minus_beta2_t.fill_(1 - group["beta2"])
        lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd_t.fill_(group["weight_decay"])

        m2_den = self._m2_denominator(g, state, group) if OPT == "m2" else None
        momentum_buffer.lerp_(g, one_minus_momentum_t)
        g = g.lerp_(momentum_buffer, momentum_t)
        if m2_den is not None:
            g = g.div_(m2_den)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
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
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


class DistMuonAdamW(MuonAdamW):
    """ZeRO-2-style distributed variant with replicated FP32 parameters.

    AdamW tensors are sharded along their first dimension. Muon tensors are
    sharded by whole matrices within each equal-shape group. Gradients use
    reduce-scatter, only the owned shard carries optimizer state, and updated
    parameters are all-gathered back to every rank.

    Muon groups execute serially and share one communication slab. This is the
    memory-critical part of the port: allocating one padded slab per matrix
    shape would erase most of the optimizer-state saving on Trainium.
    """

    handles_gradient_sync = True

    def __init__(self, param_groups):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DistMuonAdamW requires an initialized process group")
        if COM_GATHER_BF16 and COMPUTE_DTYPE != torch.bfloat16:
            raise ValueError(f"QF2_COM_GATHER_BF16=1 needs bf16 compute (the forward casts weights to bf16), got {COMPUTE_DTYPE}")
        if COM_RS_BF16 and not COM_MUON:
            raise ValueError("QF2_COM_RS_BF16=1 needs QF2_COM_MUON=1 (the per-shape Muon path reduces in fp32)")
        if PLANO_TRANSITORIO and not COM_MUON:
            raise ValueError("QF2_PLANO_TRANSITORIO=1 needs QF2_COM_MUON=1 (the slab belongs to the batched Muon path)")
        if COM_FRANJAS and (not COM_MUON or COM_RS_BF16):
            raise ValueError("QF2_COM_FRANJAS needs QF2_COM_MUON=1 and QF2_COM_RS_BF16=0 (strips move the fp32 batched Muon slab)")
        super().__init__(param_groups)
        self._compilado = qf2_opt_compilado.Compilado(self.param_groups[0]["params"][0].device) if OPT_COMPILADO else None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._adamw_grad_slices = {}
        self._shared_muon_buffers = {}
        muon_groups = [g for g in self.param_groups if g["kind"] == "muon"]
        if COM_MUON and COM_FRANJAS:
            self._reparto = qf2_franjas.RepartoFranjas(muon_groups, self.rank, self.world_size, COM_FRANJAS)
        else:
            self._reparto = qf2_reparto.RepartoMuon(muon_groups, self.rank, self.world_size, PLANO_TRANSITORIO) if COM_MUON else None

    def restaurar_fp32(self):
        qf2_reparto.restaurar_fp32(self, self._reparto)

    def _shared_muon_buffer(self, name, rows, shape, dtype, device, *, zero=False):
        need = rows * shape[0] * shape[1]
        buf = self._shared_muon_buffers.get(name)
        if buf is None or buf.numel() < need or buf.dtype != dtype or buf.device != device:
            factory = torch.zeros if zero else torch.empty
            buf = factory(need, dtype=dtype, device=device)
            self._shared_muon_buffers[name] = buf
        return buf.flatten()[:need].view(rows, *shape)

    @staticmethod
    def _copy_muon_params(params, stacked):
        if not params:
            return
        if params[0].device.type == "neuron":
            per = params[0].numel()
            torch.split_with_sizes_copy(
                stacked.reshape(-1)[:len(params) * per],
                [per] * len(params),
                out=[p.data.view(-1) for p in params],
            )
        else:
            torch._foreach_copy_(params, list(stacked[:len(params)].unbind(0)))

    @torch.no_grad()
    def _distributed_adamw_step(self, group):
        if self._compilado is not None:
            return self._distributed_adamw_step_compilado(group)
        for param_index, p in enumerate(group["params"]):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grad = p.grad
            if p.numel() < 1024:
                if not COM_ESCALARES:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                self._adamw_update(p, grad, self.state[p], group)
                p.grad = None
                continue

            if p.shape[0] % self.world_size:
                raise ValueError(
                    f"AdamW parameter first dimension {p.shape[0]} must be divisible "
                    f"by world size {self.world_size}"
                )
            rows = p.shape[0] // self.world_size
            grad_slice = self._adamw_grad_slices.get(id(p))
            if grad_slice is None or grad_slice.shape != grad[:rows].shape:
                grad_slice = torch.empty_like(grad[:rows])
                self._adamw_grad_slices[id(p)] = grad_slice
            if COM_RS_BF16:
                qf2_reparto.reducir_tabla_bf16(grad_slice, grad)
            else:
                dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG)

            start = self.rank * rows
            p_slice = p[start:start + rows]
            self._adamw_update(p_slice, grad_slice, self.state[p], group)
            if COM_GATHER_BF16:
                qf2_reparto.reunir_tabla_bf16(p, p_slice, grad, start, rows)
            else:
                dist.all_gather_into_tensor(p, p_slice)
            p.grad = None

    @torch.no_grad()
    def _distributed_adamw_step_compilado(self, group):
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        pequenos = [p for p in group["params"] if p.numel() < 1024]
        if pequenos:
            if not COM_ESCALARES:
                for p in pequenos:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            self._compilado.adamw_paso(pequenos, [p.grad for p in pequenos], [self.state[p] for p in pequenos], group,
                                       ("adamw", id(group)))
            for p in pequenos:
                p.grad = None
        for p in group["params"]:
            if p.numel() < 1024:
                continue
            if p.shape[0] % self.world_size:
                raise ValueError(f"AdamW parameter first dimension {p.shape[0]} must be divisible by world size {self.world_size}")
            grad = p.grad
            rows = p.shape[0] // self.world_size
            grad_slice = self._adamw_grad_slices.get(id(p))
            if grad_slice is None or grad_slice.shape != grad[:rows].shape:
                grad_slice = torch.empty_like(grad[:rows])
                self._adamw_grad_slices[id(p)] = grad_slice
            if COM_RS_BF16:
                qf2_reparto.reducir_tabla_bf16(grad_slice, grad)
            else:
                dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG)
            start = self.rank * rows
            p_slice = p[start:start + rows]
            self._compilado.adamw_paso([p_slice], [grad_slice], [self.state[p]], group, ("adamw", id(group), id(p)))
            if COM_GATHER_BF16:
                qf2_reparto.reunir_tabla_bf16(p, p_slice, grad, start, rows)
            else:
                dist.all_gather_into_tensor(p, p_slice)
            p.grad = None

    @torch.no_grad()
    def _distributed_muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        shape = tuple(p0.shape)
        if any(tuple(p.shape) != shape for p in params):
            raise ValueError("Muon groups must contain equal-shape matrices")

        chunk_size = (len(params) + self.world_size - 1) // self.world_size
        padded_count = chunk_size * self.world_size
        dtype, device = p0.dtype, p0.device
        stacked = self._shared_muon_buffer(
            "stacked", padded_count, shape, dtype, device, zero=True
        )
        grad_chunk = self._shared_muon_buffer(
            "grad_chunk", chunk_size, shape, dtype, device
        )
        updated = self._shared_muon_buffer(
            "updated", chunk_size, shape, dtype, device
        )

        for i, p in enumerate(params):
            if p.grad is None:
                stacked[i].zero_()
            else:
                stacked[i].copy_(p.grad)
            p.grad = None
        dist.reduce_scatter_tensor(grad_chunk, stacked, op=dist.ReduceOp.AVG)

        start = self.rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start))
        if num_owned:
            for i in range(num_owned):
                updated[i].copy_(params[start + i])
            self._muon_owned(group, shape, grad_chunk[:num_owned], updated[:num_owned], self.state[p0])

        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        if COM_GATHER_BF16:
            qf2_reparto.reunir_muon_bf16(stacked, updated, params, start, num_owned)
        else:
            dist.all_gather_into_tensor(stacked, updated)
            self._copy_muon_params(params, stacked)

    @torch.no_grad()
    def _muon_owned(self, group, shape, g, w, state):
        num_owned, dtype, device = g.shape[0], g.dtype, g.device
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_owned, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_owned, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_owned, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=torch.float32, device=device
            )
        if self._compilado is not None:
            self._compilado.muon_paso(group, shape, g, w, state["momentum_buffer"], state["second_momentum_buffer"],
                                      POLAR_EXPRESS_COEFFS[:group["ns_steps"]], COMPUTE_DTYPE == torch.bfloat16)
            return

        momentum_t = group["_momentum_t"]
        one_minus_momentum_t = group["_one_minus_momentum_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        momentum_t.fill_(group["momentum"])
        one_minus_momentum_t.fill_(1 - group["momentum"])
        one_minus_beta2_t.fill_(1 - group["beta2"])
        lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd_t.fill_(group["weight_decay"])

        momentum = state["momentum_buffer"]
        second = state["second_momentum_buffer"]
        m2_den = self._m2_denominator(g, state, group) if OPT == "m2" else None
        momentum.lerp_(g, one_minus_momentum_t)
        g = g.lerp_(momentum, momentum_t)
        if m2_den is not None:
            g = g.div_(m2_den)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        # A one-matrix shard would lower the 3-D products below as batch-1
        # bmm, which the Trainium lazy prologue cannot merge. Use ordinary
        # 2-D matmul for that case and restore the batch dimension after the
        # Polar Express iterations.
        squeezed = x.dim() == 3 and x.size(0) == 1
        if squeezed:
            x = x.squeeze(0)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
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

    @torch.no_grad()
    def step(self):
        # AdamW state is small after row sharding, so process it first. Muon
        # groups remain serial to reuse the large padded communication slab.
        if COM_ESCALARES:
            qf2_reparto.promediar_pequenos(self.param_groups)
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._distributed_adamw_step(group)
        if self._reparto is not None:
            self._reparto.paso(self, COM_GATHER_BF16, COM_RS_BF16)
            return
        for group in self.param_groups:
            if group["kind"] == "muon":
                self._distributed_muon_step(group)


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def optimizer_handles_gradient_sync(optimizer: torch.optim.Optimizer) -> bool:
    return bool(getattr(optimizer, "handles_gradient_sync", False))


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict, extra: dict | None = None, anticipado=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    estado = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if model.config.memhost and qf2_memhost_cfg.CLAVE_Q not in estado and qf2_memhost_cfg.CLAVE_Q not in (extra or {}):
        extra = {**(extra or {}), **qf2_memhost_ckpt.vacias(model.memhost_cfg)}
    if model.config.memcorpus and not all(k in (extra or {}) for k in qf2_corpus_mezcla.claves(model.memcorpus_cfg)):
        extra = {**(extra or {}), **qf2_corpus_mezcla.vacias(model.memcorpus_cfg)}
    payload = {
        "model_state": {**estado, **(extra or {})},
        "model_config": config_dict(model.config),
        "meta": meta,
    }
    if anticipado is not None and anticipado.empezado:
        payload["model_state"] = anticipado.ordenar(payload["model_state"])
        anticipado.terminar(payload)
        return
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    cabecera = qf2_memoria_anfitrion.abrir(checkpoint_path)
    if qf2_memhost_ckpt.usa_memhost(cabecera) or cabecera["model_config"].get("memcorpus"):
        return load_for_eval_memhost(checkpoint_path, device)
    if qf2_memoria_anfitrion.usa_anfitrion(checkpoint_path, MEM_EVAL_ANFITRION_BYTES):
        return load_for_eval_anfitrion(checkpoint_path, device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    tabla_mem = payload["model_state"].get("memoria.2.weight")
    model = GPT(config, memoria_dtype=tabla_mem.dtype if tabla_mem is not None else torch.float32)
    model.init_weights(memoria=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_for_eval_memhost(checkpoint_path: str, device: torch.device) -> nn.Module:
    qf2_memoria_anfitrion.precargar(checkpoint_path)
    payload = qf2_memoria_anfitrion.abrir(checkpoint_path)
    state = payload["model_state"]
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    propias = set(qf2_memhost_ckpt.claves() if config.memhost else []) | set(qf2_corpus_mezcla.claves(model.memcorpus_cfg) if config.memcorpus else [])
    res = model.load_state_dict({k: v for k, v in state.items() if k not in propias}, strict=False)
    if res.unexpected_keys or res.missing_keys:
        raise RuntimeError(f"claves inesperadas al cargar sin tablas: faltan={res.missing_keys} sobran={res.unexpected_keys}")
    model.to(device)
    if config.memhost:
        qf2_memhost_ckpt.asignar(model, state)
    if config.memcorpus:
        qf2_corpus_mezcla.asignar(model, state)
    model.eval()
    return model


def load_for_eval_anfitrion(checkpoint_path: str, device: torch.device) -> nn.Module:
    qf2_memoria_anfitrion.precargar(checkpoint_path)
    payload = qf2_memoria_anfitrion.abrir(checkpoint_path)
    state = payload["model_state"]
    config = GPTConfig(**payload["model_config"])
    model = GPT(config, memoria_dtype=state["memoria.2.weight"].dtype, memoria_device="meta")
    model.init_weights(memoria=False)
    qf2_memoria_anfitrion.cargar_resto(model, state)
    tablas, model.memoria = model.memoria, nn.ModuleDict()
    model.to(device)
    model.memoria = tablas
    qf2_memoria_anfitrion.asignar(model, state)
    model.memoria_anfitrion = True
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Training

def lr_bajada(remaining: float) -> float:
    if LR_SHAPE == "sqrt":
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * (1.0 - math.sqrt(max(0.0, 1.0 - remaining)))
    return remaining + (1.0 - remaining) * FINAL_LR_FRAC


def lr_multiplier(step: int, target_steps: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    warmdown_steps = round(WARMDOWN_RATIO * target_steps)
    if step <= target_steps - warmdown_steps:
        return 1.0
    return lr_bajada((target_steps - step) / max(1, warmdown_steps))


def lr_multiplier_reloj(step: int, progress: float) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    remaining = max(0.0, 1.0 - progress) / WARMDOWN_RATIO
    if remaining >= 1.0:
        return 1.0
    return lr_bajada(remaining)


def suelo_momentum(warmup_frac: float, warmdown_remaining: float) -> float:
    if warmup_frac < 1.0:
        return SUELO_MOMENTUM_START + (SUELO_MOMENTUM_PEAK - SUELO_MOMENTUM_START) * max(0.0, warmup_frac)
    if warmdown_remaining < 1.0:
        return SUELO_MOMENTUM_END + (SUELO_MOMENTUM_PEAK - SUELO_MOMENTUM_END) * max(0.0, warmdown_remaining)
    return SUELO_MOMENTUM_PEAK


def muon_momentum(step: int, target_steps: int) -> float:
    if not SUELO_MOM:
        return MUON_MOMENTUM
    warmdown_steps = round(WARMDOWN_RATIO * target_steps)
    warmup_frac = step / max(1.0, SUELO_MOMENTUM_WARMUP_FRAC * target_steps)
    return suelo_momentum(warmup_frac, (target_steps - step) / max(1, warmdown_steps))


def muon_momentum_reloj(progress: float) -> float:
    if not SUELO_MOM:
        return MUON_MOMENTUM
    return suelo_momentum(progress / SUELO_MOMENTUM_WARMUP_FRAC, max(0.0, 1.0 - progress) / WARMDOWN_RATIO)


def wd_muon(progress: float) -> float:
    if not WD_COS:
        return WD_PICO
    return WD_PICO * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def reloj_progress_from(control: torch.Tensor, max_train_seconds: float) -> float:
    elapsed_ms, slowest_ms = control[2:].tolist()
    return ((elapsed_ms + 2 * slowest_ms) / 1000.0 + CHECKPOINT_RESERVE_SECONDS) / max_train_seconds


def main() -> None:
    qf2_hilos_aten.fijar_proceso()
    parser = argparse.ArgumentParser(description="Neuron Competition Round 2 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--aspect-ratio", type=int, default=ASPECT_RATIO)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--device-batch-size", type=int, default=DEVICE_BATCH_SIZE)
    parser.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_NUM_STEPS", NUM_STEPS)),
                        help="explicit step override; <=0 derives steps from --target-param-data-ratio")
    parser.add_argument("--target-param-data-ratio", type=float,
                        default=float(os.environ.get("NEURON_COMPETITION_R2_TARGET_PARAM_DATA_RATIO", TARGET_PARAM_DATA_RATIO)),
                        help="target token IDs per scaling parameter (Transformer matrices + lm_head)")
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R2_OUT_DIR", "out"))
    parser.add_argument(
        "--eval-public",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NEURON_COMPETITION_R2_EVAL_PUBLIC", "0") != "0",
        help="run the optional development-shard BPB evaluation after checkpointing",
    )
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action="store_true", default=os.environ.get("NEURON_COMPETITION_R2_COMPILE", "0") != "0",
                        help="enable torch.compile on the full model")
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic training step without reading training shards")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R2_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    args = parser.parse_args()
    if LR_SCHEDULE not in ("steps", "reloj"):
        raise ValueError(f"QF2_LR_SCHEDULE must be 'steps' or 'reloj', got {LR_SCHEDULE!r}")
    if OPT not in OPT_VALORES:
        raise ValueError(f"QF2_OPT must be one of {OPT_VALORES}, got {OPT!r}")
    if EMB_SCATTER and EMB_DETACH:
        raise ValueError("QF2_EMB_SCATTER=1 and QF2_EMB_DETACH=1 both replace the table backward")
    if EMB_SCATTER_GATE and not EMB_SCATTER:
        raise ValueError("QF2_EMB_SCATTER_GATE=1 needs QF2_EMB_SCATTER=1")

    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2

    device_type = detect_device_type(args.device_type)
    constructor = None
    if MEMCORPUS and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tok_mc = ensure_tokenizer(build_if_missing=False)
        shards_mc = qf2_corpus_constructor.shards_del_constructor(list_shard_paths(PUBLIC_TRAIN_SHARDS, require_all=False), MEMCORPUS_EXCLUIR)
        constructor = qf2_corpus_constructor.Constructor(qf2_corpus_mezcla.desde_texto(MEMCORPUS), Path(MEMCORPUS_RUTA), shards_mc,
                                                         MEMCORPUS_PROCESOS, tok_mc.get_bos_token_id(), tok_mc)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE, CHECKPOINT_RESERVE_SECONDS
    COMPUTE_DTYPE = compute_dtype_for(device)
    if MEMHOST:
        CHECKPOINT_RESERVE_SECONDS = max(CHECKPOINT_RESERVE_SECONDS, MEMHOST_RESERVA_S)
    if MEMCORPUS:
        CHECKPOINT_RESERVE_SECONDS += MEMCORPUS_RESERVA_S

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    if CEFA:
        print0(cefa_descripcion())
    if SUELO_MOM or SUELO_ESC or SUELO_LR or SUELO_INIT or VALUE_EMBEDS or OPT or SEED != 42:
        print0(f"qf2_suelo: a={int(SUELO_A)} mom={int(SUELO_MOM)} esc={int(SUELO_ESC)} opt={f'{OPT}:{OPT_BETA2}:{OPT_EPS}' if OPT else '-'} lr={int(SUELO_LR)} init={int(SUELO_INIT)} ve={int(VALUE_EMBEDS)} seed={SEED} lr_scale={(args.total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0:.6f}")

    print0(f"qf2_lr: forma={LR_SHAPE} bajada={WARMDOWN_RATIO!r} final={FINAL_LR_FRAC!r} lrm(r=1,0.5,0)={lr_bajada(1.0):.6g},{lr_bajada(0.5):.6g},{lr_bajada(0.0):.6g}")
    print0(f"qf2_wd: cos={int(WD_COS)} pico={WD_PICO!r} muon_wd(p=0,0.5,1)={wd_muon(0.0):.6g},{wd_muon(0.5):.6g},{wd_muon(1.0):.6g} p={'reloj_progress' if LR_SCHEDULE == 'reloj' else 'step/target_steps'}")
    if FP8_MLP:
        print0(f"qf2_fp8_mlp: {FP8_RESUMEN}")
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        sequence_len=qf2_cola.maximo(args.seq_len) if COLA4096 else args.seq_len,
    )
    print0(f"model_config={json.dumps(config_dict(config), indent=2)}")
    repartida = bool(config.memoria) and dist.is_available() and dist.is_initialized()
    if repartida and (PROMEDIO or PUNTOS or args.eval_public):
        raise ValueError("QF2_MEM_FILAS repartida es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S y --eval-public")
    if config.memhost and (PROMEDIO or PUNTOS or args.eval_public or COLA_LARGA or not PRECARGA or repartida):
        raise ValueError("QF2_MEMHOST necesita QF2_PRECARGA=1 y es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S, --eval-public, la cola larga (QF2_SEQ_MAX, QF2_COLA_FRAC, QF2_CALIENTA_LARGO) y la memoria repartida")
    if config.memcorpus and (PROMEDIO or PUNTOS or FLUJO or COLA_LARGA or not PRECARGA or repartida):
        raise ValueError("QF2_MEMCORPUS necesita QF2_PRECARGA=1 y es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S, QF2_FLUJO, la cola larga y la memoria repartida")
    if PESO_CTX.activo and (config.memoria or config.memhost or config.memcorpus or config.mos or config.registro or config.prefijo_bos or config.ngramas
                            or CEFA or CE_TROZOS or CE_LSE or COLA_LARGA or args.seq_len % PESO_CTX.k):
        raise ValueError("QF2_PESO_CTX es incompatible con memoria, memhost, memcorpus, mos, registro, prefijo_bos, ngramas, QF2_CEFA, QF2_CE_TROZOS, QF2_CE_LSE y la cola larga, y exige seq_len multiplo de K")
    model = GPT(config, repartida=repartida)
    model.init_weights()
    tabla_mc = None
    if config.memcorpus:
        qf2_corpus_constructor.esperar_creada(Path(MEMCORPUS_RUTA))
        tabla_mc = qf2_corpus_constructor.abrir(Path(MEMCORPUS_RUTA), model.memcorpus_cfg)
        model.memcorpus_tabla = tabla_mc
        if constructor is not None:
            print0(constructor.descripcion())
            print0(f"qf2_memcorpus: tronco={MEMCORPUS_TRONCO} contexto={model.memcorpus_cfg.contexto}")
    anticipado = None
    if constructor is not None and MEMCORPUS_ANTICIPADO_S > 0:
        anticipado = qf2_guardado_anticipado.Anticipado(Path(args.out_dir) / "final.pt")
    if config.memhost and model.memhost_cfg.canon:
        model.memhost.canon.copy_(torch.from_numpy(qf2_memhost_canon.construir(tokenizer.decode, config.vocab_size, model.memhost_bos)))
    inicio = qf2_inicio.abrir(INICIO_DESDE, config_dict(config)) if INICIO_DESDE else None
    if inicio is not None:
        print0(qf2_inicio.cargar_tronco(model, inicio, BANCO_LRM_TRAMO))
    model.to(device)
    mem_obj = qf2_memoria.Memoria(config.memoria, config.memoria_ancho, device, SEED, model.memoria_bos, MEM_TRANSPORTE) if repartida else None
    mem_opt = qf2_memoria_opt.AdamShard(mem_obj.parametros(), model.memoria_lr(args.total_batch_size), MuonAdamW._adamw_update) if repartida else None
    num_params = sum(p.numel() for p in model.parameters())
    scaling_param_counts = model.num_scaling_params()
    num_scaling_params = scaling_param_counts["total"]
    flops_per_token = model.estimate_flops()
    if repartida:
        print0(f"qf2_memoria: repartida filas={config.memoria} ancho={config.memoria_ancho} rangos={world_size} filas_por_rango={mem_obj.r} params_globales={2 * config.memoria * config.memoria_ancho:,} transporte={MEM_TRANSPORTE}")
    print0(f"num_params={num_params:,} scaling_params={num_scaling_params:,} flops_per_token={flops_per_token:e}")
    if EMB_RUTA:
        print0(emb_descripcion(device, EMB_DETACH, 1 + len(model.value_embeds) + len(model.ngramas), model.transformer.wte.weight.shape, COMPUTE_DTYPE))
        if EMB_SCATTER and device.type == "neuron":
            emb_preparar()

    world_tokens_per_micro = args.device_batch_size * args.seq_len * world_size
    if args.total_batch_size % world_tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_size={args.total_batch_size} must divide micro tokens={world_tokens_per_micro}"
        )
    grad_accum_steps = args.total_batch_size // world_tokens_per_micro
    local_step_batch_size = args.device_batch_size * grad_accum_steps
    cola = qf2_cola.Cola(args.seq_len, args.device_batch_size, args.total_batch_size, world_size, config.sequence_len, print0) if COLA4096 else None
    loader_rows, loader_len = cola.cargador(local_step_batch_size, args.seq_len) if cola else (local_step_batch_size, args.seq_len)
    mem_forma = (loader_rows * loader_len // args.seq_len, args.seq_len) if cola else (loader_rows, loader_len)
    if cola:
        print0(f"{qf2_cola.descripcion()} loader={loader_rows}x{loader_len} grad_accum_split={cola.micro_partido} grad_accum_long={cola.micro_largo}")
    if args.num_steps > 0:
        target_steps = args.num_steps
        horizon_source = "explicit steps"
    elif LR_SCHEDULE == "reloj":
        target_steps = RELOJ_STEP_CEILING
        horizon_source = "reloj"
    else:
        if args.target_param_data_ratio <= 0:
            raise ValueError("--target-param-data-ratio must be positive when --num-steps is not set")
        target_tokens = int(args.target_param_data_ratio * num_scaling_params)
        target_steps = target_tokens // args.total_batch_size
        if target_steps <= 0:
            raise ValueError("target ratio produces zero optimizer steps")
        horizon_source = "data:param ratio"
    target_tokens = target_steps * args.total_batch_size
    actual_target_ratio = target_tokens / num_scaling_params
    print0(
        f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} "
        f"local_step_batch_size={local_step_batch_size}"
    )
    print0(
        f"training_horizon={horizon_source} target_steps={target_steps:,} "
        f"target_tokens={target_tokens:,} target_scaling_ratio={actual_target_ratio:.4f} "
        f"max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s"
    )

    orig_model = model
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    if args.compile_only:
        # Match the real loader's packed local-step input and int32 token dtype so
        # the timed process can reuse this exact graph specialization.
        synthetic_host_x = torch.zeros((loader_rows, loader_len), dtype=torch.int32)
        synthetic_host_y = torch.zeros((loader_rows, loader_len), dtype=torch.long)
        synthetic_x = synthetic_host_x.to(device)
        synthetic_y = synthetic_host_y.to(device)
        optimizer = orig_model.setup_optimizer(args.total_batch_size)
        synthetic_e = torch.zeros(*mem_forma, orig_model.memhost_cfg.ancho_e, dtype=COMPUTE_DTYPE, device=device) if config.memhost else None
        hojas_sinteticas = qf2_memhost_hojas.Hojas() if config.memhost else None
        for synthetic_step in (cola.pasos_compilacion() if cola else (0,)):
            model.zero_grad(set_to_none=True)
            xs, ys, n_micro, filas = cola.lote(synthetic_x, synthetic_y, synthetic_step, 0.0, grad_accum_steps) if cola else (synthetic_x, synthetic_y, grad_accum_steps, args.device_batch_size)
            if EQUILIBRIO is not None:
                qf2_equilibrio.fijar(cola is not None and qf2_cola.CARGADOR_LARGO and cola.forma is False)
            if mem_opt is not None:
                mem_opt.zero_grad()
            hs = hojas_sinteticas.repartir(synthetic_e, n_micro, filas) if hojas_sinteticas is not None else None
            for micro_step in range(n_micro):
                start = micro_step * filas
                end = start + filas
                loss = perdida_micro(model, mem_obj, xs[start:end], ys[start:end], hs[micro_step] if hs is not None else None, None, synthetic_step, micro_step)
                (loss / n_micro).backward()
            if not optimizer_handles_gradient_sync(optimizer):
                sync_gradients(orig_model)
            optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    # Transfer one complete local optimizer-step batch before launching any
    # microbatch graph. Reusing and overwriting a single device input between
    # asynchronous Neuron graph calls forces a synchronization per microbatch.
    memhost, memhost_lr = None, 0.0
    if FLUJO:
        print0(qf2_flujo.descripcion(SEED, loader_len))
    if PUNTOS:
        train_loader = qf2_flujo.make_flujo_dataloader(tokenizer, loader_rows, loader_len, device, SEED, print0 if FLUJO else (lambda *_: None),
                                                       p=None if FLUJO else 0, reanudable=True)
    elif PRECARGA:
        if config.memhost:
            memhost = qf2_memhost_hilo.Memhost(orig_model.memhost_cfg, MEMHOST_RUTA, rank, world_size, mem_forma, COMPUTE_DTYPE,
                                               orig_model.memhost_bos, MEMHOST_RETRASO, MEMHOST_OPT,
                                               orig_model.memhost.canon.cpu().numpy() if orig_model.memhost_cfg.canon else None,
                                               inicio=qf2_inicio.tablas(inicio, print0))
            memhost_lr = MEMHOST_LR or orig_model.memoria_lr(args.total_batch_size)
            if MEMHOST_DECAY:
                qf2_memhost_opt.DECAY = float(MEMHOST_DECAY)
            print0(f"qf2_memhost: cfg={config.memhost} filas={orig_model.memhost_cfg.filas:,} retraso={MEMHOST_RETRASO} opt={MEMHOST_OPT} lr={memhost_lr} decay={qf2_memhost_opt.DECAY} ruta={MEMHOST_RUTA} reserva_s={CHECKPOINT_RESERVE_SECONDS}")
        corpus_mc = (lambda x: qf2_corpus_mezcla.consultar(tabla_mc, x.numpy(), orig_model.memcorpus_bos, orig_model.memcorpus_cfg.minimo)) if tabla_mc is not None else None
        train_loader = qf2_precarga.Precarga(lambda: qf2_filas.cargador(make_dataloader(tokenizer, loader_rows, loader_len, "train", "cpu", shard_ids=qf2_inicio.shards_banco()), tokenizer, rank, world_size, SEED, args.seq_len, AGRUPA, RECOLOCADO, RECOLOCADO_CE, RECOLOCADO_RAFAGA, RECOLOCADO_FORMATO, PARES_TRAMO, PARES_ETIQUETAS), device, memhost=memhost,
                                            forma_mem=mem_forma, corpus=corpus_mc)
        print0(f"qf2_precarga: hilo con cola de {qf2_precarga.PROFUNDIDAD} lotes de {loader_rows}x{loader_len} en CPU, copia al dispositivo en el hilo principal")
    else:
        train_loader = qf2_flujo.make_flujo_dataloader(tokenizer, loader_rows, loader_len, device, SEED, print0) if FLUJO else make_dataloader(tokenizer, loader_rows, loader_len, "train", device)
    step_x, step_y, loader_state = next(train_loader)
    optimizer = orig_model.setup_optimizer(args.total_batch_size)
    restaurar_fp32 = optimizer.restaurar_fp32 if COM_GATHER_BF16 and isinstance(optimizer, DistMuonAdamW) else None
    if COM_ESCALARES or COM_MUON or COM_GATHER_BF16 or COM_RS_BF16:
        print0(qf2_reparto.descripcion(COM_ESCALARES, COM_MUON, COM_GATHER_BF16, optimizer, COM_RS_BF16))
    if OPT_COMPILADO and getattr(optimizer, "_compilado", None) is not None:
        print0(optimizer._compilado.descripcion())
    promedio = Promedio(orig_model.parameters(), PROMEDIO_MODO == "host", antes=restaurar_fp32) if PROMEDIO else None
    if promedio is not None:
        print0(promedio.descripcion())
    guarda = Guarda(list(orig_model.parameters()) + (mem_obj.parametros() if mem_obj else []), TRAIN_STARTUP_STEPS_EXCLUDED, rank) if GUARDA else None
    if guarda is not None:
        print0(guarda.descripcion())
    puntos = qf2_puntos.Puntos(PUNTOS_S, REANUDA, args.out_dir, rank, world_size, args, print0) if PUNTOS else None
    puntos_extras = {"guarda": guarda, "cola": cola, "promedio": promedio}
    if puntos is not None:
        print0(puntos.descripcion())
    reanudado = puntos.reanudar(orig_model, optimizer, train_loader, puntos_extras, restaurar_fp32 is not None, device) if puntos else None
    if reanudado:
        step_x, step_y, loader_state = reanudado["lote"]
    if EMB_SCATTER_GATE and rank == 0:
        if not emb_puerta(device, step_x[:args.device_batch_size], orig_model.transformer.wte.weight.shape, COMPUTE_DTYPE):
            raise SystemExit(3)
    if UNION is not None:
        print0(UNION.descripcion(RUIDO if RUIDO.activo else None))
    if UNION is None and (OBJETIVO is not None or RUIDO.activo):
        print0(qf2_objetivos.descripcion(OBJETIVO, RUIDO))
    if UNION is None and BRIER is not None:
        print0(BRIER.descripcion())
    if UNION is None and EQUILIBRIO is not None:
        print0(EQUILIBRIO.descripcion())
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    reloj_progress = 0.0
    paso_inicial = 0
    if reanudado:
        step, smooth_loss, total_train_tokens = reanudado["step"], reanudado["smooth_loss"], reanudado["total_train_tokens"]
        startup_allowance_est, reloj_progress = reanudado["startup_allowance_est"], reanudado["reloj_progress"]
        recent_step_times = list(reanudado["recent_step_times"])
        budget_started = time.monotonic() - reanudado["entrenado_s"]
        paso_inicial = step
    hojas = qf2_memhost_hojas.Hojas() if memhost is not None else None
    reloj_log = open(f"{RELOJ_LOG}.rank{rank}", "a" if reanudado else "w", encoding="utf-8") if RELOJ_LOG else None

    while step < target_steps:
        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if mem_opt is not None:
            mem_opt.zero_grad()
        last_loss = None
        xs, ys, n_micro, filas = cola.lote(step_x, step_y, step, reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps, grad_accum_steps) if cola else (step_x, step_y, grad_accum_steps, args.device_batch_size)
        if EQUILIBRIO is not None:
            qf2_equilibrio.fijar(cola is not None and qf2_cola.CARGADOR_LARGO and cola.forma is False)
        hs = hojas.repartir(train_loader.e, n_micro, filas) if hojas is not None else None
        cs = (train_loader.c.reshape(xs.shape), train_loader.k.reshape(xs.shape)) if tabla_mc is not None else None
        for micro_step in range(n_micro):
            start = micro_step * filas
            end = start + filas
            loss = perdida_micro(model, mem_obj, xs[start:end], ys[start:end], hs[micro_step] if hs is not None else None,
                                 (cs[0][start:end], cs[1][start:end]) if cs is not None else None, step, micro_step)
            last_loss = loss.detach()
            (loss / n_micro).backward()

        saltar = guarda is not None and guarda.saltar(step)
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not saltar and not optimizer_handles_gradient_sync(optimizer):
            sync_gradients(orig_model)
        if RELOJ_TRAMOS:
            synchronize(device)
            t_fb = time.monotonic()

        if LR_SCHEDULE == "reloj":
            lrm = lr_multiplier_reloj(step, reloj_progress)
            momentum = muon_momentum_reloj(reloj_progress)
        else:
            lrm = lr_multiplier(step, target_steps)
            momentum = muon_momentum(step, target_steps)
        if BANCO_LRM_TRAMO is not None:
            lrm = qf2_inicio.multiplicador(BANCO_LRM_TRAMO, reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = wd_muon(reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps)
        if mem_opt is not None:
            for group in mem_opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        if memhost is not None:
            g_mem = None if saltar else hojas.gradiente(train_loader.e.shape[0])
            if not saltar and g_mem is None:
                raise RuntimeError("qf2_memhost: el gradiente de los vectores recuperados no llego a las hojas")
            t_copia = time.monotonic()
            g_mem = None if saltar else g_mem.detach().to("cpu")
            memhost.anotar("copia_d2h", time.monotonic() - t_copia)
        if not saltar:
            optimizer.step()
            if mem_opt is not None:
                mem_opt.step()
        if memhost is not None:
            memhost.postear(g_mem, memhost_lr * lrm, lrm)

        synchronize(device)
        if RELOJ_TRAMOS:
            t_opt = time.monotonic()
        if not saltar:
            record_step_boundary(step)
        step_x, step_y, loader_state = next(train_loader)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        if RELOJ_TRAMOS and rank == 0:
            print(f"RELOJ_TRAMOS {step} fb={t_fb - t0:.4f} opt={t_opt - t_fb:.4f} resto={t0 + dt - t_opt:.4f} dt={dt:.4f}", flush=True)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped), so
            # restart the budget clock here to use the full charged budget.
            # If startup exceeded the cap, the excess is already charged:
            # back-date the clock by that excess, with a margin for the
            # launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(
                0.0,
                observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
            )
            budget_started = now - excess
            if constructor is not None:
                constructor.empezar()
        if puntos is not None and puntos.calienta():
            recent_step_times.pop()
            budget_started = time.monotonic() - puntos.entrenado_s
        elapsed_after_step = time.monotonic() - budget_started
        if anticipado is not None and not anticipado.empezado and step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED and (
                constructor.acabado() or elapsed_after_step >= max(args.max_train_seconds - MEMCORPUS_ANTICIPADO_S, 0.5 * args.max_train_seconds)):
            print(f"qf2_memcorpus: anticipado en el paso {step} a {elapsed_after_step:.0f} s, constructor {constructor.parar()}", flush=True)
            anticipado.empezar(qf2_corpus_mezcla.para_guardar(tabla_mc, orig_model.memcorpus_cfg.minimo))
        # Fold the deadline decision into the existing health collective. A
        # local wall-clock check can otherwise make one rank begin another
        # optimizer step while its peers enter the checkpoint barrier.
        # The budget clock is only meaningful once it has been reset at the
        # excluded-startup boundary; before that, elapsed_after_step is measured
        # from loop start and a slow first-step compile could trip the predictive
        # stop and break before five steps complete — leaving the run unscorable.
        # Gate the time-based stop behind reaching the excluded-step count so the
        # run is always scorable; an explicit target_steps budget still stops early.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= target_steps
            or (
                reached_min_steps
                and (
                    elapsed_after_step >= args.max_train_seconds
                    or elapsed_after_step
                    + max(recent_step_times[-20:])
                    + CHECKPOINT_RESERVE_SECONDS
                    >= args.max_train_seconds
                )
            )
        )
        stop_after_step = stop_after_step or (guarda is not None and guarda.agotada())
        reloj_clock = []
        if LR_SCHEDULE == "reloj" or puntos is not None:
            reloj_clock = [round(elapsed_after_step * 1000), round(max(recent_step_times[-20:]) * 1000)]
        control = torch.tensor(
            [
                1 if (not saltar and (math.isnan(train_loss) or train_loss > 100)) else 0,
                1 if stop_after_step else 0,
                *reloj_clock,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")
        if reloj_log is not None:
            suelo_campo = f" {momentum!r}" if SUELO_MOM else ""
            reloj_log.write(f"{step} {lrm!r} {reloj_progress!r} {elapsed_after_step!r}{suelo_campo}\n")
        if LR_SCHEDULE == "reloj" and reached_min_steps:
            reloj_progress = max(reloj_progress, reloj_progress_from(control, args.max_train_seconds))
        if promedio is not None and not saltar:
            promedio.tras_paso(step + 1, reloj_progress if LR_SCHEDULE == "reloj" else (step + 1) / target_steps,
                               int(control[1].item()) != 0)

        if not saltar:
            total_train_tokens += args.total_batch_size
            smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = args.total_batch_size / max(dt, 1e-6)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = target_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | "
                f"remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}"
                + (f" | mem_desbordes {int(mem_obj.desbordes.item())}" if mem_obj is not None and MEM_TRANSPORTE == "a2a" else ""),
                flush=True,
            )

        if step == paso_inicial:
            # After first step all persistent objects are allocated; freeze to avoid GC pauses
            gc.collect()
            gc.freeze()
            gc.disable()
        if not saltar:
            step += 1
        if int(control[1].item()) != 0:
            break
        if puntos is not None and puntos.toca(control):
            bucle = {"step": step, "smooth_loss": smooth_loss, "total_train_tokens": total_train_tokens,
                     "recent_step_times": recent_step_times[-20:], "startup_allowance_est": startup_allowance_est,
                     "reloj_progress": reloj_progress, "entrenado_ms": int(control[2].item()),
                     "max_train_seconds": args.max_train_seconds}
            budget_started += puntos.guardar(step, bucle, orig_model, optimizer, train_loader, puntos_extras,
                                             restaurar_fp32 is not None, device)

    if reloj_log is not None:
        reloj_log.close()
    if guarda is not None:
        print0(f"qf2_guarda: saltos={guarda.saltos}")
    if constructor is not None:
        leidos = total_train_tokens / qf2_corpus_constructor.TOKENS_POR_SHARD
        print(f"qf2_memcorpus: parado {constructor.parar()} ocupacion~{tabla_mc.ocupacion(4096)} la red leyó ~{leidos:.0f} shards de tokens"
              f" (el constructor no baja del {MEMCORPUS_EXCLUIR}){' AVISO: SOLAPE POSIBLE' if 2 * leidos > MEMCORPUS_EXCLUIR else ''}", flush=True)
    extra_memhost = None
    if memhost is not None:
        n_memhost = memhost.fin()
        print0(memhost.resumen())
        extra_memhost = qf2_memhost_ckpt.cuantizar_propias(memhost.tabla, 2 * n_memhost + 4)
    if PRECARGA:
        print0(train_loader.resumen())
        train_loader.cerrar()
    if restaurar_fp32 is not None:
        restaurar_fp32()
    if promedio is not None:
        print0(f"qf2_promedio: pasos={promedio.instalar()}")
    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    extra_mem = qf2_memoria_ckpt.consolidar(mem_obj, COMPUTE_DTYPE if COMPUTE_DTYPE == torch.bfloat16 else torch.float32) if mem_obj else extra_memhost
    if tabla_mc is not None and rank == 0:
        extra_mem = {**(extra_mem or {}), **qf2_corpus_mezcla.para_guardar(tabla_mc, orig_model.memcorpus_cfg.minimo)}
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        }, extra_mem, anticipado)
        if anticipado is not None and anticipado.empezado:
            print(f"qf2_memcorpus: tablas escritas antes en {anticipado.segundos:.0f} s; guardado final {time.monotonic() - budget_started - budget_elapsed_before_save:.1f} s", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    if constructor is not None:
        constructor.limpiar()
    if memhost is not None and rank == 0:
        memhost.tabla.cerrar()

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, args.device_batch_size, args.seq_len, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            args.device_batch_size,
            args.seq_len,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"horizon_source:    {horizon_source}")
        print(f"lr_schedule:       {LR_SCHEDULE} last_lrm={lrm:.4f}")
        print(f"target_steps:      {target_steps}")
        print(f"target_tokens_M:   {target_tokens / 1e6:.1f}")
        print(f"target_ratio:      {args.target_param_data_ratio:.4f}")
        print(f"actual_target_ratio: {actual_target_ratio:.4f}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > args.max_train_seconds}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f"depth:             {args.depth}")
        print(f"aspect_ratio:      {args.aspect_ratio}")
        print(f"seq_len:           {args.seq_len}")
        print(f"device_batch_size: {args.device_batch_size}")
        print(f"total_batch_size:  {args.total_batch_size}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        print(f"scaling_params_M:  {num_scaling_params / 1e6:.1f}")
        print(f"actual_ratio:      {total_train_tokens / num_scaling_params:.4f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")

    cleanup_runtime()


if __name__ == "__main__":
    main()

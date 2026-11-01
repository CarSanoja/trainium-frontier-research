"""EMB: embedding tables through an autograd Function. Forward is F.embedding cast to the compute dtype (the same ops as
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

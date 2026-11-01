"""EMB gate (QF2_EMB_SCATTER_GATE=1, rank 0, before the training loop): dW of emb_dw on the device over the ids of the first
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

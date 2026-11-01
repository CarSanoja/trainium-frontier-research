"""QF2_MEMHOST: checkpoint writing and evaluation reads for the host tables.

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

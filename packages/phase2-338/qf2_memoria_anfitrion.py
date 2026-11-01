from __future__ import annotations

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

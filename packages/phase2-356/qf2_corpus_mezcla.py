"""QF2_MEMCORPUS: mezcla del modelo con la memoria de n-gramas del corpus (qf2_corpus_tabla), igual en entreno y evaluacion.

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

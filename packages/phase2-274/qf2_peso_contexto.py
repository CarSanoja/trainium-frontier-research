"""QF2_PESO_CTX: pesos por token segun el beneficio del contexto, solo en entreno (brazos B y C de f2-peso-beneficio-contexto).

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

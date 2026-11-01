"""Rasgos de host y tabla para QF2_PESO_CTX_PRED=1: predice el beneficio del contexto de cada token sin medirlo.

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

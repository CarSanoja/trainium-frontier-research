"""QF2_CE_DIFICILES, QF2_CE_TRAMO y QF2_RUIDO_ENTRADA: objetivo y regularización del banco CORE bifurcado, ronda 2.

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

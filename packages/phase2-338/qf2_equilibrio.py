"""QF2_EQUILIBRIO='mezcla:tope': CE equilibrada por longitud de documento (propuesta f2-equilibrio-documental, astra
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

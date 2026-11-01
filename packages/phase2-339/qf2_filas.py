"""Filas sustituidas del lote de entreno en el paquete 302: QF2_AGRUPA y QF2_RECOLOCADO sobre filas disjuntas.

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

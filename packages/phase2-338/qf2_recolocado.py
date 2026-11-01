"""QF2_RECOLOCADO: una fraccion fija de filas de cada lote de entreno se sustituye por filas de ejemplos recolocados.

Un hilo propio lee documentos de los shards de entreno (00000-06499, los que lista prepare.list_shard_paths con
PUBLIC_TRAIN_SHARDS, con el lector de parquet de prepare), construye ejemplos (qf2_recolocado_ejemplos), los tokeniza
con el tokenizador del kit (cada ejemplo empieza en BOS, como los documentos) y los empaqueta en filas del ancho del
cargador. envolver() sustituye en cada lote, antes de que lo clone la precarga, n filas: n sale de un acumulador
(error difundido) para que la fraccion sea exacta a la larga y el numero de filas, y por tanto de tokens, no cambie;
cual de ellas se decide por (semilla, rango, paso). La evaluacion no se toca. Con fraccion 0 cargador() devuelve el
mismo generador del kit.

QF2_RECOLOCADO_CE: con "todo" los objetivos son los del texto completo. Con "continuacion" los objetivos (-1, el
ignore_index de la CE del kit) de los tokens del BOS, del enunciado y de las dos etiquetas valen -1 y solo cuenta la
continuacion; la entrada x no cambia. El corte en tokens es el prefijo comun entre la tokenizacion del ejemplo entero
y la del tramo hasta la segunda etiqueta: si un token cruza la frontera queda fuera de la mascara.

QF2_RECOLOCADO_RAFAGA: vacio deja el comportamiento anterior (un ejemplo por documento recolocado). Con "a:b" cada
documento recolocado es BOS + k ejemplos del mismo tipo sobre tramos distintos del documento fuente (k uniforme en
[a, b] con el mismo rng, qf2_recolocado_ejemplos.documentos_de). Con "continuacion" la mascara se extiende a cada ejemplo
de la rafaga: objetivo -1 desde el final de una continuacion (incluido un token que cruce esa frontera) hasta la segunda
etiqueta del ejemplo siguiente, con el mismo prefijo comun de tokens en cada frontera (qf2_recolocado_ejemplos.fronteras).

QF2_RECOLOCADO_FORMATO: "ejemplos" (por defecto) es todo lo anterior. Con "pares" cada documento del flujo se escribe entero
y en orden como pares etiquetados (qf2_recolocado_pares, tramos de QF2_PARES_TRAMO palabras, etiquetas de QF2_PARES_ETIQUETAS);
exige QF2_RECOLOCADO_CE=todo y QF2_RECOLOCADO_RAFAGA vacia. Se valida aunque la fraccion sea 0.

ocupadas (paquete 302, qf2_filas): si se da, envolver() elige sus filas solo entre ocupadas.libres(paso, filas), las que
otro sustituidor no tomo en ese paso; n sale del mismo acumulador sobre el total de filas, recortado a las libres, y lo
que no cabe queda en el acumulador. Sin ocupadas (por defecto) la eleccion es la de antes.
"""
from __future__ import annotations

import queue
import random
import threading
import time

import torch

import qf2_hilos_nice as nice
import qf2_recolocado_ejemplos as ejemplos
import qf2_recolocado_pares as pares_m

ESPERA_S = 120.0
TOKENIZAR_DE_A = 64
BUFFER_MIN = 64
CE_MODOS = ("todo", "continuacion")


def leer_rafaga(texto: str) -> tuple[int, int] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes) or not 2 <= int(partes[0]) <= int(partes[1]):
        raise ValueError(f"QF2_RECOLOCADO_RAFAGA={texto!r}: se espera 'a:b' con 2 <= a <= b")
    return int(partes[0]), int(partes[1])


def documentos_de_shards(rank: int, world: int, semilla: int, rutas=None):
    import pyarrow.parquet as pq
    if rutas is None:
        import prepare
        import qf2_inicio
        rutas = prepare.list_shard_paths(qf2_inicio.shards_banco() or prepare.PUBLIC_TRAIN_SHARDS, require_all=False)
    rutas = sorted(rutas)
    j = 0
    while True:
        pf = pq.ParquetFile(rutas[(rank + world * j + 7919 * semilla) % len(rutas)])
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g).column("text").to_pylist()
        j += 1


def flujo_ejemplos(documentos, rng, rafaga: tuple[int, int] | None = None):
    for doc in documentos:
        yield from ejemplos.documentos_de(doc, rng, rafaga)


def prefijo_comun(a: list[int], b: list[int]) -> int:
    n = 0
    for u, v in zip(a, b):
        if u != v:
            break
        n += 1
    return n


def tramos_de(tokenizador, lote, ids, bos: int) -> list[list[tuple[int, int]]]:
    fronteras = [ejemplos.fronteras(e) for e in lote]
    previos = iter(tokenizador.encode([f for fs in fronteras for f in fs], prepend=bos, num_threads=1))
    salida = []
    for t, fs in zip(ids, fronteras):
        cortes = [prefijo_comun(t, next(previos)) for _ in fs]
        inicios = [0] + cortes[1::2]
        salida.append([(a, max(a, b)) for a, b in zip(inicios, cortes[0::2])])
    return salida


def empaquetar(tokenizador, flujo, capacidad: int, ce: str = "todo"):
    if ce not in CE_MODOS:
        raise ValueError(f"QF2_RECOLOCADO_CE={ce!r} fuera de {CE_MODOS}")
    bos = tokenizador.get_bos_token_id()
    buffer: list[tuple[list[int], str, list[tuple[int, int]]]] = []
    while True:
        fila: list[int] = []
        tipos: list[str] = []
        ignorar = torch.zeros(capacidad - 1, dtype=torch.bool) if ce == "continuacion" else None
        while len(fila) < capacidad:
            while len(buffer) < BUFFER_MIN:
                lote = [next(flujo) for _ in range(TOKENIZAR_DE_A)]
                ids = tokenizador.encode([e["texto"] for e in lote], prepend=bos, num_threads=1)
                tramos = tramos_de(tokenizador, lote, ids, bos) if ce == "continuacion" else [[]] * len(lote)
                buffer.extend((t, e["tipo"], c) for t, e, c in zip(ids, lote, tramos))
            resto = capacidad - len(fila)
            caben = [k for k, (t, _, _) in enumerate(buffer) if len(t) <= resto]
            k = max(caben, key=lambda q: len(buffer[q][0])) if caben else min(range(len(buffer)), key=lambda q: len(buffer[q][0]))
            t, tipo, tramos = buffer.pop(k)
            if ignorar is not None:
                for ini, fin in tramos:
                    ignorar[max(0, len(fila) + ini - 1):max(0, min(len(fila) + fin - 1, capacidad - 1))] = True
            fila.extend(t[:resto])
            tipos.append(tipo)
        yield torch.tensor(fila, dtype=torch.long), tipos, ignorar


class Fuente:
    def __init__(self, tokenizador, documentos, semilla: int, rank: int, capacidad: int, ce: str = "todo", profundidad: int = 4,
                 rafaga: tuple[int, int] | None = None, pares: tuple | None = None):
        self.cola: queue.Queue = queue.Queue(maxsize=profundidad)
        self.error: BaseException | None = None
        self.cpu_s = 0.0
        self.filas = 0
        rng = random.Random(f"recolocado-ejemplos:{semilla}:{rank}")
        self.hilo = threading.Thread(target=self._correr, args=(tokenizador, documentos, rng, capacidad, ce, rafaga, pares), name="qf2_recolocado", daemon=True)
        self.hilo.start()

    def _correr(self, tokenizador, documentos, rng, capacidad, ce, rafaga, pares=None) -> None:
        try:
            nice.ceder("recolocado")
            t0 = time.thread_time()
            for fila in empaquetar(tokenizador, pares_m.flujo(documentos, rng, *pares) if pares else flujo_ejemplos(documentos, rng, rafaga), capacidad, ce):
                self.cpu_s = time.thread_time() - t0
                self.filas += 1
                self.cola.put(fila)
        except BaseException as e:
            self.error = e
            self.cola.put(None)

    def siguiente(self):
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            raise RuntimeError(f"qf2_recolocado: sin fila en {ESPERA_S:.0f} s") from self.error
        if item is None:
            raise RuntimeError("qf2_recolocado: el hilo de ejemplos terminó") from self.error
        return item


def envolver(base, tokenizador, rank: int, world: int, semilla: int, fraccion: float, documentos=None, registro=None, ce: str = "todo",
             rafaga: tuple[int, int] | None = None, pares: tuple | None = None, ocupadas=None):
    fuente = None
    acumulado = 0.0
    for paso, (x, y, estado) in enumerate(base):
        if fuente is None:
            docs = documentos if documentos is not None else documentos_de_shards(rank, world, semilla)
            fuente = Fuente(tokenizador, docs, semilla, rank, x.shape[1] + 1, ce, rafaga=rafaga, pares=pares)
        acumulado += fraccion * x.shape[0]
        libres = range(x.shape[0]) if ocupadas is None else ocupadas.libres(paso, x.shape[0])
        n = min(int(acumulado + 1e-9), len(libres))
        acumulado -= n
        if n:
            for r in random.Random(f"recolocado-filas:{semilla}:{rank}:{paso}").sample(libres, n):
                fila, tipos, ignorar = fuente.siguiente()
                x[r].copy_(fila[:-1])
                y[r].copy_(fila[1:])
                if ignorar is not None:
                    y[r].masked_fill_(ignorar, -1)
                if registro is not None:
                    registro.append((paso, r, tipos))
        yield x, y, estado


def cargador(base, tokenizador, rank: int, world: int, semilla: int, fraccion: float, ce: str = "todo", rafaga: str = "", formato: str = "ejemplos",
             pares_tramo: str = "8:40", pares_etiquetas: str = "Input/Output", **extra):
    if ce not in CE_MODOS:
        raise ValueError(f"QF2_RECOLOCADO_CE={ce!r} fuera de {CE_MODOS}")
    tramos_rafaga = leer_rafaga(rafaga)
    pares = pares_m.configurar(formato, ce, tramos_rafaga, pares_tramo, pares_etiquetas)
    if fraccion <= 0:
        return base
    return envolver(base, tokenizador, rank, world, semilla, fraccion, ce=ce, rafaga=tramos_rafaga, pares=pares, **extra)

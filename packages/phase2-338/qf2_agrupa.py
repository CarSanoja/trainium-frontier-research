"""Cadenas literales con BOS por secuencia de T; sustitución por acumulador de filas.

Vecinos TF-IDF o control azar entre longitudes cercanas. El último documento puede
recortarse. El flujo histórico se conserva con AGRUPA_FUENTE vacío; el nuevo reparto
por rowgroups sólo se activa explícitamente. No se promete disjunción semántica ni
idénticos multiconjuntos servidos entre modos.
"""
from __future__ import annotations

import math
import queue
import random
import threading
import time

import numpy as np
import torch

import qf2_hilos_nice as nice
import qf2_recolocado as rec
import qf2_agrupa_fuente as fuente

BITS, PREFIJO, CERCANOS, REFRESCO, TOKENIZAR_DE_A, PROFUNDIDAD, TRAZA_CADA, ESPERA_S = 12, 1024, 32, 64, 256, 16, 1024, 120.0
DIMENSIONES, MEZCLA, MODOS = 1 << BITS, np.uint64(0x9E3779B1), ("relacionados", "azar")


def leer(texto: str) -> tuple[float, int, str] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    try:
        fraccion, pool = float(partes[0]), int(partes[1])
    except (ValueError, IndexError):
        fraccion, pool = math.nan, 0
    if len(partes) != 3 or not 0 < fraccion <= 1 or pool < 2 * CERCANOS or partes[2] not in MODOS:
        raise ValueError(f"QF2_AGRUPA={texto!r}: se espera 'f:pool:modo' con 0 < f <= 1, pool >= {2 * CERCANOS} y modo en {MODOS}")
    return fraccion, pool, partes[2]


def documentos_de_rango(rank: int, world: int, semilla: int):
    if fuente.CONFIG is not None:
        yield from fuente.documentos_de_rango(rank, world)
        return
    import prepare
    import qf2_inicio
    rutas = prepare.list_shard_paths(qf2_inicio.shards_banco() or prepare.PUBLIC_TRAIN_SHARDS, require_all=False)
    turnos, turno = len(range(rank % len(rutas), world, len(rutas))), rank // len(rutas)
    for i, doc in enumerate(rec.documentos_de_shards(rank, world, semilla, rutas)):
        if i % turnos == turno:
            yield doc


class Pool:
    def __init__(self, n: int, documentos, tokenizador):
        self.n, self.documentos, self.tok, self.bos = n, documentos, tokenizador, tokenizador.get_bos_token_id()
        self.tf, self.df, self.idf = np.zeros((DIMENSIONES, n), np.float32), np.zeros(DIMENSIONES, np.int64), np.zeros(DIMENSIONES, np.float32)
        self.normas, self.largos = np.zeros(n, np.float32), np.zeros(n, np.int64)
        self.cub, self.peso, self.tokens, self.indice, self.leidos, self.espera = [None] * n, [None] * n, [None] * n, [-1] * n, 0, []
        for s in range(n):
            self.poner(s)
        self.refrescar()

    def leer_doc(self) -> tuple[int, np.ndarray]:
        if not self.espera:
            lote = [next(self.documentos) for _ in range(TOKENIZAR_DE_A)]
            self.espera = [np.asarray(t, dtype=np.int32) for t in self.tok.encode(lote, prepend=self.bos, num_threads=1)][::-1]
        self.leidos += 1
        return self.leidos - 1, self.espera.pop()

    def norma(self, s: int) -> float:
        return math.sqrt(float(np.square(self.peso[s] * self.idf[self.cub[s]]).sum()))

    def poner(self, s: int) -> None:
        if self.cub[s] is not None:
            self.tf[self.cub[s], s] = 0.0
            self.df[self.cub[s]] -= 1
        self.indice[s], t = self.leer_doc()
        u, c = np.unique(((t[1:1 + PREFIJO].astype(np.uint64) * MEZCLA) & np.uint64(0xFFFFFFFF)) >> np.uint64(32 - BITS), return_counts=True)
        u = u.astype(np.int64)
        self.cub[s], self.peso[s], self.tokens[s], self.largos[s] = u, (1.0 + np.log(c)).astype(np.float32), t, len(t)
        self.tf[u, s] = self.peso[s]
        self.df[u] += 1
        self.normas[s] = self.norma(s)

    def refrescar(self) -> None:
        self.idf = np.log((1.0 + self.n) / (1.0 + self.df)).astype(np.float32)
        for s in range(self.n):
            self.normas[s] = self.norma(s)

    def parecidos(self, s: int) -> np.ndarray:
        u = self.cub[s]
        sim = np.einsum("i,ij->j", self.peso[s] * np.square(self.idf[u]), self.tf[u])
        sim /= np.maximum(self.normas, 1e-12) * max(float(self.normas[s]), 1e-12)
        sim[s] = -np.inf
        return sim

    def al_azar(self, rng: random.Random, s: int, v: int) -> int:
        if v < 0:
            v = rng.randrange(self.n - 1)
            return v + (v >= s)
        d = np.abs(self.largos - self.largos[v])
        d[[s, v]] = np.iinfo(np.int64).max
        return int(np.argpartition(d, CERCANOS)[rng.randrange(CERCANOS)])

    def cadena(self, rng: random.Random, capacidad: int, modo: str) -> tuple[torch.Tensor, list, list]:
        s, partes, largo, sims = rng.randrange(self.n), [], 0, []
        while True:
            partes.append((self.indice[s], self.tokens[s]))
            largo += len(self.tokens[s])
            if largo >= capacidad:
                break
            sim = self.parecidos(s)
            v = int(np.argmax(sim)) if self.normas[s] > 0 else -1
            if modo == "azar" or v < 0:
                v = self.al_azar(rng, s, v)
            sims.append(float(sim[v]))
            self.poner(s)
            s = v
        self.poner(s)
        inicios = np.cumsum([0] + [len(t) for _, t in partes[:-1]])
        meta = [(i, min(len(t), capacidad - int(a)), len(t)) for (i, t), a in zip(partes, inicios)]
        return torch.from_numpy(np.concatenate([t for _, t in partes])[:capacidad].astype(np.int64)), meta, sims


class Fuente:
    def __init__(self, tokenizador, documentos, semilla: int, rank: int, ancho: int, seq_len: int, pool: int, modo: str):
        if seq_len < 1 or ancho <= seq_len or (ancho - 1) % seq_len:
            raise ValueError(f"qf2_agrupa: el ancho del cargador menos uno ({ancho - 1}) no es un múltiplo de T = --seq-len = {seq_len}")
        self.cola, self.error = queue.Queue(maxsize=PROFUNDIDAD), None
        self.cpu_s, self.arranque_s, self.suma_sim, self.filas, self.docs, self.pares = 0.0, 0.0, 0.0, 0, 0, 0
        self.config = (ancho, seq_len, (ancho - 1) // seq_len, pool, modo)
        rng = random.Random(f"agrupa:{semilla}:{rank}")
        self.hilo = threading.Thread(target=self._correr, args=(tokenizador, documentos, rng, rank), name="qf2_agrupa", daemon=True)
        self.hilo.start()

    def traza(self) -> str:
        return ("qf2_agrupa: ancho={} T={} cadenas={} pool={} modo={} ".format(*self.config) + f"dimensiones={DIMENSIONES} rango 0: {self.filas} "
                f"filas, pool lleno en {self.arranque_s:.1f} s de cpu, {self.filas / max(self.cpu_s - self.arranque_s, 1e-9):.1f} filas/s de cpu, "
                f"{self.docs / max(self.filas, 1):.2f} documentos/fila, coseno medio entre consecutivos {self.suma_sim / max(self.pares, 1):.4f}")

    def _correr(self, tokenizador, documentos, rng, rank) -> None:
        _, seq_len, k, pool, modo = self.config
        try:
            nice.ceder("agrupa")
            t0 = time.thread_time()
            p = Pool(pool, documentos, tokenizador)
            self.arranque_s = time.thread_time() - t0
            while True:
                cadenas = [p.cadena(rng, seq_len + (j == k - 1), modo) for j in range(k)]
                meta, sims = [m for _, m, _ in cadenas], [x for _, _, c in cadenas for x in c]
                self.filas, self.docs, self.pares = self.filas + 1, self.docs + sum(map(len, meta)), self.pares + len(sims)
                self.suma_sim += sum(sims)
                if self.filas % REFRESCO == 0:
                    p.refrescar()
                self.cpu_s = time.thread_time() - t0
                if rank == 0 and (self.filas == 1 or self.filas % TRAZA_CADA == 0):
                    print(self.traza(), flush=True)
                self.cola.put((torch.cat([f for f, _, _ in cadenas]), meta))
        except BaseException as e:
            self.error = e
            self.cola.put(None)

    def siguiente(self):
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            item = None
        if item is None:
            raise RuntimeError(f"qf2_agrupa: el hilo de filas terminó o no dio fila en {ESPERA_S:.0f} s") from self.error
        return item


def envolver(base, tokenizador, rank: int, world: int, semilla: int, seq_len: int, fraccion: float, pool: int, modo: str, documentos=None, registro=None):
    fuente, acumulado = None, 0.0
    for paso, (x, y, estado) in enumerate(base):
        if fuente is None:
            docs = documentos if documentos is not None else documentos_de_rango(rank, world, semilla)
            fuente = Fuente(tokenizador, docs, semilla, rank, x.shape[1] + 1, seq_len, pool, modo)
        acumulado += fraccion * x.shape[0]
        n = int(acumulado + 1e-9)
        acumulado -= n
        if n:
            for r in random.Random(f"agrupa-filas:{semilla}:{rank}:{paso}").sample(range(x.shape[0]), n):
                fila, meta = fuente.siguiente()
                x[r].copy_(fila[:-1])
                y[r].copy_(fila[1:])
                if registro is not None:
                    registro.append((paso, r, meta))
        yield x, y, estado


def cargador(base, tokenizador, rank: int, world: int, semilla: int, seq_len: int, config: tuple[float, int, str] | None):
    return base if config is None else envolver(base, tokenizador, rank, world, semilla, seq_len, *config)

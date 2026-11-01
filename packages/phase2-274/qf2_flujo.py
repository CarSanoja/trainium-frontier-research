from __future__ import annotations

import random
from collections import deque
from fractions import Fraction

import torch

import qf2_lotes
from prepare import _document_batches, get_dist_info

FRAC = Fraction(0)
LOTES_MEDIDA = 5
BUFFER_MIN = 512


def configurar(valor: str, cola_activa: bool) -> None:
    global FRAC
    texto = valor.strip() or "0"
    FRAC = Fraction(texto)
    if not 0 <= FRAC <= 1:
        raise ValueError(f"QF2_FLUJO_FRAC must be in [0, 1], got {texto}")
    if FRAC > 0 and cola_activa:
        raise ValueError("QF2_FLUJO_FRAC is incompatible with QF2_CARGADOR_LARGO, QF2_SEQ_MAX, QF2_COLA_FRAC "
                         "and QF2_CALIENTA_LARGO")


def activo() -> bool:
    return FRAC > 0


def filas_flujo_hasta(n: int, p: Fraction | None = None) -> int:
    p = FRAC if p is None else p
    return (n * p.numerator) // p.denominator


def es_fila_flujo(n: int, p: Fraction | None = None) -> bool:
    return filas_flujo_hasta(n + 1, p) > filas_flujo_hasta(n, p)


def fase(semilla: int, rango: int, seq_len: int) -> int:
    return random.Random(semilla * 1_000_003 + rango).randrange(seq_len)


class Documentos:
    def __init__(self, tokenizer, rango: int, mundo: int, hilos: int, lote_tok: int, split: str = "train", lotes=None):
        self.lotes = _document_batches(split, rango, mundo, lote_tok) if lotes is None else lotes
        self.tokenizer = tokenizer
        self.bos = tokenizer.get_bos_token_id()
        self.hilos = hilos
        self.cola: deque[list[int]] = deque()
        self.epoch = 0

    def tomar(self, n: int) -> list[list[int]]:
        if not self.cola:
            docs, self.epoch = next(self.lotes)
            self.cola.extend(self.tokenizer.encode(docs, prepend=self.bos, num_threads=self.hilos))
        return [self.cola.popleft() for _ in range(min(n, len(self.cola)))]


class Flujo:
    def __init__(self, docs: Documentos, seq_len: int, fase_inicial: int):
        self.docs = docs
        self.seq_len = seq_len
        self.flujo: list[int] = []
        self.llenar()
        del self.flujo[:fase_inicial]

    def llenar(self) -> None:
        while len(self.flujo) < self.seq_len + 1:
            for doc in self.docs.tomar(1):
                self.flujo.extend(doc)

    def fila(self) -> list[int]:
        self.llenar()
        fila = self.flujo[:self.seq_len + 1]
        del self.flujo[:self.seq_len]
        return fila


class MejorAjuste:
    def __init__(self, docs: Documentos, seq_len: int, lote_tok: int):
        self.docs = docs
        self.capacidad = seq_len + 1
        self.lote_tok = lote_tok
        self.buffer: list[list[int]] = []

    def fila(self, destino: torch.Tensor) -> None:
        pos = 0
        while pos < self.capacidad:
            while len(self.buffer) < BUFFER_MIN:
                self.buffer.extend(self.docs.tomar(self.lote_tok))
            restante = self.capacidad - pos
            mejor, largo = -1, 0
            for i, doc in enumerate(self.buffer):
                n = len(doc)
                if n <= restante and n > largo:
                    mejor, largo = i, n
            if mejor >= 0:
                doc = self.buffer.pop(mejor)
                destino[pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                pos += len(doc)
            else:
                corto = min(range(len(self.buffer)), key=lambda i: len(self.buffer[i]))
                doc = self.buffer.pop(corto)
                destino[pos:pos + restante] = torch.tensor(doc[:restante], dtype=torch.long)
                pos += restante


def descripcion(semilla: int, seq_len: int) -> str:
    _ddp, rango, _local, _mundo = get_dist_info()
    return f"qf2_flujo: p={FRAC} ({float(FRAC):.6f}) seed={semilla} phase_rank{rango}={fase(semilla, rango, seq_len)}"


class Cargador:
    def __init__(self, tokenizer, batch_size: int, seq_len: int, device, semilla: int, avisar=print,
                 tokenizer_threads: int = 4, tokenizer_batch_size: int = 128, split: str = "train",
                 p: Fraction | None = None, fase_inicial: int | None = None, reanudable: bool = False):
        self.p = FRAC if p is None else Fraction(p)
        _ddp, rango, _local, mundo = get_dist_info()
        self.origen = (split, rango, mundo, tokenizer_batch_size)
        lotes = qf2_lotes.Lotes(*self.origen) if reanudable else None
        self.docs = Documentos(tokenizer, rango, mundo, tokenizer_threads, tokenizer_batch_size, split, lotes)
        self.fase0 = fase(semilla, rango, seq_len) if fase_inicial is None else fase_inicial
        self.flujo = Flujo(self.docs, seq_len, self.fase0) if self.p > 0 else None
        self.empaque = MejorAjuste(self.docs, seq_len, tokenizer_batch_size)
        self.bos = self.docs.bos
        self.batch_size = batch_size
        self.avisar = avisar
        dev = torch.device(device)
        self.pin = dev.type == "cuda"
        self.filas = torch.empty((batch_size, seq_len + 1), dtype=torch.long)
        self.cpu_x = torch.empty((batch_size, seq_len), dtype=torch.int32, pin_memory=self.pin)
        self.cpu_y = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=self.pin)
        self.x = torch.empty((batch_size, seq_len), dtype=torch.int32, device=dev)
        self.y = torch.empty((batch_size, seq_len), dtype=torch.long, device=dev)
        self.n = 0
        self.lotes = 0
        self.cuenta_flujo = 0
        self.cuenta_bos = 0
        self.de_flujo: list[int] = []

    def __iter__(self):
        return self

    def close(self) -> None:
        pass

    def actual(self):
        return self.x, self.y, {"epoch": self.docs.epoch, "stream_rows": tuple(self.de_flujo)}

    def __next__(self):
        p, filas, batch_size = self.p, self.filas, self.batch_size
        de_flujo = []
        for i in range(batch_size):
            if self.flujo is not None and es_fila_flujo(self.n, p):
                filas[i] = torch.tensor(self.flujo.fila(), dtype=torch.long)
                de_flujo.append(i)
            else:
                self.empaque.fila(filas[i])
            self.n += 1
        self.lotes += 1
        if self.lotes <= LOTES_MEDIDA:
            self.cuenta_flujo += len(de_flujo)
            self.cuenta_bos += int((filas[:, 0] == self.bos).sum())
            if self.lotes == LOTES_MEDIDA:
                total = LOTES_MEDIDA * batch_size
                self.avisar(f"qf2_flujo: p={float(p):.6f} rows_per_batch={batch_size} stream_rows_per_batch="
                            f"{self.cuenta_flujo / LOTES_MEDIDA:.3f} stream_frac={self.cuenta_flujo / total:.6f} "
                            f"bos_pos0_frac={self.cuenta_bos / total:.6f} batches={LOTES_MEDIDA} phase={self.fase0}")
        self.de_flujo = de_flujo
        self.cpu_x.copy_(filas[:, :-1].to(dtype=torch.int32))
        self.cpu_y.copy_(filas[:, 1:])
        self.x.copy_(self.cpu_x, non_blocking=self.pin)
        self.y.copy_(self.cpu_y, non_blocking=self.pin)
        return self.actual()

    def estado(self) -> dict:
        if not isinstance(self.docs.lotes, qf2_lotes.Lotes):
            raise RuntimeError("qf2_flujo: the loader was not built resumable")
        return {"p": str(self.p), "forma": list(self.cpu_x.shape), "lotes": self.docs.lotes.posicion(),
                "epoch": self.docs.epoch, "cola": qf2_lotes.empaquetar(self.docs.cola),
                "buffer": qf2_lotes.empaquetar(self.empaque.buffer),
                "flujo": qf2_lotes.empaquetar([self.flujo.flujo]) if self.flujo is not None else None,
                "contadores": [self.n, self.lotes, self.cuenta_flujo, self.cuenta_bos], "fase0": self.fase0,
                "de_flujo": list(self.de_flujo), "x": self.cpu_x.clone(), "y": self.cpu_y.clone()}

    def restaurar(self, e: dict):
        if e["p"] != str(self.p) or list(e["forma"]) != list(self.cpu_x.shape) or (e["flujo"] is None) != (self.flujo is None):
            raise ValueError(f"qf2_flujo: saved loader p={e['p']} rows={e['forma']} does not match p={self.p} "
                             f"rows={list(self.cpu_x.shape)}")
        self.docs.lotes = qf2_lotes.Lotes(*self.origen, posicion=e["lotes"])
        self.docs.epoch = e["epoch"]
        self.docs.cola = deque(qf2_lotes.desempaquetar(e["cola"]))
        self.empaque.buffer = qf2_lotes.desempaquetar(e["buffer"])
        if self.flujo is not None:
            self.flujo.flujo = qf2_lotes.desempaquetar(e["flujo"])[0]
        self.n, self.lotes, self.cuenta_flujo, self.cuenta_bos = e["contadores"]
        self.fase0 = e["fase0"]
        self.de_flujo = list(e["de_flujo"])
        self.cpu_x.copy_(e["x"])
        self.cpu_y.copy_(e["y"])
        self.x.copy_(self.cpu_x, non_blocking=self.pin)
        self.y.copy_(self.cpu_y, non_blocking=self.pin)
        return self.actual()


def make_flujo_dataloader(tokenizer, batch_size: int, seq_len: int, device, semilla: int, avisar=print,
                          tokenizer_threads: int = 4, tokenizer_batch_size: int = 128, split: str = "train",
                          p: Fraction | None = None, fase_inicial: int | None = None, reanudable: bool = False):
    return Cargador(tokenizer, batch_size, seq_len, device, semilla, avisar, tokenizer_threads, tokenizer_batch_size,
                    split, p, fase_inicial, reanudable)

"""QF2_PRECARGA: the training loader's best-fit packing and tokenization run in a thread while the device computes.

The kit calls next(train_loader) after synchronize(device), inside dt, so on every step the device waits for the CPU
packing of the next batch (and, every few steps, for a tokenizer refill); with 64 ranks the step pays the slowest one.
Here a daemon thread drives prepare.make_dataloader on the CPU (same generator, same order, so the batches are the
same), clones each yielded batch (the generator reuses its buffers) into a queue of PROFUNDIDAD, and the main thread
only copies the ready batch into two persistent device buffers, as the kit's loader does. No device work runs in the
thread. Phase 1: ARCH_PREFETCH (P62, ../trainium-frontier/experiments/log/20260908T032945Z.md) took data loading from
0.007-0.037 s per step to 0.000.
"""
from __future__ import annotations

import queue
import threading

import torch

import qf2_hilos_nice as nice

PROFUNDIDAD = 2
ESPERA_S = 120.0


class Precarga:
    def __init__(self, fabrica, device, profundidad: int = PROFUNDIDAD, memhost=None, forma_mem=None, corpus=None):
        self.device = torch.device(device)
        self.memhost = memhost
        self.forma_mem = forma_mem
        self.corpus = corpus
        self.e = self.c = self.k = None
        self.producidos = 0
        self.cola: queue.Queue = queue.Queue(maxsize=profundidad)
        self.parar = threading.Event()
        self.error: BaseException | None = None
        self.x = self.y = None
        self.servidos = 0
        self.esperas = 0
        self.hilo = threading.Thread(target=self._correr, args=(fabrica,), name="qf2_precarga", daemon=True)
        self.hilo.start()

    def _poner(self, item) -> bool:
        while not self.parar.is_set():
            try:
                self.cola.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _correr(self, fabrica) -> None:
        try:
            nice.ceder("precarga")
            for x, y, estado in fabrica():
                item = (x.clone(), y.clone(), dict(estado))
                if self.corpus is not None:
                    c, k = self.corpus(item[0] if self.forma_mem is None else item[0].reshape(self.forma_mem))
                    item = item + (c.reshape(x.shape), k.reshape(x.shape))
                if self.memhost is not None:
                    self.memhost.registrar(self.producidos, item[0] if self.forma_mem is None else item[0].reshape(self.forma_mem))
                    self.producidos += 1
                if not self._poner(item):
                    return
        except BaseException as e:
            self.error = e
        self._poner(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cola.empty():
            self.esperas += 1
        try:
            item = self.cola.get(timeout=ESPERA_S)
        except queue.Empty:
            raise RuntimeError(f"qf2_precarga: sin lote en {ESPERA_S:.0f} s (hilo vivo={self.hilo.is_alive()})") from self.error
        if item is None:
            raise RuntimeError("qf2_precarga: el hilo del cargador terminó") from self.error
        x, y, estado = item[:3]
        if self.corpus is not None:
            if self.c is None:
                self.c = torch.empty(item[3].shape, dtype=item[3].dtype, device=self.device)
                self.k = torch.empty(item[4].shape, dtype=item[4].dtype, device=self.device)
            self.c.copy_(item[3])
            self.k.copy_(item[4])
        if self.x is None:
            self.x = torch.empty(x.shape, dtype=x.dtype, device=self.device)
            self.y = torch.empty(y.shape, dtype=y.dtype, device=self.device)
        self.x.copy_(x)
        self.y.copy_(y)
        if self.memhost is not None:
            e = self.memhost.E(self.servidos)
            if self.e is None:
                self.e = torch.empty(e.shape, dtype=e.dtype, device=self.device)
            self.e.copy_(e)
        self.servidos += 1
        return self.x, self.y, estado

    def cerrar(self) -> None:
        self.parar.set()
        while True:
            try:
                self.cola.get_nowait()
            except queue.Empty:
                break
        self.hilo.join(timeout=10.0)

    def resumen(self) -> str:
        return f"qf2_precarga: servidos={self.servidos} esperas={self.esperas} profundidad={self.cola.maxsize}"

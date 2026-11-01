"""QF2_MEMCORPUS_ANTICIPADO: write the corpus tables into out/final.pt as soon as they stop changing, finish the file at the end.

Saving 5 x 2^34 tables (344 GB) took 644 s on the block 3 host (gp3 root volume, ~0.11 GB/s of writeback): past the
dirty-page threshold the kernel throttles the writer to disk speed. The tables are final once the builder stops, hours
before the end of the run, so rank 0 opens out/final.pt with torch's own zip writer, writes the table storages as the
first records from a background thread (the writer releases the GIL) and leaves the kernel the rest of the run to flush
them. At the end torch.serialization._save pickles the full payload into the same writer; the table storages, already
written under the keys the pickler gives them (tables first in model_state, so keys 0..n-1), are skipped and every other
record is written as torch.save would. The result is a standard torch zip: torch.load (with or without mmap) and
prepare.checkpoint_size_report read it like any other final.pt.
"""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

import torch
import torch.serialization as ts


class _Zip:
    def __init__(self, escritor: torch._C.PyTorchFileWriter, previos: dict):
        self.escritor = escritor
        self.previos = previos
        self.saltados = []

    def write_record(self, nombre, datos, n):
        if nombre in self.previos:
            ptr, nbytes = self.previos[nombre]
            if not isinstance(datos, torch.UntypedStorage) or datos.data_ptr() != ptr or int(n) != nbytes:
                raise RuntimeError(f"guardado anticipado: {nombre} no es la tabla escrita antes")
            self.saltados.append(nombre)
            return
        self.escritor.write_record(nombre, datos, n)

    def write_record_metadata(self, nombre, n):
        raise RuntimeError("guardado anticipado: skip_data no está soportado")


class Anticipado:
    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.escritor = None
        self.previos: dict = {}
        self.nombres: list = []
        self.hilo: threading.Thread | None = None
        self.error: BaseException | None = None
        self.segundos = 0.0

    @property
    def empezado(self) -> bool:
        return self.escritor is not None

    def empezar(self, tablas: dict) -> None:
        if self.empezado:
            raise RuntimeError("guardado anticipado ya empezado")
        self.escritor = torch._C.PyTorchFileWriter(str(self.ruta), True, ts._get_storage_alignment())
        self.nombres = list(tablas)
        trabajo = []
        for i, t in enumerate(tablas.values()):
            s = t.untyped_storage()
            if t.storage_offset() != 0 or s.nbytes() != t.numel() * t.element_size():
                raise RuntimeError("guardado anticipado: cada tabla debe ocupar su almacenamiento entero")
            self.previos[f"data/{i}"] = (s.data_ptr(), s.nbytes())
            trabajo.append((f"data/{i}", s))
        self.hilo = threading.Thread(target=self._escribir, args=(trabajo,), name="qf2_anticipado", daemon=True)
        self.hilo.start()

    def _escribir(self, trabajo) -> None:
        t0 = time.monotonic()
        try:
            for nombre, s in trabajo:
                self.escritor.write_record(nombre, s, s.nbytes())
        except BaseException as ex:
            self.error = ex
        self.segundos = time.monotonic() - t0

    def listo(self) -> bool:
        return self.empezado and self.hilo is not None and not self.hilo.is_alive()

    def terminar(self, payload: dict) -> list:
        if not self.empezado:
            raise RuntimeError("guardado anticipado sin empezar")
        self.hilo.join()
        if self.error is not None:
            raise RuntimeError(f"guardado anticipado: falló la escritura de las tablas: {self.error!r}")
        zip_ = _Zip(self.escritor, self.previos)
        ts._save(payload, zip_, pickle, ts.DEFAULT_PROTOCOL, False)
        if sorted(zip_.saltados) != sorted(self.previos):
            raise RuntimeError(f"guardado anticipado: claves escritas {sorted(self.previos)} y saltadas {sorted(zip_.saltados)} no coinciden")
        self.escritor.write_end_of_file()
        return zip_.saltados


    def ordenar(self, estado: dict) -> dict:
        faltan = [k for k in self.nombres if k not in estado]
        if faltan:
            raise RuntimeError(f"guardado anticipado: faltan las tablas {faltan} en el estado final")
        return {**{k: estado[k] for k in self.nombres}, **{k: v for k, v in estado.items() if k not in self.nombres}}

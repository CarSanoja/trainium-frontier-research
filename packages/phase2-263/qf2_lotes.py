"""Resumable document batches for QF2_PUNTOS_S / QF2_REANUDA (default OFF).

Lotes yields exactly what prepare._document_batches yields (same row groups, same slices, same epoch counter) and exposes
its position (epoch, row group, offset), so a loader restarts from a saved point without replaying the data it consumed.
empaquetar / desempaquetar turn lists of token lists into two int tensors for torch.save(weights_only) and back.
"""
from __future__ import annotations

import pyarrow.parquet as pq
import torch

from prepare import _row_groups_for_split


class Lotes:
    def __init__(self, split: str, inicio: int, paso: int, lote: int, posicion=(1, 0, 0)):
        self.grupos = _row_groups_for_split(split, inicio, paso)
        self.lote = lote
        self.epoch, self.grupo, self.desde = (int(v) for v in posicion)
        if not 0 <= self.grupo < len(self.grupos):
            raise ValueError(f"qf2_lotes: row group {self.grupo} out of range ({len(self.grupos)} groups)")
        self.filas = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.filas is None:
                ruta, rg = self.grupos[self.grupo]
                self.filas = pq.ParquetFile(ruta).read_row_group(rg).column("text").to_pylist()
            if self.desde < len(self.filas):
                docs = self.filas[self.desde:self.desde + self.lote]
                self.desde += self.lote
                return docs, self.epoch
            self.filas, self.desde, self.grupo = None, 0, self.grupo + 1
            if self.grupo == len(self.grupos):
                self.grupo, self.epoch = 0, self.epoch + 1

    def posicion(self) -> list[int]:
        return [self.epoch, self.grupo, self.desde]


def empaquetar(listas) -> dict:
    largos = torch.tensor([len(l) for l in listas], dtype=torch.int64)
    plano = torch.tensor([t for l in listas for t in l], dtype=torch.int64)
    return {"largos": largos, "plano": plano}


def desempaquetar(d: dict) -> list[list[int]]:
    plano, salida, i = d["plano"].tolist(), [], 0
    for n in d["largos"].tolist():
        salida.append(plano[i:i + n])
        i += n
    return salida

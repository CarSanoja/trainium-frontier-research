"""QF2_MEMHOST: epoch barriers between the ranks of one host, kept in a shared file.

Every rank owns one int64 slot, written only by it. wait() polls until all slots reach an epoch and raises after the
deadline, so a dead rank stops the others with a visible error instead of leaving them blocked. Nothing here uses
torch.distributed: the default process group on the instance is neuron and no gloo group runs next to it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

PLAZO_S = float(os.environ.get("QF2_MEMHOST_PLAZO_S", "300"))
SONDEO_S = 1e-4
SONDEO_MAX_S = 5e-3
INICIO_PROCESO = time.time() - 60.0


def mapear(ruta: Path, dtype: torch.dtype, n: int, crear: bool) -> torch.Tensor:
    if crear:
        with open(ruta, "wb") as fh:
            fh.truncate(n * torch.empty((), dtype=dtype).element_size())
    return torch.from_file(str(ruta), shared=True, size=n, dtype=dtype)


def marca_valida(ruta: Path, contenido: str | None) -> bool:
    try:
        return ruta.stat().st_mtime >= INICIO_PROCESO and (contenido is None or ruta.read_text() == contenido)
    except (OSError, ValueError):
        return False


def esperar_marca(ruta: Path, contenido: str | None = None, plazo: float = PLAZO_S) -> None:
    t0 = time.monotonic()
    while not marca_valida(ruta, contenido):
        if time.monotonic() - t0 > plazo:
            raise TimeoutError(f"qf2_memhost: sin {ruta} tras {plazo:.0f} s")
        time.sleep(0.01)


class Barrera:
    def __init__(self, ruta: Path, rango: int, mundo: int, crear: bool, plazo: float = PLAZO_S):
        self.rango, self.mundo, self.plazo = rango, mundo, plazo
        self.ind = mapear(ruta / "indicadores.i64", torch.int64, mundo, crear)
        if crear:
            self.ind.zero_()

    def subir(self, valor: int) -> None:
        self.ind[self.rango] = valor

    def esperar(self, valor: int, que: str = "") -> None:
        t0 = time.monotonic()
        while int(self.ind.min()) < valor:
            if time.monotonic() - t0 > self.plazo:
                raise TimeoutError(f"qf2_memhost: barrera {que} epoca {valor} tras {self.plazo:.0f} s, indicadores={self.ind.tolist()}")
            time.sleep(SONDEO_S if time.monotonic() - t0 < 2e-3 else SONDEO_MAX_S)

    def pasar(self, valor: int, que: str = "") -> None:
        self.subir(valor)
        self.esperar(valor, que)

"""QF2_MEMHOST: the master table, one fp32 file mapped by every rank of the host.

q [R, W] fp32 masters (rows start at zero: the files are sparse and pages are paid on first touch), acc [R] fp32 second
moment per row and cnt [R] int16 touches per row. For the Adam arm m and v [R, W] are added. Rows are owned in contiguous
ranges of ceil(R / world), so each row has one writer per step. leer() returns the rows as evaluation will read them
(quantised to int8 with an fp16 scale); signs are applied by the caller.

Each run lives in its own directory under the base path, named by MASTER_PORT and the launcher pid shared by its ranks.
Rank 0 removes directories whose owner died, refuses a directory whose owner is alive, and writes the run id into the
ready mark that the other ranks wait for, so a stale mark or a neighbour run never gets mapped.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import qf2_memhost_cfg as cfgm
import qf2_memhost_cuantiza as cuant
import qf2_memhost_sincro as sincro

CNT_TOPE = 30000


def id_corrida() -> str:
    return f"{os.environ.get('MASTER_PORT', '0')}-{os.getppid()}"


def borrar(d: Path) -> None:
    if not d.is_dir():
        return
    for f in d.iterdir():
        if f.is_dir():
            borrar(f)
        else:
            f.unlink(missing_ok=True)
    try:
        d.rmdir()
    except OSError:
        pass


def vivo(d: Path) -> bool:
    try:
        os.kill(int((d / "duenio").read_text()), 0)
    except (OSError, ValueError):
        return False
    return True


def preparar(base: Path, ruta: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for d in base.iterdir():
        if d.is_dir() and d != ruta and not vivo(d):
            borrar(d)
    if ruta.exists() and vivo(ruta):
        raise RuntimeError(f"qf2_memhost: {ruta} pertenece a una corrida viva (pid {(ruta / 'duenio').read_text()})")
    borrar(ruta)
    ruta.mkdir()
    (ruta / "duenio").write_text(str(os.getpid()))


class Tabla:
    def __init__(self, cfg: cfgm.Cfg, ruta: str, rango: int, mundo: int, adam: bool = False):
        self.cfg, self.rango, self.mundo, self.adam = cfg, rango, mundo, adam
        self.id = id_corrida()
        self.base = Path(ruta)
        self.ruta = self.base / self.id
        self.r, self.w = cfg.filas, cfg.ancho
        self.ro = -(-self.r // mundo)
        crear = rango == 0
        marca = self.ruta / "listo"
        if crear:
            preparar(Path(ruta), self.ruta)
        else:
            sincro.esperar_marca(marca, self.id)
        self.q = sincro.mapear(self.ruta / "q.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
        self.acc = sincro.mapear(self.ruta / "acc.f32", torch.float32, self.r, crear)
        self.cnt = sincro.mapear(self.ruta / "cnt.i16", torch.int16, self.r, crear)
        self.m = self.v = None
        if adam:
            self.m = sincro.mapear(self.ruta / "m.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
            self.v = sincro.mapear(self.ruta / "v.f32", torch.float32, self.r * self.w, crear).view(self.r, self.w)
        self.barrera = sincro.Barrera(self.ruta, rango, mundo, crear)
        if crear:
            marca.write_text(self.id)

    @property
    def propias(self) -> tuple:
        return self.rango * self.ro, min(self.r, (self.rango + 1) * self.ro)

    def limites(self) -> torch.Tensor:
        return torch.tensor([min(self.r, o * self.ro) for o in range(self.mundo + 1)], dtype=torch.int64)

    def leer(self, ids: torch.Tensor) -> torch.Tensor:
        return cuant.ida_vuelta(self.q.index_select(0, ids))

    def leer_exacto(self, ids: torch.Tensor) -> torch.Tensor:
        return self.q.index_select(0, ids)

    def cerrar(self) -> None:
        borrar(self.ruta)
        try:
            self.base.rmdir()
        except OSError:
            pass

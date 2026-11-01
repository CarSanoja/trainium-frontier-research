"""Resumable training points (QF2_PUNTOS_S=N, QF2_REANUDA=1; both default OFF).

QF2_PUNTOS_S=N: every N seconds of training (the run's own budget clock, reduced with MAX across ranks inside the existing
control all_reduce, so every rank decides the same step) the run writes <out-dir>/puntos/p<step>/: rank<r>.pt per rank
(optimizer shard, loader position and pending batch, RNG, loop counters, guard/queue/averaging counters, own bf16-inexact
parameter ranges with QF2_COM_GATHER_BF16) and modelo.pt from rank 0. Each file goes to a temporary name, is fsynced and
renamed; after a barrier rank 0 writes HECHO the same way, so a point without HECHO never counts. Only the two newest
complete points are kept. The write time is taken off the budget clock.
QF2_REANUDA=1: every rank loads the newest complete point of the same world size and the loop continues from it with the
trained seconds already charged. The first TRAIN_STARTUP_STEPS_EXCLUDED resumed steps (recompilation) are left out of the
clock and of the recent step times, as the runner does with startup. Without a point the run starts from zero and says so.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import qf2_puntos_estado as est
from prepare import TRAIN_STARTUP_STEPS_EXCLUDED

LIBRES = ("QF2_PUNTOS_S", "QF2_REANUDA", "QF2_RELOJ_LOG", "QF2_RELOJ_TRAMOS")
ARGS = ("depth", "aspect_ratio", "head_dim", "seq_len", "device_batch_size", "total_batch_size", "num_steps",
        "target_param_data_ratio", "compile")


def borrar(d: Path) -> None:
    try:
        for f in d.iterdir():
            f.unlink(missing_ok=True)
        d.rmdir()
    except OSError:
        pass


def escribir(objeto, ruta: Path) -> None:
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "wb") as fh:
        if isinstance(objeto, (bytes, bytearray)):
            fh.write(objeto)
        else:
            torch.save(objeto, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, ruta)


def configuracion(args) -> dict:
    banderas = {k: v for k, v in os.environ.items() if k.startswith("QF2_") and k not in LIBRES}
    return {"env": dict(sorted(banderas.items())), "args": {k: getattr(args, k) for k in ARGS}}


def distribuido() -> bool:
    return dist.is_available() and dist.is_initialized()


class Puntos:
    def __init__(self, cada: str, reanuda: bool, out_dir: str, rank: int, mundo: int, args, avisar=print):
        self.cada_ms = round(float(cada) * 1000) if cada.strip() not in ("", "0") else 0
        if self.cada_ms < 0:
            raise ValueError(f"QF2_PUNTOS_S must be >= 0, got {cada!r}")
        self.reanuda, self.rank, self.mundo, self.avisar = reanuda, rank, mundo, avisar
        self.dir = Path(out_dir) / "puntos"
        self.config = configuracion(args)
        self.max_train_seconds = args.max_train_seconds
        self.proximo_ms = self.cada_ms
        self.ultimo_ms = 0
        self.calentar = 0
        self.entrenado_s = 0.0

    def descripcion(self) -> str:
        return f"qf2_puntos: cada_s={self.cada_ms / 1000 if self.cada_ms else '-'} reanuda={int(self.reanuda)} dir={self.dir}"

    def completos(self) -> list:
        salida = []
        for d in self.dir.glob("p*"):
            try:
                meta = json.loads((d / "HECHO").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("mundo") == self.mundo:
                salida.append((int(meta["paso"]), d))
        return sorted(salida)

    def rotar(self) -> None:
        guardar = {d for _paso, d in self.completos()[-2:]}
        for d in self.dir.glob("p*"):
            if d not in guardar:
                borrar(d)

    def toca(self, control) -> bool:
        if not self.cada_ms:
            return False
        ms = int(control[2].item())
        if ms < self.ultimo_ms:
            self.proximo_ms = (ms // self.cada_ms + 1) * self.cada_ms
        self.ultimo_ms = ms
        return ms >= self.proximo_ms

    def calienta(self) -> bool:
        if self.calentar <= 0:
            return False
        self.calentar -= 1
        return True

    def guardar(self, paso: int, bucle: dict, modelo, opt, cargador, extras: dict, bf16: bool, device) -> float:
        inicio = time.monotonic()
        entrenado_ms = bucle["entrenado_ms"]
        d = self.dir / f"p{paso:09d}"
        d.mkdir(parents=True, exist_ok=True)
        propio = {"paso": paso, "mundo": self.mundo, "rango": self.rank, "bf16": bf16, "config": self.config,
                  "bucle": {**bucle, "entrenado_s": entrenado_ms / 1000.0}, "rng": est.estado_rng(device),
                  "opt": est.estado_opt(opt), "datos": cargador.estado(), "extras": est.estado_extras(extras),
                  "modelo": est.modelo_propio(modelo) if bf16 else None}
        escribir(propio, d / f"rank{self.rank}.pt")
        if self.rank == 0:
            promedio = extras.get("promedio")
            escribir({"modelo": est.modelo_base(modelo, bf16), "bf16": bf16,
                      "promedio": [s.detach().to("cpu", copy=True) for s in promedio.suma] if promedio else None}, d / "modelo.pt")
        if distribuido():
            dist.barrier()
        if self.rank == 0:
            escribir(json.dumps({"paso": paso, "mundo": self.mundo, "entrenado_s": entrenado_ms / 1000.0,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode(), d / "HECHO")
            self.rotar()
        self.proximo_ms = (entrenado_ms // self.cada_ms + 1) * self.cada_ms
        segundos = time.monotonic() - inicio
        mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 2**20 if self.rank == 0 else 0.0
        self.avisar(f"qf2_puntos: saved step={paso} trained_s={entrenado_ms / 1000:.1f} write_s={segundos:.2f} "
                    f"dir_MB_seen_by_rank0={mb:.1f} next_at_s={self.proximo_ms / 1000:.1f}")
        return segundos

    def reanudar(self, modelo, opt, cargador, extras: dict, bf16: bool, device) -> dict | None:
        if not self.reanuda:
            return None
        lista = self.completos()
        paso, d = lista[-1] if lista else (-1, None)
        if distribuido():
            t = torch.tensor([paso, -paso], dtype=torch.int64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if int(t[0].item()) != -int(t[1].item()):
                raise RuntimeError(f"qf2_puntos: ranks see different newest points (max {int(t[0])}, min {-int(t[1])})")
        if d is None:
            self.avisar(f"qf2_puntos: no complete point in {self.dir} for world={self.mundo}: starting from zero")
            return None
        propio = torch.load(d / f"rank{self.rank}.pt", map_location="cpu", weights_only=True)
        comun = torch.load(d / "modelo.pt", map_location="cpu", weights_only=True, mmap=True)
        self.comprobar(propio, comun, bf16)
        est.poner_modelo(modelo, comun["modelo"], propio["modelo"])
        est.poner_opt(opt, propio["opt"])
        est.poner_extras(extras, propio["extras"])
        if extras.get("promedio") is not None:
            est.poner_promedio(extras["promedio"], comun["promedio"])
        est.poner_rng(propio["rng"], device)
        lote = cargador.restaurar(propio["datos"])
        bucle = propio["bucle"]
        self.entrenado_s = bucle["entrenado_s"]
        if self.cada_ms:
            self.proximo_ms = (bucle["entrenado_ms"] // self.cada_ms + 1) * self.cada_ms
        self.ultimo_ms = bucle["entrenado_ms"]
        self.calentar = TRAIN_STARTUP_STEPS_EXCLUDED if bucle["step"] >= TRAIN_STARTUP_STEPS_EXCLUDED else 0
        if distribuido():
            dist.barrier()
        if self.rank == 0:
            self.rotar()
        self.avisar(f"qf2_puntos: resumed from {d} step={bucle['step']} trained_s={self.entrenado_s:.1f} "
                    f"tokens={bucle['total_train_tokens']} excluded_steps={self.calentar}")
        return {**bucle, "lote": lote}

    def comprobar(self, propio: dict, comun: dict, bf16: bool) -> None:
        if propio["mundo"] != self.mundo or propio["rango"] != self.rank or propio["bf16"] != bf16 or comun["bf16"] != bf16:
            raise ValueError(f"qf2_puntos: point world={propio['mundo']} rank={propio['rango']} bf16={propio['bf16']} "
                             f"does not match world={self.mundo} rank={self.rank} bf16={bf16}")
        guardada, ahora = propio["config"], self.config
        if guardada != ahora:
            raise ValueError(f"qf2_puntos: point was saved with another configuration: saved={guardada} now={ahora}")
        if self.max_train_seconds != propio["bucle"]["max_train_seconds"]:
            self.avisar(f"qf2_puntos: max_train_seconds {propio['bucle']['max_train_seconds']} -> {self.max_train_seconds}")

"""QF2_MEMHOST: the table thread of one rank.

Per step e the main thread posts the gradient of the recovered vectors. This thread aggregates it, exchanges it through
shared memory (barrier A), updates the rows it owns (barrier B) and then gathers the vectors of batch e + RETRASO + 1,
quantised and signed, for the data loader to copy to the device. With RETRASO = 1 the gather of batch k sees the table
after the updates up to k - 2. Batches 0..RETRASO see the zero table. A skipped step still crosses both barriers with an
empty gradient. Any failure is kept and raised by revisar() in the main thread.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict

import numpy as np
import torch

import qf2_hilos_aten as aten
import qf2_hilos_nice as nice
import qf2_memhost_cfg as cfgm
import qf2_memhost_hash as hashm
import qf2_memhost_intercambio as inter
import qf2_memhost_opt as optm
import qf2_memhost_tabla as tab
import qf2_memhost_traza as traza

ESPERA_S = 300.0


class Memhost:
    def __init__(self, cfg: cfgm.Cfg, ruta: str, rango: int, mundo: int, forma: tuple, dtype: torch.dtype, bos: int,
                 retraso: int = 1, modo: str = "rms", canon=None):
        self.cfg, self.rango, self.mundo, self.dtype, self.bos, self.canon = cfg, rango, mundo, dtype, bos, canon
        self.b, self.t = forma
        self.retraso, self.modo = retraso, modo
        self.tabla = tab.Tabla(cfg, ruta, rango, mundo, adam=modo == "adam")
        self.inter = inter.Intercambio(self.tabla.ruta, rango, mundo, self.b * self.t * cfg.s, cfg.ancho, rango == 0)
        self.limites = self.tabla.limites()
        self.cond = threading.Condition()
        self.entradas: dict = {}
        self.listos: dict = {}
        self.jobs: queue.Queue = queue.Queue()
        self.cierre = threading.Event()
        self.error: BaseException | None = None
        self.aplicadas = 0
        self.pasos_opt = 0
        self.vistas: dict = {}
        self.esperas = 0
        self.servidos = 0
        self.pasos = 0
        self.tiempos: dict = defaultdict(float)
        self.tabla.barrera.pasar(0, "arranque")
        self.hilo = threading.Thread(target=self._correr, name="qf2_memhost", daemon=True)
        self.hilo.start()

    def indices(self, x) -> tuple:
        ids, sig = hashm.indices(x.cpu().numpy() if hasattr(x, "cpu") else x, self.cfg, self.bos, self.canon)
        return torch.from_numpy(ids.reshape(-1)), torch.from_numpy(sig.reshape(-1).view(np.int64))

    def registrar(self, k: int, x) -> None:
        ids, sig = self.indices(x)
        with self.cond:
            self.entradas[k] = (ids, sig)
            self.cond.notify_all()

    def _entrada(self, k: int):
        with self.cond:
            t0 = time.monotonic()
            while k not in self.entradas:
                if self.cierre.is_set():
                    return None
                if time.monotonic() - t0 > ESPERA_S:
                    raise TimeoutError(f"qf2_memhost: sin ids del lote {k} en {ESPERA_S:.0f} s")
                self.cond.wait(0.5)
            return self.entradas[k]

    def _signos(self, sig: torch.Tensor):
        if not self.cfg.signos:
            return None
        return torch.from_numpy(hashm.signos(sig.numpy().view(np.uint64), self.cfg.ancho)).reshape(-1, self.cfg.ancho)

    def _reunir_lote(self, k: int) -> None:
        ent = self._entrada(k)
        if ent is None:
            return
        ids, sig = ent
        t0 = time.perf_counter()
        filas = self.tabla.leer(ids)
        sg = self._signos(sig)
        if sg is not None:
            filas = filas * sg
        e = filas.view(self.b, self.t, self.cfg.ancho_e).to(self.dtype)
        self.tiempos["gather"] += time.perf_counter() - t0
        with self.cond:
            self.listos[k] = e
            self.vistas[k] = self.aplicadas
            self.cond.notify_all()

    def _actualizar(self, e: int, grad, lr: float, lrm: float = 1.0) -> None:
        ent = self._entrada(e)
        if ent is None:
            raise RuntimeError(f"qf2_memhost: faltan los ids del lote {e} al actualizar")
        ids, sig = ent
        t0 = time.perf_counter()
        if grad is None:
            uid, g = torch.empty(0, dtype=torch.int64), torch.empty(0, self.cfg.ancho)
        else:
            gf = torch.nan_to_num(grad.float().reshape(-1, self.cfg.ancho))
            sg = self._signos(sig)
            if sg is not None:
                gf = gf * sg
            uid, g = inter.agregar(ids, gf / self.mundo)
        t1 = time.perf_counter()
        self.inter.publicar(uid, g, self.limites)
        t2 = time.perf_counter()
        self.tabla.barrera.pasar(2 * e + 1, "A")
        t3 = time.perf_counter()
        pid, pg = self.inter.reunir(self.rango)
        t4 = time.perf_counter()
        if grad is not None:
            self.pasos_opt += 1
            cabs = traza.activa(self.pasos_opt, self.rango, self.cfg, self.mundo) if self.modo == "rms" else ()
            est = traza.previa(self.tabla, pid, pg, self.cfg, cabs) if cabs else None
            optm.aplicar(self.tabla, self.modo, pid, pg, lr, self.pasos_opt, lrm)
            if est is not None:
                print("\n".join(traza.lineas(est, self.tabla, self.pasos_opt, lrm, optm.DECAY, self.rango)), flush=True)
        self.aplicadas += 1
        t5 = time.perf_counter()
        self.tabla.barrera.pasar(2 * e + 2, "B")
        t6 = time.perf_counter()
        for k, v in (("agregar", t1 - t0), ("publicar", t2 - t1), ("espera_A", t3 - t2), ("reunir", t4 - t3), ("optimizar", t5 - t4), ("espera_B", t6 - t5)):
            self.tiempos[k] += v
        with self.cond:
            self.entradas.pop(e, None)

    def _correr(self) -> None:
        try:
            nice.ceder("hilo de tablas")
            aten.fijar_hilo("hilo de tablas")
            for k in range(self.retraso + 1):
                self._reunir_lote(k)
            e = 0
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                self._actualizar(e, *job)
                self._reunir_lote(e + self.retraso + 1)
                e += 1
            self.pasos = e
        except BaseException as ex:
            self.error = ex
            with self.cond:
                self.cond.notify_all()

    def revisar(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"qf2_memhost: fallo del hilo de tabla: {self.error!r}") from self.error

    def anotar(self, clave: str, segundos: float) -> None:
        self.tiempos[clave] += segundos

    def postear(self, grad, lr: float, lrm: float = 1.0) -> None:
        self.revisar()
        self.jobs.put((grad, lr, lrm))

    def E(self, k: int) -> torch.Tensor:
        t0 = time.monotonic()
        with self.cond:
            if k not in self.listos:
                self.esperas += 1
            while k not in self.listos:
                self.revisar()
                if time.monotonic() - t0 > ESPERA_S:
                    raise TimeoutError(f"qf2_memhost: sin vectores del lote {k} en {ESPERA_S:.0f} s")
                self.cond.wait(0.5)
            self.servidos += 1
            return self.listos.pop(k)

    def fin(self) -> int:
        self.jobs.put(None)
        self.cierre.set()
        self.hilo.join(timeout=ESPERA_S)
        self.revisar()
        n = self.pasos
        self.tabla.barrera.pasar(2 * n + 3, "fin")
        return n

    def resumen(self) -> str:
        ms = " ".join(f"{k}={1000 * v / max(1, self.aplicadas if k != 'gather' else self.servidos):.1f}ms" for k, v in sorted(self.tiempos.items()))
        return (f"qf2_memhost: servidos={self.servidos} esperas={self.esperas} actualizaciones={self.aplicadas} retraso={self.retraso} "
                f"modo={self.modo} por_paso: {ms}")

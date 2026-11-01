"""QF2_MEMCORPUS: el constructor de la memoria del corpus, en el anfitrion y dentro del presupuesto de entreno.

El rango local 0 crea las tablas como ficheros compartidos (un uint64 o uint32 por cubo y orden) y, antes de iniciar el dispositivo,
bifurca un gestor sin hilos de Neuron. El gestor espera la marca de arranque, que el bucle escribe cuando empieza el reloj
oficial (paso TRAIN_STARTUP_STEPS_EXCLUDED), y entonces reparte los shards de entrenamiento que la red no lee (de 06499
hacia abajo, sin los primeros EXCLUIR) entre procesos con nice 10. Cada proceso tokeniza como el cargador (BOS delante de
cada documento) y actualiza las tablas. La marca de parada, que escribe el rango 0 antes de guardar el checkpoint, congela
la tabla: los procesos acaban el grupo de filas en curso y el gestor escribe 'parado'. Si el rango 0 muere, el gestor y sus
procesos mueren con el (PR_SET_PDEATHSIG en Linux, y el gestor comprueba a su padre); los ficheros de cada corrida se crean
de nuevo (inodo nuevo), asi que un proceso de una corrida anterior no puede escribir en las tablas de la siguiente.
"""
from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

import qf2_corpus_tabla as qt

ESPERA_S = 0.5
TOKENS_POR_SHARD = 5.4e7
G = {}


def ficheros(ruta: Path, ks, ancho: int = 64) -> dict:
    return {k: ruta / f"k{k}.u{ancho}" for k in ks}


def abrir(ruta: Path, cfg, crear: bool = False) -> qt.Tabla:
    arrays = {k: np.memmap(f, dtype=qt.TIPO[cfg.ancho], mode="w+" if crear else "r+", shape=(1 << cfg.bits,))
              for k, f in ficheros(ruta, cfg.ks, cfg.ancho).items()}
    return qt.Tabla.desde_arrays(cfg.ks, cfg.bits, arrays)


def shards_del_constructor(rutas: list, excluir: int) -> list:
    return sorted((p for p in rutas if int(p.stem.split("_")[-1]) >= excluir), key=lambda p: -int(p.stem.split("_")[-1]))


def morir_con_el_padre():
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)


def _iniciar(ruta, cfg, bos, tokenizador):
    morir_con_el_padre()
    os.nice(10)
    G.update(ruta=Path(ruta), bos=bos, tok=tokenizador, tabla=abrir(Path(ruta), cfg))


def _shard(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    ntok = pares = 0
    for g in range(pf.num_row_groups):
        if (G["ruta"] / "parar").exists():
            break
        textos = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        listas = G["tok"].encode(textos, prepend=G["bos"], num_threads=1)
        t = np.concatenate([np.asarray(x, dtype=np.uint16) for x in listas]) if listas else np.zeros(0, np.uint16)
        ntok += int(t.size)
        pares += G["tabla"].actualizar(t, G["bos"])
    return str(path), ntok, pares


def _escribir(ruta: Path, nombre: str, datos: dict):
    tmp = ruta / f".{nombre}.tmp"
    tmp.write_text(json.dumps(datos))
    tmp.replace(ruta / nombre)


def _gestor(ruta, cfg, shards, procesos, bos, tokenizador, padre):
    morir_con_el_padre()
    ruta = Path(ruta)
    os.nice(10)
    while not (ruta / "empezar").exists() and not (ruta / "parar").exists():
        if os.getppid() != padre:
            return
        time.sleep(ESPERA_S)
    t0, hechos, tokens, pares = time.time(), 0, 0, 0
    if not (ruta / "parar").exists():
        with mp.get_context("fork").Pool(procesos, initializer=_iniciar, initargs=(str(ruta), cfg, bos, tokenizador)) as pool:
            for _, ntok, np_ in pool.imap_unordered(_shard, shards):
                hechos, tokens, pares = hechos + 1, tokens + ntok, pares + np_
                _escribir(ruta, "progreso.json", {"shards": hechos, "de": len(shards), "tokens": tokens, "pares": pares,
                                                  "t": time.time() - t0})
                if (ruta / "parar").exists() or os.getppid() != padre:
                    break
            pool.terminate()
    _escribir(ruta, "parado", {"shards": hechos, "de": len(shards), "tokens": tokens, "pares": pares, "t": time.time() - t0,
                               "completo": hechos == len(shards)})


class Constructor:
    def __init__(self, cfg, ruta: Path, shards: list, procesos: int, bos: int, tokenizador):
        self.cfg, self.ruta, self.shards = cfg, Path(ruta), list(shards)
        self.ruta.mkdir(parents=True, exist_ok=True)
        for f in list(ficheros(self.ruta, cfg.ks, cfg.ancho).values()) + [self.ruta / n for n in ("creada", "empezar", "parar", "parado", "progreso.json")]:
            f.unlink(missing_ok=True)
        abrir(self.ruta, cfg, crear=True)
        _escribir(self.ruta, "creada", {"corrida": corrida()})
        atexit.register(self.parar, 5.0)
        self.gestor = mp.get_context("fork").Process(target=_gestor, args=(str(self.ruta), cfg, self.shards, procesos, bos, tokenizador, os.getpid()),
                                                     name="qf2_memcorpus", daemon=False)
        self.final = None
        self.gestor.start()

    def empezar(self):
        (self.ruta / "empezar").write_text(str(time.time()))

    def acabado(self) -> bool:
        return (self.ruta / "parado").exists()

    def progreso(self) -> dict:
        try:
            return json.loads((self.ruta / "progreso.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def parar(self, espera_s: float = 120.0) -> dict:
        if self.final is not None:
            return self.final
        try:
            (self.ruta / "parar").write_text(str(time.time()))
        except OSError:
            self.gestor.terminate()
            self.final = {"ruta_perdida": True}
            return self.final
        limite = time.time() + espera_s
        while not (self.ruta / "parado").exists() and time.time() < limite and self.gestor.is_alive():
            time.sleep(ESPERA_S)
        if self.gestor.is_alive() and not (self.ruta / "parado").exists():
            self.gestor.terminate()
        try:
            self.final = json.loads((self.ruta / "parado").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.final = {"parado_a_la_fuerza": True, **self.progreso()}
        self.gestor.join(timeout=10.0)
        return self.final

    def limpiar(self):
        self.parar(5.0)
        for f in list(ficheros(self.ruta, self.cfg.ks, self.cfg.ancho).values()) + [self.ruta / n for n in ("creada", "empezar", "parar", "parado", "progreso.json")]:
            f.unlink(missing_ok=True)
        try:
            self.ruta.rmdir()
        except OSError:
            pass

    def descripcion(self) -> str:
        return (f"qf2_memcorpus: bits={self.cfg.bits} ks={self.cfg.ks} shards={len(self.shards)} "
                f"(de {self.shards[0].name if self.shards else '-'} a {self.shards[-1].name if self.shards else '-'}) ruta={self.ruta}"
                + ("" if self.shards else " AVISO: SIN SHARDS, la memoria queda vacia"))


def corrida() -> str:
    return os.environ.get("TORCHELASTIC_RUN_ID", "") or str(os.getppid())


def esperar_creada(ruta: Path, espera_s: float = 600.0):
    limite = time.time() + espera_s
    while True:
        try:
            if json.loads((Path(ruta) / "creada").read_text()).get("corrida") == corrida():
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.time() > limite:
            raise RuntimeError(f"qf2_memcorpus: las tablas de esta corrida no aparecieron en {ruta} en {espera_s:.0f} s")
        time.sleep(ESPERA_S)

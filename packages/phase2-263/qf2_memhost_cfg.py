"""QF2_MEMHOST: configuration of the n-gram memory tables that live in the host RAM.

The configuration travels in the checkpoint as one string (GPTConfig.memhost) so that evaluation rebuilds exactly the
layout used in training: width W of one row, one (order, rows) entry per hash head, the injection sites, the sign trick
the tokenizer compression flag and the gate function of the gated sites (absent from the string when it is the default
sigmoid, so the specs written before the option keep their exact text). A head with F rows owns rows [base, base + F) of the global table; its hash modulus
is the largest prime below F and the row base + prime is the sentinel for positions without enough history.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

CABEZAS_BASE = "2x2:128000000,3x4:240000000"
SITIOS_BASE = "in,1,11"
ANCHO_BASE = 64
PUERTAS = ("sig", "firm", "rec")
VIAS = ("res", "v")
CLAVE_Q = "memhost.q.weight"
CLAVE_S = "memhost.s.weight"
CLAVE_CANON = "memhost.canon"


@dataclass(frozen=True)
class Cfg:
    ancho: int
    cabezas: tuple
    sitios: tuple
    signos: bool
    canon: bool
    puerta: str = "sig"
    via: str = "res"

    @property
    def s(self) -> int:
        return len(self.cabezas)

    @property
    def ancho_e(self) -> int:
        return self.s * self.ancho

    @property
    def filas(self) -> int:
        return sum(f for _o, f in self.cabezas)

    @property
    def bases(self) -> tuple:
        out, acc = [], 0
        for _o, f in self.cabezas:
            out.append(acc)
            acc += f
        return tuple(out)

    @property
    def primos(self) -> tuple:
        usados, out = set(), []
        for _o, f in self.cabezas:
            p = primo_bajo(f)
            while p in usados:
                p = primo_bajo(p)
            usados.add(p)
            out.append(p)
        return tuple(out)

    @property
    def gated(self) -> tuple:
        return tuple(int(s) for s in self.sitios if s != "in")

    @property
    def entrada(self) -> bool:
        return "in" in self.sitios


def es_primo(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@lru_cache(maxsize=None)
def primo_bajo(filas: int) -> int:
    n = filas - 1
    while n > 2 and not es_primo(n):
        n -= 1
    return n


def parse_cabezas(texto: str) -> tuple:
    out = []
    for grupo in texto.split(","):
        ob, filas = grupo.split(":")
        orden, n = ob.split("x")
        if int(orden) not in (2, 3):
            raise ValueError(f"orden {orden} no soportado (2 o 3)")
        out.extend((int(orden), int(filas)) for _ in range(int(n)))
    if not out or any(f < 8 for _o, f in out):
        raise ValueError(f"cabezas invalidas: {texto!r}")
    return tuple(out)


def desde_texto(spec: str) -> Cfg:
    d = dict(p.split("=", 1) for p in spec.split(";") if p)
    sitios = tuple(s for s in d.get("sitios", SITIOS_BASE).split(",") if s)
    if "in" not in sitios:
        raise ValueError(f"sitios {sitios} sin 'in': tabla y Linear_val empiezan a cero y ninguna recibe gradiente")
    puerta = d.get("puerta", "sig")
    if puerta not in PUERTAS:
        raise ValueError(f"puerta {puerta!r} fuera de {PUERTAS}")
    via = d.get("via", "res")
    if via not in VIAS:
        raise ValueError(f"via {via!r} fuera de {VIAS}")
    return Cfg(int(d.get("ancho", ANCHO_BASE)), parse_cabezas(d.get("cab", CABEZAS_BASE)), sitios,
               d.get("signos", "1") != "0", d.get("canon", "0") != "0", puerta, via)


def a_texto(cfg: Cfg) -> str:
    cab: dict = {}
    for o, f in cfg.cabezas:
        cab[(o, f)] = cab.get((o, f), 0) + 1
    grupos = ",".join(f"{o}x{n}:{f}" for (o, f), n in cab.items())
    puerta = "" if cfg.puerta == "sig" else f";puerta={cfg.puerta}"
    via = "" if cfg.via == "res" else f";via={cfg.via}"
    return f"ancho={cfg.ancho};cab={grupos};sitios={','.join(cfg.sitios)};signos={int(cfg.signos)};canon={int(cfg.canon)}{puerta}{via}"


def desde_banderas(activo: str, ancho: str, cabezas: str, sitios: str, signos: str, canon: str,
                   puerta: str = "sig", via: str = "res") -> str:
    if activo == "0":
        return ""
    return a_texto(desde_texto(f"ancho={ancho};cab={cabezas};sitios={sitios};signos={signos};canon={canon};puerta={puerta};via={via}"))

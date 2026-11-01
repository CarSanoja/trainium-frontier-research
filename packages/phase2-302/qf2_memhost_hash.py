"""QF2_MEMHOST: causal 64-bit n-gram hash, the only implementation (data loader, table thread and evaluation import it).

For every position the ids of the S heads depend only on the tokens up to that position. Validity follows
qf2_ngramas.indices: the bigram needs one earlier position and a current token that is not BOS; the trigram also needs
a second earlier position and a first earlier token that is not BOS. Invalid references map to the sentinel row of the
head. Arithmetic is uint64 with wrap-around in numpy; there is no int32 anywhere.
"""
from __future__ import annotations

import numpy as np

import qf2_memhost_cfg as cfgm

SEMILLA = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S33 = np.uint64(33)


def splitmix(x: int) -> int:
    x = (x + SEMILLA) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def multiplicadores(cabeza: int) -> tuple:
    return tuple(np.uint64(splitmix(1000 * (cabeza + 1) + j) | 1) for j in range(3))


def sal(cabeza: int) -> np.uint64:
    return np.uint64(splitmix(7777 + cabeza))


def mezcla64(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> S33)
    x = x * C1
    x = x ^ (x >> S33)
    x = x * C2
    return x ^ (x >> S33)


def desplazar(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a)
    out[:, k:] = a[:, :a.shape[1] - k]
    return out


def canon_identidad(vocab: int) -> np.ndarray:
    return np.arange(vocab, dtype=np.int64)


def indices(tokens, cfg: cfgm.Cfg, bos: int, canon=None) -> tuple:
    t = np.asarray(tokens).astype(np.int64)
    if t.ndim != 2:
        raise ValueError("tokens debe ser [B, T]")
    c = (np.asarray(canon).astype(np.int64)[t] if canon is not None else t).astype(np.uint64)
    pos = np.arange(t.shape[1])[None, :]
    vale2 = (t != bos) & (pos >= 1)
    vale3 = vale2 & (desplazar(t, 1) != bos) & (pos >= 2)
    c1, c2 = desplazar(c, 1), desplazar(c, 2)
    bases, primos = cfg.bases, cfg.primos
    ids = np.empty(t.shape + (cfg.s,), dtype=np.int64)
    sig = np.empty(t.shape + (cfg.s,), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for h, (orden, _f) in enumerate(cfg.cabezas):
            ma, mb, mc = multiplicadores(h)
            mix = (c * ma) ^ (c1 * mb)
            valido = vale2
            if orden == 3:
                mix = mix ^ (c2 * mc)
                valido = vale3
            mix = mezcla64(mix)
            fila = (mix % np.uint64(primos[h])).astype(np.int64) + bases[h]
            ids[..., h] = np.where(valido, fila, bases[h] + primos[h])
            sig[..., h] = np.where(valido, mezcla64(mix ^ sal(h)), np.uint64(0))
    return ids, sig


def signos(sig: np.ndarray, ancho: int) -> np.ndarray:
    bits = (sig[..., None] >> (np.arange(ancho, dtype=np.uint64) % np.uint64(64))) & np.uint64(1)
    return bits.astype(np.float32) * 2.0 - 1.0

"""Memoria de n-gramas largos del corpus: una tabla de cubos por orden k con el siguiente token mayoritario de cada contexto.

Cada cubo es un uint64: cuenta neta (bits 0-23), siguiente token (24-38) y huella del contexto (39-55, nunca 0); con ancho 32,
un uint32: cuenta (0-5, tope 63), siguiente (6-20) y huella (21-31, de 1 a 2047), el doble de entradas por byte. El cubo y la
huella salen del hash del contexto de k tokens (sin BOS dentro, siguiente distinto de BOS). La actualizacion es una mayoria
por cubo: la masa del par guardado suma, la de cualquier otro par resta, y si la cuenta no queda positiva el par mas
frecuente del trozo ocupa el cubo. Los contextos vistos una vez se desalojan con la primera colision; los repetidos con un
siguiente dominante se quedan. La consulta devuelve, por posicion, el candidato del k mas largo con huella y su cuenta.
"""
from __future__ import annotations

import numpy as np

VOCAB = 32768
KS_DEF = (8, 12, 16, 24, 32)
TROZO = 4_000_000
M64 = (1 << 64) - 1
CNT_MAX = (1 << 24) - 1
BITS_CUENTA = {64: 24, 32: 6}
BITS_HUELLA = {64: 17, 32: 11}
TIPO = {64: np.uint64, 32: np.uint32}


def _splitmix(i: int) -> int:
    z = ((i + 0x5EED) * 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) | 1


MULT = np.array([_splitmix(i) for i in range(64)], dtype=np.uint64)


def _mezcla(x):
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xC4CEB9FE1A85EC53)
    return x ^ (x >> np.uint64(33))


def claves(t: np.ndarray, ks, bos: int):
    """Para t [..., L] y k crecientes: (k, h, valido) del contexto de k tokens que acaba en cada posicion i >= k-1."""
    t64 = t.astype(np.uint64) + np.uint64(1)
    largo = t.shape[-1]
    s = np.zeros(t.shape, np.uint64)
    malo = np.zeros(t.shape, bool)
    j = 0
    with np.errstate(over="ignore"):
        for k in sorted(ks):
            while j < k:
                if j < largo:
                    s[..., j:] += t64[..., :largo - j] * MULT[j]
                    malo[..., j:] |= t[..., :largo - j] == bos
                j += 1
            yield k, _mezcla(s[..., k - 1:] ^ np.uint64(k)), ~malo[..., k - 1:]


def cuenta_max(tipo) -> int:
    return (1 << BITS_CUENTA[np.dtype(tipo).itemsize * 8]) - 1


def _partes(h: np.ndarray, bits: int, ancho: int = 64):
    cubo = (h & np.uint64((1 << bits) - 1)).astype(np.int64)
    if ancho == 32:
        return cubo, ((h >> np.uint64(40)) % np.uint64(2047) + np.uint64(1)).astype(np.int64)
    return cubo, ((h >> np.uint64(47)) | np.uint64(1)).astype(np.int64)


def _unicos(cubo: np.ndarray, fv: np.ndarray, desp: int, bits: int):
    if bits + desp <= 63:
        clave, c = np.unique((cubo << desp) | fv, return_counts=True)
        return clave >> desp, clave & ((1 << desp) - 1), c
    o = np.lexsort((fv, cubo))
    cb, f = cubo[o], fv[o]
    ini = np.flatnonzero(np.r_[True, (cb[1:] != cb[:-1]) | (f[1:] != f[:-1])])
    return cb[ini], f[ini], np.diff(np.r_[ini, cb.size])


class Tabla:
    def __init__(self, ks=KS_DEF, bits: int = 24, bufs=None, ancho: int = 64):
        self.ks, self.bits, self.ancho = tuple(sorted(ks)), bits, ancho
        n, tipo = 1 << bits, TIPO[ancho]
        self.t = {k: (np.zeros(n, tipo) if bufs is None else np.ndarray((n,), tipo, buffer=bufs[k])) for k in self.ks}

    @classmethod
    def desde_arrays(cls, ks, bits: int, arrays: dict) -> "Tabla":
        tb = cls.__new__(cls)
        tb.ks, tb.bits, tb.t = tuple(sorted(ks)), bits, {k: arrays[k] for k in sorted(ks)}
        tb.ancho = arrays[tb.ks[0]].dtype.itemsize * 8
        return tb

    @property
    def cb(self) -> int:
        return BITS_CUENTA[self.ancho]

    def actualizar(self, tokens: np.ndarray, bos: int) -> int:
        """Cuenta los pares (contexto, siguiente) de una secuencia de tokens de documentos con BOS delante; devuelve pares."""
        kmax, total = self.ks[-1], 0
        for a in range(0, max(tokens.size - self.ks[0], 0), TROZO):
            s = tokens[a:min(tokens.size, a + TROZO + kmax)]
            for k, h, ok in claves(s[:-1], self.ks, bos):
                h, ok = h[:TROZO], ok[:TROZO]
                nx = s[k:][:TROZO]
                ok &= nx != bos
                total += int(ok.sum())
                self._sumar(k, h[ok], nx[ok].astype(np.int64))
        return total

    def _sumar(self, k: int, h: np.ndarray, nx: np.ndarray):
        if h.size == 0:
            return
        cubo, huella = _partes(h, self.bits, self.ancho)
        cb_, cmax = self.cb, (1 << self.cb) - 1
        cb, fv, c = _unicos(cubo, (huella << 15) | nx, 15 + BITS_HUELLA[self.ancho], self.bits)
        ini = np.flatnonzero(np.r_[True, cb[1:] != cb[:-1]])
        tam = np.diff(np.r_[ini, cb.size])
        g = np.repeat(np.arange(ini.size), tam)
        tot = np.add.reduceat(c, ini)
        gmax = np.maximum.reduceat(c, ini)
        es_max = c == gmax[g]
        _, primero = np.unique(g[es_max], return_index=True)
        cand = np.flatnonzero(es_max)[primero]
        tabla = self.t[k]
        bloque = cb[ini]
        e = tabla[bloque].astype(np.int64)
        e_fv, e_c = e >> cb_, e & cmax
        m_e = np.add.reduceat(np.where(fv == e_fv[g], c, 0), ini)
        nuevo = e_c + 2 * m_e - tot
        queda = (nuevo > 0) & (e_c > 0)
        c_rep = np.maximum(2 * c[cand] - tot - e_c, 1)
        val = np.where(queda, (e_fv << cb_) | np.minimum(nuevo, cmax), (fv[cand] << cb_) | np.minimum(c_rep, cmax))
        tabla[bloque] = val.astype(tabla.dtype)

    def consultar(self, ids: np.ndarray, bos: int, minimo: int = 1):
        """ids [B, T]: candidato para el token T+1 de cada posicion (-1 si ninguno), indice del k usado y cuenta neta."""
        forma = ids.shape
        tok = np.full(forma, -1, np.int64)
        ki = np.full(forma, -1, np.int8)
        cnt = np.zeros(forma, np.int64)
        cb_ = self.cb
        for i, (k, h, ok) in enumerate(claves(ids, self.ks, bos)):
            cubo, huella = _partes(h, self.bits, self.ancho)
            e = self.t[k][cubo].astype(np.int64)
            ec = e & ((1 << cb_) - 1)
            hit = ok & ((e >> (cb_ + 15)) == huella) & (ec >= minimo)
            sl = (Ellipsis, slice(k - 1, None))
            tok[sl] = np.where(hit, (e >> cb_) & ((1 << 15) - 1), tok[sl])
            ki[sl] = np.where(hit, i, ki[sl])
            cnt[sl] = np.where(hit, ec, cnt[sl])
        return tok, ki, cnt

    def ocupacion(self, paso: int = 1) -> dict:
        return {k: float(np.count_nonzero(v[::paso]) / v[::paso].size) for k, v in self.t.items()}

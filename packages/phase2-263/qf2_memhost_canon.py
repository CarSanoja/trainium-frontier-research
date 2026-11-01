"""QF2_MEMHOST_CANON: tokenizer compression for the hash, derived only from the provided tokenizer.

Tokens whose decoded text is equal after NFKC, lower-casing and stripping edge whitespace and punctuation share one
representative id (the smallest id of the class). BOS and tokens whose normalised text is empty map to themselves. The
map travels in the checkpoint as the int32 buffer memhost.canon.
"""
from __future__ import annotations

import string
import unicodedata

import numpy as np

BORDES = string.punctuation + string.whitespace


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKC", texto).lower().strip(BORDES)


def construir(decode, vocab: int, bos: int) -> np.ndarray:
    canon = np.arange(vocab, dtype=np.int32)
    rep: dict = {}
    for t in range(vocab):
        if t == bos:
            continue
        try:
            clave = normalizar(decode([t]))
        except Exception:
            continue
        if clave:
            canon[t] = rep.setdefault(clave, t)
    return canon

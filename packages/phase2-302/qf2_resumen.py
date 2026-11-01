"""QF2_RESUMEN='entrada:salida': causal summary of the current document, read after one block and added before a later one.

After block `entrada` (0-based index of the block loop of GPT.forward) the normalised residual stream h = norm(x) is
averaged over the positions of the current document up to and including each position: m_t is the mean of h_s for s in
[start of the document of t, t]. A document starts at every BOS token, and at the start of the row. Before block `salida`
the residual stream receives x + W_o(norm(m)), where W_o is one D x D matrix without bias initialised to zero, so at
initialisation the model is exactly the one without the branch.

The mean uses matrix products only, with no cumsum, cummax or scan along the sequence (Neuron does not lower them in the
compiled forward): c = L @ [idx == BOS] numbers the documents of each row (L is the T x T lower triangle of ones, in
float32 so the count stays exact), M[b, i, j] = (j <= i) & (c[b, j] == c[b, i]) selects the earlier positions of the same
document, S = M @ h and the count is M.sum(-1). Every output position reads only its own row and positions <= itself.
W_o is built under a forked RNG and zeroed in iniciar(), so the initialisation stream of the rest of the model is untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def capas(texto: str, n_layer: int) -> tuple[int, int]:
    partes = texto.strip().split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes):
        raise ValueError(f"QF2_RESUMEN={texto!r} debe ser 'entrada:salida' con dos enteros")
    entrada, salida = int(partes[0]), int(partes[1])
    if not 0 <= entrada < salida < n_layer:
        raise ValueError(f"QF2_RESUMEN={texto!r} exige 0 <= entrada < salida < n_layer={n_layer}")
    return entrada, salida


def mascara(idx: torch.Tensor, bos: int, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(idx.size(1), device=idx.device)
    inferior = pos.view(1, -1) <= pos.view(-1, 1)
    es_bos = (idx == bos).to(torch.float32)
    c = F.linear(es_bos, inferior.to(torch.float32))
    return (inferior.unsqueeze(0) & (c.unsqueeze(1) == c.unsqueeze(2))).to(dtype)


def media(h: torch.Tensor, idx: torch.Tensor, bos: int) -> torch.Tensor:
    m = mascara(idx, bos, h.dtype)
    suma = torch.matmul(m, h)
    cuenta = m.sum(-1, keepdim=True)
    return suma / cuenta


class Resumen(nn.Module):
    def __init__(self, texto: str, n_layer: int, n_embd: int, bos: int, lineal):
        super().__init__()
        self.capa_entrada, self.capa_salida = capas(texto, n_layer)
        self.bos = bos
        with torch.random.fork_rng(devices=[]):
            self.proy = lineal(n_embd, n_embd, bias=False)

    @torch.no_grad()
    def iniciar(self) -> None:
        nn.init.zeros_(self.proy.weight)

    def media(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return media(h, idx, self.bos)

    def inyectar(self, m: torch.Tensor) -> torch.Tensor:
        return self.proy(F.rms_norm(m, (m.size(-1),)))

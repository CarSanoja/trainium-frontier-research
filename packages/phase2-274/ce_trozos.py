from __future__ import annotations

import torch


def _trozo(x: torch.Tensor, wc: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = (x @ wc.t()).float()
    t = torch.tanh(z / cap)
    return t, cap * t


class _CETrozos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, cap, vocab, trozo):
        n = x.size(0)
        vale = y != -1
        lses = []
        total = x.new_zeros((), dtype=torch.float32)
        wc = w[:vocab].to(x.dtype)
        for i in range(0, n, trozo):
            _t, zc = _trozo(x[i:i + trozo], wc, cap)
            lse = torch.logsumexp(zc, dim=-1)
            zt = zc.gather(1, y[i:i + trozo].clamp_min(0).unsqueeze(1)).squeeze(1)
            total = total + torch.where(vale[i:i + trozo], lse - zt, 0.0).sum()
            lses.append(lse)
        lse = torch.cat(lses)
        ctx.save_for_backward(x, w, y, lse)
        ctx.cap, ctx.vocab, ctx.trozo = cap, vocab, trozo
        return total / vale.sum()

    @staticmethod
    def backward(ctx, g):
        x, w, y, lse = ctx.saved_tensors
        cap, vocab, trozo = ctx.cap, ctx.vocab, ctx.trozo
        vale = (y != -1)
        escala = g.float() / vale.sum()
        dw = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
        dxs = []
        wc = w[:vocab].to(x.dtype)
        for i in range(0, x.size(0), trozo):
            xc, yc, vc = x[i:i + trozo], y[i:i + trozo], vale[i:i + trozo]
            t, zc = _trozo(xc, wc, cap)
            d = torch.exp(zc - lse[i:i + trozo].unsqueeze(1))
            d.scatter_add_(1, yc.clamp_min(0).unsqueeze(1), -torch.ones_like(d[:, :1]))
            d = d * (escala * vc.float()).unsqueeze(1) * (1.0 - t * t)
            dz = d.to(x.dtype)
            dxs.append(dz @ wc)
            dw[:vocab] += (dz.t() @ xc).to(w.dtype)
        return torch.cat(dxs), dw, None, None, None, None


def ce_trozos(x: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, cap: float, vocab: int, trozo: int) -> torch.Tensor:
    if trozo <= 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} debe ser positivo")
    trozo = min(trozo, x.size(0))
    if x.size(0) % trozo != 0:
        raise ValueError(f"QF2_CE_TROZOS={trozo} no divide las {x.size(0)} filas")
    return _CETrozos.apply(x, w, targets, cap, vocab, trozo)

"""QF2_MEMHOST: per-row int8 quantisation with an fp16 scale, shared by training reads and the checkpoint.

scale = fp16(amax / 127) with a floor of 2^-14. After the fp16 rounding amax / scale lies in [126.9, 127.1], so q reaches 127
and |w - dq(q(w))| <= scale / 2 for every element (the clamp loses at most 0.07 scale); re-quantising dq(q(w)) returns the
same scale and the same q. The training gather applies ida_vuelta to the fp32 masters,
so the values the graph sees in the last step are bit for bit those evaluation reads back from the checkpoint.
"""
from __future__ import annotations

import torch

SUELO = 2.0 ** -14


def escala(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax(dim=-1)
    return (amax / 127.0).clamp_min(SUELO).to(torch.float16)


def cuantizar(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s16 = escala(w)
    q = torch.round(w / s16.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q, s16


def descuantizar(q: torch.Tensor, s16: torch.Tensor) -> torch.Tensor:
    return q.float() * s16.float().unsqueeze(-1)


def ida_vuelta(w: torch.Tensor) -> torch.Tensor:
    q, s16 = cuantizar(w)
    return descuantizar(q, s16)

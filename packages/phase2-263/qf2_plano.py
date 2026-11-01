"""Transient Muon slab (QF2_PLANO_TRANSITORIO, default OFF).

The flat slab of RepartoMuon (world * chunk elements, 3.11 GB per rank at the d24 shape) is only used inside the
optimizer step and restaurar(). With the switch ON it is allocated at zero when the step begins and dropped when it
ends, so the allocator can hand the same bytes to activations and gradients during forward and backward.
"""

from __future__ import annotations

import torch


class Losa:
    def __init__(self, numel: int, dtype: torch.dtype, device: torch.device):
        self.numel, self.dtype, self.device = numel, dtype, device
        self.aperturas = 0

    def abrir(self) -> torch.Tensor:
        self.aperturas += 1
        return torch.zeros(self.numel, dtype=self.dtype, device=self.device)

    def cerrar(self) -> None:
        return None

    def bytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()

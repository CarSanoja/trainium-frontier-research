"""Adam over the row shards of the memory tables (QF2_MEM_FILAS). The shards are not in DistMuonAdamW: their gradient is
already the global average (see qf2_memoria.Consulta.backward). The update is MuonAdamW._adamw_update, passed in by train."""
from __future__ import annotations

import torch

BETAS = (0.8, 0.995)
EPS = 1e-10


class AdamShard:
    def __init__(self, params, lr: float, actualizar):
        self.actualizar = actualizar
        self.state: dict = {}
        self.param_groups = [dict(kind="adamw", params=list(params), lr=lr, initial_lr=lr, betas=BETAS, eps=EPS, weight_decay=0.0)]
        for g in self.param_groups:
            g["_lr_t"] = torch.tensor(float(g["lr"]), dtype=torch.float32)
            g["_wd_t"] = torch.tensor(0.0, dtype=torch.float32)
            g["_one_minus_beta1_t"] = torch.tensor(1 - BETAS[0], dtype=torch.float32)
            g["_one_minus_beta2_t"] = torch.tensor(1 - BETAS[1], dtype=torch.float32)
            g["_eps_t"] = torch.tensor(EPS, dtype=torch.float32)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    self.actualizar(p, p.grad, self.state.setdefault(p, {}), g)

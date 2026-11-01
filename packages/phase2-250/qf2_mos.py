"""QF2_MOS: mixture of two softmax distributions over the shared output head (astra r37, proposal C).

From the final hidden state h (after the final norm) two representations are built: h1 = h and h2 = norm(h + W2 h). Both go through
the same lm_head and the same logit softcap of the kit, and a gate pi(h) = sigmoid(w_pi . h + b_pi) mixes the two distributions:
log p = logaddexp(log pi + log_softmax(logits_1), log(1 - pi) + log_softmax(logits_2)). Everything is per position, so causality and
batch independence are those of the rest of the model.

Training loss: -mean(log p[target]) is computed from the per-branch logsumexp and the target logit only,
log p[target] = logaddexp_i(log pi_i + z_i[target] - lse_i), so no mixed [N, V] tensor is built. With cefa=True the lse comes from
cefa_perdida.cef_lse (bf16 residual per branch); otherwise it is the explicit softcap + logsumexp route, which keeps two fp32
[N, V] tensors per branch for the backward pass (tanh output and softcapped logits).

Evaluation (targets=None): the model returns log p over the full vocabulary as its logits. exp(log p) sums to 1 per position, so the
log_softmax / cross-entropy applied by prepare.py and core_eval returns exactly -log p[target] without altering the scoring procedure.

W2 and w_pi/b_pi are plain Parameters created empty (no RNG consumed) and initialised last, so with QF2_MOS=0 the model and its
initialisation stream are untouched. W2 starts at W2_ESCALA times the usual uniform init so the two branches already differ at step 0
(an exact copy h2 = h would give a mixture identical to the single softmax and a symmetric start); w_pi starts at 0 and b_pi at 0, so
pi = 0.5.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RAMAS = 2
W2_ESCALA = 0.5
PI_LR_BASE = 0.2


def comprobar(ramas: int) -> None:
    if ramas != RAMAS:
        raise ValueError(f"QF2_MOS={ramas} debe ser 0 (apagado) o {RAMAS} (dos ramas)")


def pi_lr(n_embd: int) -> float:
    return PI_LR_BASE / n_embd


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def logits_rama(h: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float) -> torch.Tensor:
    z = lm_head(h)[..., :vocab].float()
    return cap * torch.tanh(z / cap)


def lse_y_objetivo(h: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, ids: torch.Tensor, cefa: bool):
    if cefa:
        from cefa_perdida import cef_lse
        w = lm_head.weight
        lse = cef_lse(F.linear(h, w[:vocab].to(dtype=h.dtype), None))
        zt = (h * F.embedding(ids, w).to(dtype=h.dtype)).float().sum(-1)
        return lse, cap * torch.tanh(zt / cap)
    z = logits_rama(h, lm_head, vocab, cap)
    return torch.logsumexp(z, dim=-1), z.gather(1, ids.unsqueeze(1)).squeeze(1)


class Mezcla(nn.Module):
    def __init__(self, n_embd: int, ramas: int = RAMAS):
        super().__init__()
        comprobar(ramas)
        self.w2 = nn.Parameter(torch.empty(n_embd, n_embd))
        self.pi_w = nn.Parameter(torch.empty(n_embd, 1))
        self.pi_b = nn.Parameter(torch.empty(1))

    @torch.no_grad()
    def iniciar(self) -> None:
        n_embd = self.w2.size(1)
        s = W2_ESCALA * math.sqrt(3.0) * n_embd ** -0.5
        nn.init.uniform_(self.w2, -s, s)
        nn.init.zeros_(self.pi_w)
        nn.init.zeros_(self.pi_b)

    def ramas(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return h, norm(h + F.linear(h, self.w2.to(h.dtype)))

    def log_puertas(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = h.float() @ self.pi_w.float() + self.pi_b.float()
        return F.logsigmoid(g), F.logsigmoid(-g)

    def log_prob(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float) -> torch.Tensor:
        (h1, h2), (l1, l2) = self.ramas(x), self.log_puertas(x)
        a = l1 + F.log_softmax(logits_rama(h1, lm_head, vocab, cap), dim=-1)
        b = l2 + F.log_softmax(logits_rama(h2, lm_head, vocab, cap), dim=-1)
        return torch.logaddexp(a, b)

    def nll_tokens(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, targets: torch.Tensor, cefa: bool):
        n_embd = x.size(-1)
        xf, ids = x.reshape(-1, n_embd), targets.reshape(-1).clamp_min(0)
        (h1, h2), (l1, l2) = self.ramas(xf), self.log_puertas(xf)
        lse1, zt1 = lse_y_objetivo(h1, lm_head, vocab, cap, ids, cefa)
        lse2, zt2 = lse_y_objetivo(h2, lm_head, vocab, cap, ids, cefa)
        return -torch.logaddexp(l1.squeeze(1) + zt1 - lse1, l2.squeeze(1) + zt2 - lse2)

    def forward(self, x: torch.Tensor, lm_head: nn.Module, vocab: int, cap: float, targets: torch.Tensor | None = None,
                loss_reduction: str = "mean", cefa: bool = False) -> torch.Tensor:
        if targets is None:
            return self.log_prob(x, lm_head, vocab, cap)
        nll = self.nll_tokens(x, lm_head, vocab, cap, targets, cefa)
        valido = targets.reshape(-1) >= 0
        nll = torch.where(valido, nll, torch.zeros_like(nll))
        if loss_reduction == "none":
            return nll.view_as(targets)
        return nll.sum() / valido.sum()

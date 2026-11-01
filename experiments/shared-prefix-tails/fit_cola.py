"""POC de sufijo evaluado: adapters entrenables, tronco/cache congelados, sin CORE.
No carga GPT/pesos ni lanza trabajos. Caller aporta Suffix(**cache), batches y sync.
"""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Delta B@A con escala fija 1, independiente del rango; B inicial cero."""
    def __init__(self, base, rank, generator):
        super().__init__()
        if rank < 1 or getattr(base, 'bias', None) is not None:
            raise ValueError('Linear sin bias y rango positivo requeridos')
        self.base = base
        self.base.requires_grad_(False)
        shape = base.weight.shape
        self.a = nn.Parameter(torch.randn(rank, shape[1], generator=generator, dtype=torch.float32) * .02)
        self.b = nn.Parameter(torch.zeros(shape[0], rank, dtype=torch.float32))
        self.to(device=base.weight.device)

    def forward(self, x):
        effective = self.base.weight + self.b @ self.a  # Merge FP32, antes del dot.
        return F.linear(x, effective.to(x.dtype), None)


def preparar(suffix, rank, seed):
    """Consume un Suffix privado: no compartir Blocks mutables entre brazos."""
    if len(suffix.blocks) != 4:
        raise ValueError('Se requieren exactamente cuatro Blocks finales')
    suffix.requires_grad_(False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for block in suffix.blocks:
        for owner, names in ((block.attn, ('c_q', 'c_k', 'c_v', 'c_proj')),
                             (block.mlp, ('c_fc', 'c_proj'))):
            for name in names:
                base = getattr(owner, name)
                if isinstance(base, LoRALinear):
                    raise ValueError('Sufijo ya adaptado')
                setattr(owner, name, LoRALinear(base, rank, generator))
    suffix.eval()  # IMPORTANTE: training FP8 accede a .weight y omitiría wrappers.
    return suffix


def parametros(suffix):
    return [p for p in suffix.parameters() if p.requires_grad]


def verificar_independencia(tails):
    """Base congelada puede compartirse; adapters y Blocks no pueden aliasarse."""
    seen = set()
    blocks = set()
    for tail in tails:
        for block in tail.blocks:
            if id(block) in blocks:
                raise ValueError('Block mutable compartido')
            blocks.add(id(block))
        for p in parametros(tail):
            if p.device.type == 'cpu':
                key = ('cpu_storage', p.untyped_storage().data_ptr())
            else:
                key = ('native_object_only_requires_functional_anchor', id(p))
            if key in seen:
                raise ValueError('Adapter compartido')
            seen.add(key)
    native = any(p.device.type != 'cpu' for tail in tails for p in parametros(tail))
    return {'native_storage_verified': False if native else None,
            'mode': 'object_identity_plus_external_functional_anchor' if native else 'cpu_storage'}


def documentos(batches):
    ids = set()
    for batch in batches:
        names = batch['document_ids']
        if len(names) != batch['targets'].shape[0] or not all(isinstance(n, str) and n for n in names):
            raise ValueError('Un ID documental auténtico por fila es obligatorio')
        ids.update(names)
    return ids


def validar_split(train, heldout):
    if not train or not heldout or documentos(train) & documentos(heldout):
        raise ValueError('TRAIN y heldout deben ser no vacíos y disjuntos por documento')
    for batch in (*train, *heldout):
        if batch['targets'].shape != batch['cache']['idx'].shape:
            raise ValueError('Targets deben alinear todas las posiciones causales de cache')
        if any(isinstance(v, torch.Tensor) and v.requires_grad for v in batch['cache'].values()):
            raise ValueError('Cache prefix debe estar detached/congelada')


def log_probs(tail, batch):
    tail.eval()
    return F.log_softmax(tail(**batch['cache']).float(), dim=-1)


def target_logp(logp, targets):
    if targets.dtype not in (torch.int32, torch.int64) or bool(((targets < -1) | (targets >= logp.shape[-1])).any()):
        raise ValueError('Targets sólo enteros -1 o [0,vocab)')
    valid = targets != -1
    if not bool(valid.any()):
        raise ValueError('Batch sin targets puntuables')
    chosen = logp.gather(-1, targets.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
    return chosen, valid


def ajustar(tails, train, *, max_steps=200, max_tokens=30_000_000,
            max_seconds=1200, lr=1e-3, sync=lambda: None, batch_transform=lambda x: x):
    """CE ordinaria independiente por cola, mismos batches; nunca train().
    Límite wall entre updates, NO hard timeout: caller debe imponer sobre externo.
    """
    if not 0 < max_seconds <= 1200 or max_steps < 1 or max_tokens < 1:
        raise ValueError('Presupuesto POC inválido')
    verificar_independencia(tails)
    optimizers = [torch.optim.AdamW(parametros(t), lr=lr, weight_decay=0) for t in tails]
    sync()
    started = time.monotonic()
    steps = tokens = 0
    wall_per_tail = [0.] * len(tails)
    gradient_seen = {f'{i}:{n}': False for i, t in enumerate(tails)
                     for n, p in t.named_parameters() if p.requires_grad}
    losses = []
    schedule = []
    while steps < max_steps and time.monotonic() - started < max_seconds:
        batch = train[steps % len(train)]
        count = int((batch['targets'] != -1).sum())
        if tokens + count > max_tokens:
            break
        batch = batch_transform(batch)
        row = []
        with torch.enable_grad():
            for i, (tail, optimizer) in enumerate(zip(tails, optimizers)):
                sync()
                arm_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                lp, valid = target_logp(log_probs(tail, batch), batch['targets'])
                loss = -lp[valid].mean()
                if not bool(torch.isfinite(loss).cpu().item()):
                    raise FloatingPointError('CE no finita')
                loss.backward()
                for name, p in tail.named_parameters():
                    if p.requires_grad:
                        if p.grad is None or not bool(torch.isfinite(p.grad).all().cpu().item()):
                            raise FloatingPointError('Adapter sin gradiente finito: ' + name)
                        gradient_seen[f'{i}:{name}'] |= bool((p.grad != 0).any().cpu().item())
                optimizer.step()
                sync()
                wall_per_tail[i] += time.monotonic() - arm_started
                row.append(float(loss.detach().cpu()))
        sync()
        losses.append(row)
        schedule.append(list(batch['document_ids']))
        tokens += count
        steps += 1
    return {'steps': steps, 'complete_requested_steps': steps == max_steps, 'document_schedule': schedule, 'tokens_per_tail': tokens, 'tokens_unique_schedule': tokens,
            'wall_seconds': time.monotonic() - started, 'wall_per_tail_seconds': wall_per_tail, 'losses': losses,
            'trainable_parameters': [sum(p.numel() for p in parametros(t)) for t in tails],
            'nonzero_gradient_seen': gradient_seen, 'path': 'eval_with_grad',
            'lora_scale': 1., 'optimizer': 'AdamW lr declared, wd0', 'lr': lr}


def medir(tails, batches, *, sync=lambda: None):
    from metricas import medir as measure
    return measure(tails, batches, sync=sync)


def verificar_capacidad(control, pair):
    if len(pair) != 2:
        raise ValueError('Se requieren dos colas')
    verificar_independencia([control, *pair])
    counts = [sum(p.numel() for p in parametros(t)) for t in [control, *pair]]
    if counts[0] != counts[1] + counts[2] or counts[1] != counts[2]:
        raise ValueError('Capacidad entrenable no igualada')
    return {'control': counts[0], 'pair': counts[1:], 'scale_each': 1.}

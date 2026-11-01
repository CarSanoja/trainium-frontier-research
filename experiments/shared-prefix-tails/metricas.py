"""Decisión POC heldout: bootstrap pareado por documento, no CORE."""
import random
import math
import time
import torch


def comparar(pair, baseline, control, *, draws=10000, seed=20261101, splitbase=None):
    names = sorted(pair['documents'])
    if not names or any(set(r['documents']) != set(names) for r in (baseline, control)):
        raise ValueError('Documentos heldout no alineados')
    rng = random.Random(seed)
    result = {}
    references = [('original_monolithic', baseline), ('control_rank2r', control)]
    if splitbase is not None:
        references.append(('splitbase_variant', splitbase))
    for label, reference in references:
        if set(reference['documents']) != set(names):
            raise ValueError('Documentos de referencia no alineados')
        rows = []
        for name in names:
            p, r = pair['documents'][name], reference['documents'][name]
            if p['tokens'] != r['tokens'] or p['tokens'] < 1:
                raise ValueError('Targets por documento no alineados')
            rows.append((p['mixture_nll_sum'] - r['mixture_nll_sum'], p['tokens']))
        delta = sum(a for a, _ in rows) / sum(n for _, n in rows)
        boot = []
        for _ in range(draws):
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            boot.append(sum(a for a, _ in sample) / sum(n for _, n in sample))
        boot.sort()
        upper = boot[min(draws - 1, int(.975 * draws))]
        result[label] = {'delta_nats_per_token': delta, 'paired_doc_ci95_upper': upper,
                         'passes': delta <= -.01 and upper < 0}
    return {'comparisons': result, 'generalization_passes': all(r['passes'] for r in result.values()),
            'bootstrap': {'unit': 'document', 'draws': draws, 'seed': seed,
                          'interval': 'two-sided percentile 95%, upper97.5'},
            'scope': 'Condicionado a 200 updates comunes completos; no CORE ni financiación'}


@torch.no_grad()
def medir(tails, batches, *, sync):
    from fit_cola import log_probs, target_logp
    if len(tails) not in (1, 2):
        raise ValueError('Una o dos colas')
    totals = [0.] * len(tails)
    correct = [0] * len(tails)
    mixture = 0.
    mixture_correct = disagreement = tokens = 0
    docs = {}
    sync()
    started = time.monotonic()
    for batch in batches:
        logs = [log_probs(t, batch) for t in tails]
        if not all(bool(torch.isfinite(lp).all()) for lp in logs):
            raise FloatingPointError('Heldout log-prob no finito')
        selected = [target_logp(lp, batch['targets']) for lp in logs]
        valid = selected[0][1]
        predictions = [lp.argmax(-1) for lp in logs]
        for i, (lp, _) in enumerate(selected):
            totals[i] -= float(lp[valid].sum())
            correct[i] += int(((predictions[i] == batch['targets']) & valid).sum())
        # Full-vocab mezcla, necesaria para argmax mixto; NO sólo target elegido.
        full_mixture = torch.logsumexp(torch.stack(logs), dim=0) - math.log(len(tails))
        if not bool(torch.isfinite(full_mixture).all()):
            raise FloatingPointError('Heldout mezcla no finita')
        mixed, _ = target_logp(full_mixture, batch['targets'])
        if not bool(torch.isfinite(mixed).all()):
            raise FloatingPointError('Heldout NLL no finita')
        mix_matches = (full_mixture.argmax(-1) == batch['targets']) & valid
        mixture -= float(mixed[valid].sum())
        mixture_correct += int(mix_matches.sum())
        if len(tails) == 2:
            disagreement += int(((predictions[0] != predictions[1]) & valid).sum())
        tokens += int(valid.sum())
        for row, name in enumerate(batch['document_ids']):
            rec = docs.setdefault(name, {'tokens': 0, 'mixture_nll_sum': 0., 'mixture_correct': 0})
            rec['tokens'] += int(valid[row].sum())
            rec['mixture_nll_sum'] -= float(mixed[row][valid[row]].sum())
            rec['mixture_correct'] += int(mix_matches[row].sum())
    if not tokens:
        raise ValueError('Heldout sin tokens')
    sync()
    return {'documents': docs, 'tokens': tokens, 'tail_nll': [v / tokens for v in totals],
            'tail_accuracy': [v / tokens for v in correct], 'mixture_accuracy': mixture_correct / tokens,
            'mixture_nll': mixture / tokens, 'mixture_weights': [1 / len(tails)] * len(tails),
            'argmax_disagreement_fraction': disagreement / tokens,
            'wall_seconds': time.monotonic() - started, 'not_CORE_or_BPB': True}

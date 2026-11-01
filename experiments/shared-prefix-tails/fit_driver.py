"""Caller B POC; sólo invocable por entry autorizado, sin loader GPT/Native propio."""
import json
from pathlib import Path
import time
import torch
import datos_fit as datos
import fit_cola as fit
import metricas


def ejecutar_poc(prefix_callable, suffix_factory, compile_suffix, sync, device,
                  tokenizer, prepare, out, budget_started, *, validar_inicial,
                  provenance, original_logits, variant_authorization, rank=4, lr=.001,
                  max_seconds=1020, eval_reserve_seconds=180):
    """200 updates C,A,B; 1200s TOTAL desde inicio exterior, no matched wall.
    Caller impone hard timeout y autentica fuentes/admisión. Aoriginal FAILED se preserva; nueva variante requiere admisión aparte.
    """
    if not 0 < max_seconds <= 1200 or not 0 < eval_reserve_seconds < max_seconds:
        raise ValueError('Compute deadline debe ser positivo y≤1200s total')
    if not callable(sync) or not callable(validar_inicial) or budget_started > time.monotonic():
        raise ValueError('Sync/anchor/initmonotónico obligatorios')
    if (not callable(original_logits) or not isinstance(variant_authorization, dict)
            or variant_authorization.get('mode') != 'split_variant_vs_original'
            or variant_authorization.get('A_original_equivalence_passed') is not False
            or variant_authorization.get('A_cache_copy_passed') is not True
            or not variant_authorization.get('new_preregister')):
        raise ValueError('Variante Split no admitida explícitamente; Aoriginal sigueFAILED')
    out = Path(out)
    out.mkdir(exist_ok=False)
    result = {'complete': False, 'status': 'INCONCLUSO', 'scope': 'POC no elegible, no CORE',
              'timings': {}, 'rank': rank, 'lr': lr, 'provenance': provenance,
              'variant_authorization': variant_authorization}
    def save():
        result['total_seconds'] = time.monotonic() - budget_started
        temporary = out / 'resultado.tmp'
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(out / 'resultado.json')
    def check_budget():
        if time.monotonic() - budget_started >= max_seconds:
            raise TimeoutError('Compute deadline absoluto agotado')
    def phase(label, callable_):
        check_budget()
        sync()
        begin = time.monotonic()
        value = callable_()
        sync()
        result['timings'][label] = time.monotonic() - begin
        save()
        return value
    try:
        selected = phase('document_selection', lambda: datos.seleccionar(
            prepare, tokenizer, out, check_budget=check_budget, provenance=provenance))
        documents_fit, documents_heldout, _ = selected
        train, train_receipt = phase('prefix_fit_cache', lambda: datos.construir_cache(
            documents_fit, prefix_callable, device, out, 'fit', sync=sync,
            check_budget=check_budget, host_bytes_remaining=2**30))
        heldout, heldout_receipt = phase('prefix_heldout_cache', lambda: datos.construir_cache(
            documents_heldout, prefix_callable, device, out, 'heldout', sync=sync,
            check_budget=check_budget, host_bytes_remaining=2**30-train_receipt['host_bytes']))
        datos.write_json(out / 'cache-manifest.json', {'fit': train_receipt, 'heldout': heldout_receipt})
        fit.validar_split(train, heldout)
        result['cache_host_bytes'] = train_receipt['host_bytes'] + heldout_receipt['host_bytes']
        # Privacidad de Blocks es responsabilidad factory y se comprueba antes de compilar.
        baseline = suffix_factory().eval().requires_grad_(False)
        control, a, b = [fit.preparar(suffix_factory(), r, seed)
                         for r, seed in ((2*rank, 11), (rank, 12), (rank, 13))]
        result['independence'] = fit.verificar_independencia([baseline, control, a, b])
        result['capacity'] = fit.verificar_capacidad(control, [a, b])
        frozen = [(p, p._version, p.dtype) for t in (baseline, control, a, b)
                  for p in t.parameters() if not p.requires_grad]
        models = {name: compile_suffix(tail) for name, tail in
                  (('baseline', baseline), ('control', control), ('a', a), ('b', b))}
        anchor = phase('zero_adapter_anchor', lambda: validar_inicial(
            models, datos.a_device(train[0], device)))
        if (not isinstance(anchor, dict) or anchor.get('valido') is not True
                or anchor.get('reference') != 'splitbase_variant'):
            raise RuntimeError('LoRAB0 no exacto contraSplitbase; no aprender')
        result['initial_anchor'] = anchor
        remaining = max_seconds - (time.monotonic()-budget_started) - eval_reserve_seconds
        if remaining <= 0:
            raise TimeoutError('Sin reserva para200updates comunes yheldout')
        result['fit'] = phase('fit_C_A_B', lambda: fit.ajustar(
            [models['control'], models['a'], models['b']], train, max_steps=200,
            max_tokens=200*1024, max_seconds=min(1200, remaining), lr=lr, sync=sync,
            batch_transform=lambda batch: datos.a_device(batch, device)))
        if not result['fit']['complete_requested_steps']:
            result['reason'] = 'Menos de200updates comunes; ningún brazo se completa después'
            return result
        if not all(result['fit']['nonzero_gradient_seen'].values()):
            raise RuntimeError('Algún adapter nunca recibió gradiente no cero')
        result['frozen_parameter_versions_unchanged'] = all(
            p._version == version and p.dtype == dtype for p, version, dtype in frozen)
        if not result['frozen_parameter_versions_unchanged']:
            raise RuntimeError('Fuente congelada cambió')
        # Save adapters únicamente; no copiar ni serializar pesos base.
        for name, tail in (('control', control), ('a', a), ('b', b)):
            def save_adapter():
                path = out / ('adapter-' + name + '.pt')
                with path.open('xb') as stream:
                    torch.save({n: p.detach().cpu() for n, p in tail.named_parameters()
                                if p.requires_grad}, stream)
                return {'path': path.name, 'sha256': datos.sha(path), 'bytes': path.stat().st_size}
            result.setdefault('adapters', {})[name] = phase('save_' + name, save_adapter)
        # Cada métrica recorre exactamente los mismos32docs/targets, sin actualizar.
        evaluations = {}
        class Original(torch.nn.Module):
            def forward(self, **cache):
                return original_logits(cache['idx'])
        original = Original().eval()
        for name, group in (('original', [original]), ('splitbase', [models['baseline']]), ('control', [models['control']]),
                            ('a', [models['a']]), ('b', [models['b']]),
                            ('pair', [models['a'], models['b']])):
            def evaluate():
                def batches():
                    for batch in heldout:
                        check_budget()
                        yield datos.a_device(batch, device)
                return fit.medir(group, batches(), sync=sync)
            evaluations[name] = phase('heldout_' + name, evaluate)
        result['heldout'] = evaluations
        if not all(p._version == version and p.dtype == dtype for p, version, dtype in frozen):
            raise RuntimeError('Fuente congelada cambió durante evaluación')
        result['decision'] = phase('bootstrap_doc', lambda: metricas.comparar(
            evaluations['pair'], evaluations['original'], evaluations['control'],
            splitbase=evaluations['splitbase']))
        check_budget()
        result['complete'] = True
        result['status'] = 'POSITIVO_POC' if result['decision']['generalization_passes'] else 'NEGATIVO_POC'
        return result
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        save()

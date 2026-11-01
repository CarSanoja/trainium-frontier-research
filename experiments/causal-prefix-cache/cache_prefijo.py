"""Soporte disperso del siguiente token observado, calculado sólo desde el prefijo."""
from collections import Counter
import torch

ORDEN_MIN = 2
ORDEN_MAX = 8
SOPORTE = 16


def soporte(filas, bos):
    indices, pesos = [], []
    for fila in filas:
        mapas = [dict() for _ in range(ORDEN_MAX + 1)]
        inicio, ids_fila, p_fila = 0, [], []
        for t, token in enumerate(fila):
            ids, probs = [0] * SOPORTE, [0.] * SOPORTE
            if token == bos:
                mapas = [dict() for _ in range(ORDEN_MAX + 1)]
                inicio = t + 1
            else:
                for k in range(ORDEN_MIN, min(ORDEN_MAX, t - inicio) + 1):
                    clave = tuple(fila[t-k:t])
                    cuenta = mapas[k].setdefault(clave, Counter())
                    cuenta[token] += 1
                for k in range(min(ORDEN_MAX, t - inicio + 1), ORDEN_MIN - 1, -1):
                    cuenta = mapas[k].get(tuple(fila[t-k+1:t+1]))
                    if cuenta:
                        total = sum(cuenta.values())
                        for j, (v, n) in enumerate(sorted(cuenta.items(), key=lambda p: (-p[1], p[0]))[:SOPORTE]):
                            ids[j], probs[j] = v, n / total
                        break
            ids_fila.append(ids)
            p_fila.append(probs)
        indices.append(ids_fila)
        pesos.append(p_fila)
    return indices, pesos


def aplicar(input_ids, logits, beta, bos):
    if beta == 0 or logits is None:
        return logits
    ids, pesos = soporte(input_ids.detach().cpu().tolist(), bos)
    posiciones = torch.tensor(ids, dtype=torch.long, device=logits.device)
    valores = torch.tensor(pesos, dtype=logits.dtype, device=logits.device)
    return logits.scatter_add_(2, posiciones, valores * beta)

"""Referencia por prefijos, estado oficial y mutantes que filtran futuro o mezclan filas."""
from collections import Counter
from pathlib import Path
import sys
import torch

from cache_prefijo import aplicar, soporte, SOPORTE
from cache_prefijo_modelo import Predictor
from estado_oficial import funciones
import cache_inference
import cache_train


def referencia(fila, bos):
    indices, pesos = [], []
    for t, token in enumerate(fila):
        inicio = max((j + 1 for j in range(t + 1) if fila[j] == bos), default=0)
        ids, probs = [0] * SOPORTE, [0.] * SOPORTE
        for k in range(min(8, t - inicio + 1), 1, -1):
            clave = fila[t-k+1:t+1]
            cuenta = Counter(fila[j] for j in range(inicio + k, t + 1) if fila[j-k:j] == clave)
            if cuenta:
                for j, (v, n) in enumerate(sorted(cuenta.items(), key=lambda p: (-p[1], p[0]))[:SOPORTE]):
                    ids[j], probs[j] = v, n / sum(cuenta.values())
                break
        indices.append(ids);pesos.append(probs)
    return indices, pesos


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tabla = torch.nn.Embedding(128, 128)
        self.max_seq_len = 2048

    def forward(self, ids):
        return self.tabla(ids)

    def get_eval_worker(self):
        return dict(worker_id=0, worker_count=1, is_leader=True)


def validar(lector):
    filas = [[0,1,2,3,1,2,4,1,2,3,1,2,5,1,2,3],
             ([1,2,3]*6)[:16], [9,8,9,8,7,9,8,7,9,8,0,9,8,6,9,8]]
    ids, pesos = lector(filas, 0)
    for r, fila in enumerate(filas):
        assert (ids[r], pesos[r]) == referencia(fila, 0)
    for t in range(16):
        otras = [f[:t+1]+[31]*(15-t) for f in filas]
        u,v = lector(otras,0)
        assert [r[:t+1] for r in u] == [r[:t+1] for r in ids]
        assert [r[:t+1] for r in v] == [r[:t+1] for r in pesos]


def main():
    torch.set_num_threads(1);torch.manual_seed(10730)
    assert cache_inference.load_for_eval is cache_train.load_for_eval
    validar(soporte)
    filas = [[0,1,2,3,1,2,4,1,2,3,1,2,5,1,2,3],
             [9,8,9,8,7,9,8,7,9,8,0,9,8,6,9,8]]
    for fila in filas:
        assert soporte([fila], 0) == tuple([v] for v in referencia(fila, 0))
    ids, pesos = soporte(filas, 0)
    assert any(sum(p) > 0 for r in pesos for p in r), 'ON nunca actúa'
    assert soporte(filas[::-1], 0) == (ids[::-1], pesos[::-1])
    for t in range(len(filas[0])):
        mutadas = [f[:t+1] + [31] * (len(f)-t-1) for f in filas]
        u, v = soporte(mutadas, 0)
        assert [r[:t+1] for r in u] == [r[:t+1] for r in ids]
        assert [r[:t+1] for r in v] == [r[:t+1] for r in pesos]
    assert soporte([filas[0]], 0) == tuple([v] for v in referencia(filas[0], 0))
    muchos = [0]
    for v in range(10, 34):muchos += [1,2,v]
    muchos += [1,2]
    a,b = soporte([muchos], 0)
    assert abs(sum(b[0][-1]) - 16/24) < 1e-12, 'renormalizó soporte truncado'
    base = Base().eval();x = torch.tensor(filas)
    original = base(x).detach().clone()
    assert aplicar(x, original.clone(), 0., 0) is not None
    assert torch.equal(aplicar(x, original.clone(), 0., 0), original)
    esperado = original.clone()
    for r in range(len(filas)):
        for t in range(len(filas[r])):
            for v,p in zip(ids[r][t],pesos[r][t]):esperado[r,t,v] += p
    assert torch.equal(aplicar(x, original.clone(), 1., 0), esperado)
    oficial = funciones(Path('upstream/round2/core_eval/evaluator.py'))
    modelo = Predictor(base, 1., 0)
    antes = oficial['snapshot_model_state'](modelo)
    with torch.no_grad():
        a = modelo(x).clone();modelo(torch.flip(x,[1]));b = modelo(x)
        assert torch.equal(a,b)
        for r in range(len(filas)):assert torch.equal(a[r], modelo(x[r:r+1])[0])
    oficial['assert_model_state_unchanged'](modelo, antes)
    fuente = Path(__file__).with_name('cache_prefijo.py').read_text()
    mutantes = [fuente.replace('for t, token in enumerate(fila):',
                 'for t, token in enumerate(fila):\n            token = fila[min(t+1,len(fila)-1)]'),
                 fuente.replace('if token == bos:', 'if False:'),
                 fuente.replace('    for fila in filas:\n        mapas = [dict() for _ in range(ORDEN_MAX + 1)]',
                 '    mapas = [dict() for _ in range(ORDEN_MAX + 1)]\n    for fila in filas:')]
    for numero, codigo in enumerate(mutantes):
        assert codigo != fuente
        espacio = {};exec(compile(codigo, f'mutante{numero}', 'exec'), espacio)
        try:
            validar(espacio['soporte'])
        except AssertionError:
            pass
        else:
            raise AssertionError(f'mutante{numero} no mordió')
    for t in (128,257,511,1024,1537,2048):
        fila = ([1,2,3,1,2,4]*((t+5)//6))[:t]
        u,v = soporte([fila],0)
        assert len(u[0]) == len(v[0]) == t
    print('PRUEBA_CACHE_PREFIJO fallos=0 referencia/OFF/ON/futuro/lotes/BOS/soporte16/estado/3mutantes/6T')


if __name__ == '__main__':
    main()

"""QF2_MEMHOST: fast table thread, bit for bit the same arithmetic as the reference path of qf2_memhost_hilo.

Work is done in blocks of about 1 MiB with fixed buffers (no large temporaries, so no page faults of fresh memory), signs
come from the unpacked bits of the 64-bit hash word, and the rows of a step are sorted once to aggregate duplicates in the
same order (stable, rank order for the exchange) as torch.unique + index_add_. Only the rms optimiser with signs, an
ancho multiple of 64 and bf16 vectors is covered: anything else must use the reference path (aplica()).
"""
from __future__ import annotations

import numpy as np
import torch

import qf2_memhost_cuantiza as cuant
import qf2_memhost_opt as optm
import qf2_memhost_tabla as tab

BLOQUE_BYTES = 1 << 20
ANILLO_EXTRA = 4


def aplica(cfg, modo: str, dtype: torch.dtype) -> bool:
    return modo == "rms" and cfg.signos and cfg.ancho % 64 == 0 and dtype == torch.bfloat16


class Estado:
    def __init__(self, cfg, tabla, intercambio, rango: int, retraso: int):
        self.cfg, self.tabla, self.inter, self.rango = cfg, tabla, intercambio, rango
        self.mundo = tabla.mundo
        w = cfg.ancho
        self.rb = max(256, BLOQUE_BYTES // (4 * w))
        self.cap = intercambio.cap
        self.bufE = [torch.empty(self.cap, w, dtype=torch.bfloat16) for _ in range(retraso + ANILLO_EXTRA)]
        self.k = 0
        self.wb = torch.empty(self.rb, w)
        self.tb = torch.empty(self.rb, w)
        self.pg = torch.empty(int(self.cap * 1.5), w)


def signo(sigb: np.ndarray) -> torch.Tensor:
    bits = np.unpackbits(sigb, axis=1, bitorder="little")
    return torch.from_numpy(bits).to(torch.float32).mul_(2.0).sub_(1.0)


def reunir_lote(est: Estado, ent: tuple, b: int, t: int) -> torch.Tensor:
    ids, sig = ent
    w = est.cfg.ancho
    n = ids.numel()
    E = est.bufE[est.k % len(est.bufE)]
    est.k += 1
    sigb = sig.numpy().view(np.uint8).reshape(n, 8)
    rb = est.rb
    for a in range(0, n, rb):
        z = min(n, a + rb)
        m = z - a
        wb, tb = est.wb[:m], est.tb[:m]
        torch.index_select(est.tabla.q, 0, ids[a:z], out=wb)
        torch.abs(wb, out=tb)
        s16 = (tb.amax(dim=-1) / 127.0).clamp_min(cuant.SUELO).to(torch.float16)
        s32 = s16.float().unsqueeze(-1)
        torch.div(wb, s32, out=tb)
        torch.round(tb, out=tb)
        tb.clamp_(-127, 127).add_(0.0).mul_(s32)
        tb.view(m, w // 64, 64).mul_(signo(sigb[a:z]).unsqueeze(1))
        E[a:z].copy_(tb)
    return E[:n].view(b, t, est.cfg.ancho_e)


def _segmentos(idn: np.ndarray) -> tuple:
    n = idn.size
    order = np.argsort(idn, kind="stable")
    sid = idn[order]
    es_ini = np.empty(n, dtype=bool)
    es_ini[0] = True
    np.not_equal(sid[1:], sid[:-1], out=es_ini[1:])
    starts = np.flatnonzero(es_ini)
    segid = np.cumsum(es_ini) - 1
    ends = np.append(starts, n)
    return order, sid[starts], es_ini, starts, segid, ends


def _sumar(est: Estado, nu: int, es_ini, starts, segid, ends, out: torch.Tensor, fuente) -> None:
    ub = max(64, int(est.rb * 0.85))
    for u0 in range(0, nu, ub):
        u1 = min(nu, u0 + ub)
        a, z = int(ends[u0]), int(ends[u1])
        gf = fuente(a, z)
        ob = out[u0:u1]
        torch.index_select(gf, 0, torch.from_numpy(starts[u0:u1] - a), out=ob)
        dp = np.flatnonzero(~es_ini[a:z])
        if dp.size:
            ob.index_add_(0, torch.from_numpy(segid[a:z][dp] - u0), gf.index_select(0, torch.from_numpy(dp)))


def agregar(est: Estado, ent: tuple, grad: torch.Tensor) -> tuple:
    ids, sig = ent
    w = est.cfg.ancho
    n = ids.numel()
    order, uid, es_ini, starts, segid, ends = _segmentos(ids.numpy())
    nu = uid.size
    gref = grad.reshape(-1, w)
    sigb = sig.numpy().view(np.uint8).reshape(n, 8)
    orderT = torch.from_numpy(order)

    def fuente(a: int, z: int) -> torch.Tensor:
        gf = gref.index_select(0, orderT[a:z]).float()
        torch.nan_to_num_(gf)
        gf.view(z - a, w // 64, 64).mul_(signo(sigb[order[a:z]]).unsqueeze(1))
        return gf.div_(est.mundo)

    out = est.inter.gr[est.rango]
    _sumar(est, nu, es_ini, starts, segid, ends, out, fuente)
    return torch.from_numpy(uid), out[:nu]


def publicar(est: Estado, uid: torch.Tensor, lim: torch.Tensor) -> None:
    i = est.inter
    n = uid.numel()
    if n > i.cap:
        raise RuntimeError(f"qf2_memhost: {n} filas unicas superan la capacidad {i.cap}")
    i.ids[i.rango, :n] = uid
    i.cab[i.rango, 0] = n
    i.cab[i.rango, 1:] = torch.searchsorted(uid, lim)


def reunir(est: Estado) -> tuple:
    i, cap, w = est.inter, est.inter.cap, est.cfg.ancho
    cab, idsn = i.cab.numpy(), i.ids.numpy()
    idl, fl = [], []
    for s in range(est.mundo):
        a, b = int(cab[s, 1 + est.rango]), int(cab[s, 2 + est.rango])
        if b > a:
            idl.append(idsn[s, a:b])
            fl.append(np.arange(s * cap + a, s * cap + b))
    if not idl:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, w)
    order, uid, es_ini, starts, segid, ends = _segmentos(np.concatenate(idl))
    nu = uid.size
    if nu > est.pg.size(0):
        est.pg = torch.empty(int(nu * 1.3), w)
    grflat = i.gr.view(-1, w)
    flo = torch.from_numpy(np.concatenate(fl)[order])
    _sumar(est, nu, es_ini, starts, segid, ends, est.pg, lambda a, z: grflat.index_select(0, flo[a:z]))
    return torch.from_numpy(uid), est.pg[:nu]


@torch.no_grad()
def optimizar(est: Estado, pid: torch.Tensor, pg: torch.Tensor, lr: float, lrm: float) -> None:
    t = est.tabla
    beta, eps = optm.BETA, optm.EPS
    c = 1.0 - optm.DECAY * lrm
    nu = pid.numel()
    for a in range(0, nu, est.rb):
        z = min(nu, a + est.rb)
        ib, g = pid[a:z], pg[a:z]
        w = t.q.index_select(0, ib)
        acc = t.acc.index_select(0, ib)
        n = t.cnt.index_select(0, ib).float() + 1.0
        acc = beta * acc + (1.0 - beta) * g.pow(2).mean(dim=-1)
        ahat = acc / (1.0 - beta ** n)
        tmp = g * lr
        tmp.div_((ahat.sqrt() + eps).unsqueeze(-1))
        w.mul_(c).sub_(tmp)
        t.q.index_copy_(0, ib, w)
        t.acc.index_copy_(0, ib, acc)
        t.cnt.index_copy_(0, ib, n.clamp_(max=tab.CNT_TOPE).to(torch.int16))

"""EMB: NKI 0.5.0 kernel dW fp32 [rows + 128, D] = scatter-add of dy [n, D] over ids [n] (port of the F1 EMBKF kernel).

dW is zeroed per program, then the ids are walked in tiles of 128 (nl.sequential_range: tiles RMW the same rows). Inside a tile the
duplicates are summed on the PE (equality matrix x dy into PSUM fp32); only the first occurrence of an id keeps its real row and each
repeat goes to its own in-range garbage row rows + m (m = its row inside the tile), so every indirect RMW instruction (nisa.dma_compute,
fp32 add) sees 128 unique in-range indices. oob_mode=skip aborted on the chip (NRT 1203) and is not used. Grid [lnc]: program p writes
only its D/lnc columns, in chunks of at most 512 fp32 columns (one PSUM bank). The caller slices [:rows].
"""
from __future__ import annotations


class _SinNki:
    @staticmethod
    def jit(f):
        return f


try:
    import nki as _nki
    import nki.isa as nisa
    import nki.language as nl
    HAY_NKI = True
except Exception:
    _nki, nl, nisa = _SinNki(), None, None
    HAY_NKI = False

EMB_P = 128


@_nki.jit
def emb_dw_kernel(dy, idx, rows):
    n, d = dy.shape
    pid = nl.program_id(0)
    dh = d // nl.num_programs(0)
    c0 = pid * dh
    nch = (dh + 511) // 512
    cw = dh // nch
    tot = rows + EMB_P
    dw = nl.ndarray((tot, d), dtype=nl.float32, buffer=nl.shared_hbm)
    zero = nl.ndarray((EMB_P, dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    nfull = tot // EMB_P
    for r in nl.affine_range(nfull):
        nisa.dma_copy(dst=dw[r * EMB_P:(r + 1) * EMB_P, c0:c0 + dh], src=zero)
    if tot > nfull * EMB_P:
        nisa.dma_copy(dst=dw[nfull * EMB_P:tot, c0:c0 + dh], src=zero[0:tot - nfull * EMB_P, :])
    pos = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pos, pattern=[[1, EMB_P]], offset=0, channel_multiplier=-1)
    low = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low, data=pos, op0=nl.less, operand0=0.0)
    slot = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(slot, pattern=[[0, 1]], offset=rows, channel_multiplier=1)
    for t in nl.sequential_range(n // EMB_P):
        idc = nl.ndarray((EMB_P, 1), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idc, src=idx.ap(pattern=[[1, EMB_P], [1, 1]], offset=t * EMB_P))
        idr = nl.ndarray((EMB_P, EMB_P), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idr, src=idx.ap(pattern=[[0, EMB_P], [1, EMB_P]], offset=t * EMB_P))
        idcf = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idcf, src=idc)
        idrf = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idrf, src=idr)
        eq = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=eq, data=idrf, op0=nl.equal, operand0=idcf)
        prev = nl.ndarray((EMB_P, EMB_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prev, data1=eq, data2=low, op=nl.multiply)
        cnt = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cnt, op=nl.add, data=prev, axis=1)
        fol = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fol, data=cnt, op0=nl.minimum, operand0=1.0)
        gap = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gap, data1=slot, data2=idcf, op=nl.subtract)
        pen = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pen, data1=fol, data2=gap, op=nl.multiply)
        efff = nl.ndarray((EMB_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=efff, data1=idcf, data2=pen, op=nl.add)
        eff = nl.ndarray((EMB_P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eff, src=efff)
        eqm = nl.ndarray((EMB_P, EMB_P), dtype=dy.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eqm, src=eq)
        for c in nl.static_range(nch):
            cc = c0 + c * cw
            dyt = nl.ndarray((EMB_P, cw), dtype=dy.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=dyt, src=dy[t * EMB_P:(t + 1) * EMB_P, cc:cc + cw])
            ps = nl.ndarray((EMB_P, cw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=eqm, moving=dyt)
            acc = nl.ndarray((EMB_P, cw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=acc, src=ps)
            tgt = dw.ap(pattern=[[d, EMB_P], [1, cw]], offset=cc, vector_offset=eff, indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, acc], reduce_op=nl.add)
    return dw

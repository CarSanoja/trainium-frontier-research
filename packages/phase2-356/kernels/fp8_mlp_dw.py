from .fp8_mlp_comun import _DR, _FP8, _FP8G, _NT8, _P8, _nki, nisa, nl


def _dw8_fase(a, pk, sc, fase):
    M, D = a.shape
    KH = pk.shape[0]
    H = KH * _P8
    NS = M // (2 * _P8)
    KD = D // _P8
    ND = (D + _NT8 - 1) // _NT8
    out = nl.ndarray((H, D) if fase == 0 else (D, H), dtype=a.dtype, buffer=nl.shared_hbm)
    scs = nl.load(sc)
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        nprg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
    else:
        nprg = 1
        pid = 0
    KHP = KH // nprg
    bs = nl.ndarray((_P8, NS, 2, D), dtype=_FP8 if fase == 0 else _FP8G, buffer=nl.sbuf)
    for s in nl.static_range(NS):
        t = nl.ndarray((_P8, 2, D), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t, a.ap(pattern=[[2 * D, _P8], [D, 2], [1, D]], offset=s * 2 * _P8 * D))
        if fase == 0:
            nisa.tensor_scalar(bs[:, s, :, :], t, nl.minimum, 240.0, op1=nl.maximum, operand1=-240.0)
        else:
            nisa.tensor_scalar(bs[:, s, :, :], t, nl.multiply, scs[:, 0:1])
    for i in nl.affine_range(KHP):
        ht = pid * KHP + i
        st = nl.ndarray((_P8, NS, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(st, pk[ht, :, :, :, :])
        sv = st.view(_FP8G if fase == 0 else _FP8)
        pss = nl.ndarray((_P8, ND, _NT8), dtype=nl.float32, buffer=nl.psum)
        for s in nl.static_range(NS):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                nisa.nc_matmul(pss[:, nd, 0:w], sv[:, s, :, :], bs[:, s, :, nd * _NT8:nd * _NT8 + w], accumulate=(s > 0), perf_mode=_DR)
        o = nl.ndarray((_P8, D), dtype=a.dtype, buffer=nl.sbuf)
        for nd in nl.static_range(ND):
            w = min(_NT8, D - nd * _NT8)
            nisa.tensor_scalar(o[:, nd * _NT8:nd * _NT8 + w], pss[:, nd, 0:w], nl.multiply, scs[:, 1 + fase:2 + fase])
        if fase == 0:
            nl.store(out[ht * _P8:(ht + 1) * _P8, :], o)
        else:
            for kd in nl.static_range(KD):
                pt = nl.ndarray((_P8, _P8), dtype=a.dtype, buffer=nl.psum)
                nisa.nc_transpose(pt, o[:, kd * _P8:(kd + 1) * _P8], engine=nisa.engine.tensor)
                ot = nl.ndarray((_P8, _P8), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(ot, pt)
                nl.store(out[kd * _P8:(kd + 1) * _P8, ht * _P8:(ht + 1) * _P8], ot)
    return out


@_nki.jit
def mlp_fp8_dw8_x(x, dpk, sc):
    return _dw8_fase(x, dpk, sc, 0)


@_nki.jit
def mlp_fp8_dw8_y(dy, hpk, sc):
    return _dw8_fase(dy, hpk, sc, 1)

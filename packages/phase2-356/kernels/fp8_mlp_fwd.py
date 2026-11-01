from .fp8_mlp_comun import _DR, _FEP, _FP8, _H_CLAMP, _KAM, _NT8, _P8, _SCALE_H, _TB8, _nki, nisa, nl


@_nki.jit
def mlp_fp8_fwd_r(x, w1, w2, sc):
    M, D = x.shape
    scs = nl.load(sc)
    if _FEP:
        tope = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(tope, scs[:, 0:1], nl.multiply, _H_CLAMP / _SCALE_H)
        qe = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(qe, scs[:, 3:4], scs[:, 3:4], nl.multiply)
    _, H = w1.shape
    out = nl.ndarray((M, D), dtype=x.dtype, buffer=nl.shared_hbm)
    rT = nl.ndarray((H, M), dtype=x.dtype, buffer=nl.shared_hbm)
    KD = D // _P8
    NCD = KD // 2
    TLD = (KD % 2) * (_P8 // 2)
    KH = H // _P8
    NCH = KH // 2
    ND = (D + _NT8 - 1) // _NT8
    w1s = nl.ndarray((_P8, NCD, 2, H), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCD):
        for j in nl.static_range(2):
            t = nl.load(w1[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w1s[:, c, j, :], t, nl.multiply, scs[:, 0:1])
    if TLD:
        w1t = nl.ndarray((TLD, 2, H), dtype=_FP8, buffer=nl.sbuf)
        for j in nl.static_range(2):
            t = nl.load(w1[NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD, :])
            nisa.tensor_scalar(w1t[:, j, :], t, nl.multiply, scs[0:TLD, 0:1])
    w2s = nl.ndarray((_P8, NCH, 2, D), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCH):
        for j in nl.static_range(2):
            t = nl.load(w2[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w2s[:, c, j, :], t, nl.multiply, scs[:, 1:2])
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        nprg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
    else:
        nprg = 1
        pid = 0
    NB = (M // _TB8) // nprg
    for mb0 in nl.affine_range(NB):
        mb = pid * NB + mb0
        xs = nl.ndarray((_P8, NCD, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
        for c in nl.static_range(NCD):
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(x[mb * _TB8:(mb + 1) * _TB8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                    nisa.tensor_copy(xs[:, c, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(x[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                        nisa.tensor_copy(xs[:, c, j, t * _P8:(t + 1) * _P8], xt)
        if TLD:
            xtl = nl.ndarray((TLD, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(x[mb * _TB8:(mb + 1) * _TB8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                    nisa.tensor_copy(xtl[:, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(x[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                        nisa.tensor_copy(xtl[:, j, t * _P8:(t + 1) * _P8], xt)
        hs = nl.ndarray((_P8, NCH, 2, _TB8), dtype=_FP8, buffer=nl.sbuf)
        for c in nl.static_range(NCH):
            for j in nl.static_range(2):
                ht = 2 * c + j
                ps = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.psum)
                for cd in nl.static_range(NCD):
                    nisa.nc_matmul(ps, w1s[:, cd, :, ht * _P8:(ht + 1) * _P8], xs[:, cd, :, :], accumulate=(cd > 0), perf_mode=_DR)
                if TLD:
                    nisa.nc_matmul(ps, w1t[:, :, ht * _P8:(ht + 1) * _P8], xtl, accumulate=True, perf_mode=_DR)
                if _FEP:
                    hc = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if _FEP == 2:
                        nisa.tensor_scalar(hc, ps, nl.maximum, 0.0, op1=nl.minimum, operand1=tope, engine=nisa.engine.scalar)
                    else:
                        nisa.tensor_scalar(hc, ps, nl.maximum, 0.0, op1=nl.minimum, operand1=tope)
                    nisa.scalar_tensor_tensor(dst=hs[:, c, j, :], data=hc, op0=nl.multiply, operand0=qe, op1=nl.multiply, operand1=hc)
                    nl.store(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], hc)
                else:
                    hb = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(hb, nl.relu, ps, scale=scs[:, 3:4])
                    hc = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(hc, hb, nl.minimum, _H_CLAMP)
                    nisa.tensor_tensor(hs[:, c, j, :], hc, hc, nl.multiply)
                    nl.store(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], hc)
        for t in nl.static_range(_TB8 // _P8):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                ps2 = nl.ndarray((_P8, w), dtype=nl.float32, buffer=nl.psum)
                for c in nl.static_range(NCH):
                    nisa.nc_matmul(ps2, hs[:, c, :, t * _P8:(t + 1) * _P8], w2s[:, c, :, nd * _NT8:nd * _NT8 + w], accumulate=(c > 0), perf_mode=_DR)
                o = nl.ndarray((_P8, w), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(o, ps2, nl.multiply, scs[:, 2:3])
                nl.store(out[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, nd * _NT8:nd * _NT8 + w], o)
    return out, rT

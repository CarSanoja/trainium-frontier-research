from .fp8_mlp_comun import (MLP_DW8, MLP_DW8_EVSB, MLP_RECORTE_GRAD, _DR, _EPI, _FEP, _FP8, _FP8G, _H_CLAMP, _KAM, _NT8, _P8,
                            _S8G, _SCALE_H, _TB8, _nki, nisa, nl)


@_nki.jit
def mlp_fp8_bwd_act(dy, w2T, w1T, rT, sc):
    M, D = dy.shape
    _, H = w2T.shape
    dpreT = nl.ndarray((H, M), dtype=dy.dtype, buffer=nl.shared_hbm)
    dx = nl.ndarray((M, D), dtype=dy.dtype, buffer=nl.shared_hbm)
    scs = nl.load(sc)
    if _EPI:
        kgs = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
        if _FEP:
            eun = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(eun, scs[:, 3:4], scs[:, 4:5], nl.multiply)
            ese = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(ese, eun, nl.multiply, _SCALE_H)
            kg0 = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(kg0, scs[:, 2:3], scs[:, 4:5], nl.multiply)
            nisa.tensor_tensor(kgs, kg0, ese, nl.multiply)
            qe = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(qe, ese, ese, nl.multiply)
            if MLP_RECORTE_GRAD:
                tpb = nl.ndarray((_P8, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(tpb, scs[:, 1:2], nl.multiply, _H_CLAMP / _SCALE_H)
        else:
            nisa.tensor_tensor(kgs, scs[:, 2:3], scs[:, 4:5], nl.multiply)
    KD = D // _P8
    NCD = KD // 2
    TLD = (KD % 2) * (_P8 // 2)
    KH = H // _P8
    NCH = KH // 2
    ND = (D + _NT8 - 1) // _NT8
    if MLP_DW8:
        dpk = nl.ndarray((KH, _P8, M // (2 * _P8), 2, _P8), dtype=nl.uint8, buffer=nl.shared_hbm)
        hpk = nl.ndarray((KH, _P8, M // (2 * _P8), 2, _P8), dtype=nl.uint8, buffer=nl.shared_hbm)
    w2s = nl.ndarray((_P8, NCD, 2, H), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCD):
        for j in nl.static_range(2):
            t = nl.load(w2T[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w2s[:, c, j, :], t, nl.multiply, scs[:, 0:1])
    if TLD:
        w2t = nl.ndarray((TLD, 2, H), dtype=_FP8, buffer=nl.sbuf)
        for j in nl.static_range(2):
            t = nl.load(w2T[NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD, :])
            nisa.tensor_scalar(w2t[:, j, :], t, nl.multiply, scs[0:TLD, 0:1])
    w1s = nl.ndarray((_P8, NCH, 2, D), dtype=_FP8, buffer=nl.sbuf)
    for c in nl.static_range(NCH):
        for j in nl.static_range(2):
            t = nl.load(w1T[(2 * c + j) * _P8:(2 * c + j + 1) * _P8, :])
            nisa.tensor_scalar(w1s[:, c, j, :], t, nl.multiply, scs[:, 1:2])
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
        dys = nl.ndarray((_P8, NCD, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
        for c in nl.static_range(NCD):
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(dy[mb * _TB8:(mb + 1) * _TB8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                    if _S8G:
                        nisa.tensor_scalar(dys[:, c, j, :], xt, nl.multiply, scs[:, 4:5])
                    else:
                        nisa.tensor_copy(dys[:, c, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(dy[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, (2 * c + j) * _P8:(2 * c + j + 1) * _P8])
                        if _S8G:
                            nisa.tensor_scalar(dys[:, c, j, t * _P8:(t + 1) * _P8], xt, nl.multiply, scs[:, 4:5])
                        else:
                            nisa.tensor_copy(dys[:, c, j, t * _P8:(t + 1) * _P8], xt)
        if TLD:
            dytl = nl.ndarray((TLD, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
            for j in nl.static_range(2):
                if _KAM:
                    xt = nl.load_transpose2d(dy[mb * _TB8:(mb + 1) * _TB8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                    if _S8G:
                        nisa.tensor_scalar(dytl[:, j, :], xt, nl.multiply, scs[0:TLD, 4:5])
                    else:
                        nisa.tensor_copy(dytl[:, j, :], xt)
                else:
                    for t in nl.static_range(_TB8 // _P8):
                        xt = nl.load_transpose2d(dy[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, NCD * 2 * _P8 + j * TLD:NCD * 2 * _P8 + (j + 1) * TLD])
                        if _S8G:
                            nisa.tensor_scalar(dytl[:, j, t * _P8:(t + 1) * _P8], xt, nl.multiply, scs[0:TLD, 4:5])
                        else:
                            nisa.tensor_copy(dytl[:, j, t * _P8:(t + 1) * _P8], xt)
        dps = nl.ndarray((_P8, NCH, 2, _TB8), dtype=_FP8G, buffer=nl.sbuf)
        for c in nl.static_range(NCH):
            for j in nl.static_range(2):
                ht = 2 * c + j
                ps = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.psum)
                for cd in nl.static_range(NCD):
                    nisa.nc_matmul(ps, w2s[:, cd, :, ht * _P8:(ht + 1) * _P8], dys[:, cd, :, :], accumulate=(cd > 0), perf_mode=_DR)
                if TLD:
                    nisa.nc_matmul(ps, w2t[:, :, ht * _P8:(ht + 1) * _P8], dytl, accumulate=True, perf_mode=_DR)
                rt = nl.load(rT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8])
                if _EPI == 1:
                    rtm = rt
                    if MLP_RECORTE_GRAD:
                        rtm = nl.ndarray((_P8, _TB8), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.scalar_tensor_tensor(dst=rtm, data=rt, op0=nl.less, operand0=tpb, op1=nl.multiply, operand1=rt)
                    nisa.scalar_tensor_tensor(dst=dps[:, c, j, :], data=ps, op0=nl.multiply, operand0=kgs, op1=nl.multiply, operand1=rtm)
                elif _EPI == 2:
                    g = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(g, ps, rt, nl.multiply)
                    nisa.tensor_scalar(dps[:, c, j, :], g, nl.multiply, kgs)
                else:
                    g = nl.ndarray((_P8, _TB8), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(g, ps, rt, nl.multiply)
                    gb = nl.ndarray((_P8, _TB8), dtype=dy.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(gb, g, nl.multiply, scs[:, 2:3])
                    if not MLP_DW8:
                        nl.store(dpreT[ht * _P8:(ht + 1) * _P8, mb * _TB8:(mb + 1) * _TB8], gb)
                    if _S8G:
                        nisa.tensor_scalar(dps[:, c, j, :], gb, nl.multiply, scs[:, 4:5])
                    else:
                        nisa.tensor_copy(dps[:, c, j, :], gb)
                if MLP_DW8:
                    h8 = nl.ndarray((_P8, _TB8), dtype=_FP8, buffer=nl.sbuf)
                    if _FEP:
                        nisa.scalar_tensor_tensor(dst=h8, data=rt, op0=nl.multiply, operand0=qe, op1=nl.multiply, operand1=rt)
                    else:
                        nisa.tensor_tensor(h8, rt, rt, nl.multiply)
                    if _KAM:
                        pk = nl.ndarray((_P8, 4, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        src = dps[:, c, j, :]
                        sv = src.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        sh = h8.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, 2 + k, :], sh[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 4, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 4, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(dpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk[:, 0:2, :, :])
                        nl.store(hpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk[:, 2:4, :, :])
                    else:
                        pk = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        src = dps[:, c, j, :]
                        sv = src.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 2, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(dpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk)
                        pk = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.psum)
                        sv = h8.view(nl.bfloat16)
                        for k in nl.static_range(2):
                            nisa.nc_transpose(pk[:, k, :], sv[:, k * _P8:(k + 1) * _P8], engine=nisa.engine.tensor)
                        tk = nl.ndarray((_P8, 2, 2, _P8), dtype=nl.uint8, buffer=nl.sbuf)
                        if MLP_DW8_EVSB:
                            tb = nl.ndarray((_P8, 2, _P8), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_copy(tb, pk)
                            nisa.tensor_copy(tk, tb.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        else:
                            nisa.tensor_copy(tk, pk.view(nl.uint8).reshape_dim(2, (_P8, 2)).permute((0, 1, 3, 2)))
                        nl.store(hpk[ht, :, 2 * mb:2 * mb + 2, :, :], tk)
        for t in nl.static_range(_TB8 // _P8):
            for nd in nl.static_range(ND):
                w = min(_NT8, D - nd * _NT8)
                ps2 = nl.ndarray((_P8, w), dtype=nl.float32, buffer=nl.psum)
                for c in nl.static_range(NCH):
                    nisa.nc_matmul(ps2, dps[:, c, :, t * _P8:(t + 1) * _P8], w1s[:, c, :, nd * _NT8:nd * _NT8 + w], accumulate=(c > 0), perf_mode=_DR)
                o = nl.ndarray((_P8, w), dtype=dy.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(o, ps2, nl.multiply, scs[:, 3:4])
                nl.store(dx[mb * _TB8 + t * _P8:mb * _TB8 + (t + 1) * _P8, nd * _NT8:nd * _NT8 + w], o)
    if MLP_DW8:
        return dpk, hpk, dx
    return dpreT, dx

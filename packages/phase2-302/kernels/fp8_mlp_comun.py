from __future__ import annotations

import os


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

MLP_FP8_LNC = 2
MLP_FP8_BWDK = os.environ.get("QF2_FP8_BWDK", "1") != "0"
MLP_FP8_EAGER = False
MLP_FP8_DYSCALE = int(os.environ.get("QF2_FP8_DYSCALE", "16"))
MLP_DW8 = True
MLP_DW8_EVSB = False
MLP_EPIPASS = 1
MLP_FWDEPI = 1
KA_MLP = 1
MLP_RECORTE_GRAD = os.environ.get("QF2_FP8_RECORTE_GRAD", "1") != "0"

_P8, _NT8, _TB8 = 128, 512, 512
_FP8 = nl.float8_e4m3 if HAY_NKI else None
_FP8G = nl.float8_e5m2 if HAY_NKI else None
_SCALE_H = 0.25
_H_CLAMP = 15.4375
_DR = nisa.matmul_perf_mode.double_row if HAY_NKI else None
_S8G = float(2 ** MLP_FP8_DYSCALE) if MLP_FP8_DYSCALE > 0 else 0.0
_EPI = MLP_EPIPASS if MLP_DW8 and _S8G else 0
_FEP = MLP_FWDEPI if _EPI == 1 else 0
_KAM = KA_MLP == 1
if MLP_RECORTE_GRAD and not (_EPI == 1 and _FEP == 1):
    raise ValueError("MLP_RECORTE_GRAD necesita MLP_EPIPASS=1 y MLP_FWDEPI=1 (el recorte se enmascara en la vuelta de esa rama)")


def formas_ok(M: int, D: int, H: int, rejilla: int = MLP_FP8_LNC) -> bool:
    return (D % _P8 == 0 and H % (2 * _P8) == 0 and M % (_TB8 * rejilla) == 0 and M % (2 * _P8) == 0
            and (H // _P8) % rejilla == 0)


RESUMEN = (f"nki={int(HAY_NKI)} lnc={MLP_FP8_LNC} bwdk={int(MLP_FP8_BWDK)} eager={int(MLP_FP8_EAGER)} dyscale=2^{MLP_FP8_DYSCALE} "
           f"dw8={int(MLP_DW8)} epipass={_EPI} fwdepi={_FEP} ka_mlp={int(_KAM)} tb8={_TB8} h_clamp={_H_CLAMP} "
           f"recorte_grad={int(MLP_RECORTE_GRAD)}")

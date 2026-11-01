"""Estados entrantes de un arbol afin; VJP con todos los ATen visibles."""

from __future__ import annotations

import torch


def pares(t):
    return t.reshape(-1, 2, *t.shape[-2:]).unbind(1)


def intercalar(a, b):
    return torch.stack((a, b), 1).flatten(0, 1)


class ArbolAfin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.ndim != 5 or b.ndim != 5 or a.shape[:4] != b.shape[:4] or a.shape[-2] != a.shape[-1]:
            raise ValueError("Formas afines B,H,T,K,K y B,H,T,K,V")
        if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype or a.device != b.device:
            raise ValueError("Estado afin requiere FP32/FP64 y mismo dispositivo")
        lote, cabezas, t, k, _ = a.shape
        if t < 1:
            raise ValueError("El arbol necesita T>=1")
        v = b.shape[-1]
        n = 1 << (t - 1).bit_length()
        if n != t:
            identidad = torch.eye(k, dtype=a.dtype, device=a.device).expand(lote, cabezas, n-t, k, k)
            a = torch.cat((a, identidad), 2)
            b = torch.cat((b, b.new_zeros(lote, cabezas, n-t, k, v)), 2)
        aa = [a.reshape(-1, k, k)]
        bb = [b.reshape(-1, k, v)]
        m = n
        while m > 1:
            izquierda, derecha = pares(aa[-1])
            bi, bd = pares(bb[-1])
            aa.append(torch.bmm(derecha, izquierda))
            bb.append(torch.bmm(derecha, bi) + bd)
            m //= 2
        niveles = len(aa)
        pp = [None] * niveles
        pp[-1] = b.new_zeros(lote*cabezas, k, v)
        for nivel in range(niveles-2, -1, -1):
            izquierda, _ = pares(aa[nivel])
            bi, _ = pares(bb[nivel])
            derecho = torch.bmm(izquierda, pp[nivel+1]) + bi
            pp[nivel] = intercalar(pp[nivel+1], derecho)
        ctx.save_for_backward(*aa, *bb, *pp)
        ctx.forma = (lote, cabezas, t, k, v, n, niveles)
        return pp[0].reshape(lote, cabezas, n, k, v)[:, :, :t]

    @staticmethod
    def backward(ctx, grad):
        lote, cabezas, t, k, v, n, niveles = ctx.forma
        aa = ctx.saved_tensors[:niveles]
        bb = ctx.saved_tensors[niveles:2*niveles]
        pp = ctx.saved_tensors[2*niveles:]
        da = [torch.zeros_like(x) for x in aa]
        db = [torch.zeros_like(x) for x in bb]
        if n != t:
            grad = torch.cat((grad, grad.new_zeros(lote, cabezas, n-t, k, v)), 2)
        dp = grad.reshape(-1, k, v)
        for nivel in range(niveles-1):
            gi, gd = pares(dp)
            ai, _ = pares(aa[nivel])
            di, _ = pares(da[nivel])
            ei, _ = pares(db[nivel])
            di.add_(torch.bmm(gd, pp[nivel+1].transpose(-2, -1)))
            ei.add_(gd)
            dp = gi + torch.bmm(ai.transpose(-2, -1), gd)
        for nivel in range(niveles-2, -1, -1):
            ai, ad = pares(aa[nivel])
            bi, _ = pares(bb[nivel])
            dai, dad = pares(da[nivel])
            dbi, dbd = pares(db[nivel])
            dad.add_(torch.bmm(da[nivel+1], ai.transpose(-2, -1)))
            dad.add_(torch.bmm(db[nivel+1], bi.transpose(-2, -1)))
            dai.add_(torch.bmm(ad.transpose(-2, -1), da[nivel+1]))
            dbi.add_(torch.bmm(ad.transpose(-2, -1), db[nivel+1]))
            dbd.add_(db[nivel+1])
        return (da[0].reshape(lote, cabezas, n, k, k)[:, :, :t],
                db[0].reshape(lote, cabezas, n, k, v)[:, :, :t])


def estados_entrantes(a, b):
    return ArbolAfin.apply(a, b)

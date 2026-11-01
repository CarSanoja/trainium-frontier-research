from __future__ import annotations

import torch

FRACCIONES = (0.9, 0.925, 0.95, 0.975)
ESTADOS = len(FRACCIONES) + 1


class Promedio:
    def __init__(self, params, en_host: bool = False, antes=None):
        self.params = list(params)
        self.antes = antes
        self.en_host = en_host
        self.suma = [torch.zeros(p.shape, dtype=torch.float32, device="cpu" if en_host else p.device) for p in self.params]
        with torch.no_grad():
            for s, p in zip(self.suma, self.params):
                s.copy_(p)
                s.add_(p.to(s.device))
                s.div_(float(ESTADOS))
                s.zero_()
        self.pendientes = list(FRACCIONES)
        self.pasos: list[int] = []

    def descripcion(self) -> str:
        n = sum(s.numel() for s in self.suma)
        return f"qf2_promedio: fracciones={','.join(map(str, FRACCIONES))},1.0 tensores={len(self.suma)} destino={'host' if self.en_host else 'dispositivo'} acumulador_fp32_MB={4 * n / 2**20:.1f}"

    @torch.no_grad()
    def _sumar(self, pasos: int, veces: int) -> None:
        if self.antes is not None:
            self.antes()
        for s, p in zip(self.suma, self.params):
            q = p.to(s.device)
            for _ in range(veces):
                s.add_(q)
        self.pasos.extend([pasos] * veces)

    def tras_paso(self, pasos: int, progreso: float, final: bool) -> None:
        cruzadas = sum(1 for f in self.pendientes if progreso >= f)
        self.pendientes = self.pendientes[cruzadas:]
        veces = cruzadas
        if final:
            veces += len(self.pendientes) + 1
            self.pendientes = []
        if veces:
            self._sumar(pasos, veces)

    @torch.no_grad()
    def instalar(self) -> list[int]:
        if len(self.pasos) != ESTADOS:
            raise RuntimeError(f"qf2_promedio: {len(self.pasos)} estados sumados, se esperaban {ESTADOS}")
        for s, p in zip(self.suma, self.params):
            s.div_(float(ESTADOS))
            p.copy_(s)
        self.suma = []
        return self.pasos

"""ReplicasMuonBF16 con maestroFP32 persistente solo en cada dueno."""

from __future__ import annotations

import torch

from qf2_franjas import RepartoFranjas


class RepartoMaestro(RepartoFranjas):
    def __init__(self, grupos, rango: int, mundo: int, franjas: int):
        if any(g["kind"] != "muon" for g in grupos):
            raise ValueError("El maestro solo admite gruposMuon")
        if any(p.dtype != torch.float32 for g in grupos for p in g["params"]):
            raise ValueError("Inicializar el maestro desde pesosFP32 originales")
        super().__init__(grupos, rango, mundo, franjas)
        with torch.no_grad():
            for _grupo, _forma, params, w, _ini, _fin in self._propios():
                for k, p in enumerate(params):
                    w[k].copy_(p)
            for p in self.params:
                p.data = p.data.to(torch.bfloat16)

    def descripcion(self) -> str:
        return f"{super().descripcion()} maestro_fp32_persistente=1 replicas=bf16"

    @torch.no_grad()
    def paso(self, opt, bf16: bool, rs_bf16: bool = False) -> None:
        if not bf16 or rs_bf16:
            raise ValueError("El maestro exige gatherBF16 y reduccionFP32")
        self.franjas.abrir()
        self.franjas.reducir()
        for grupo, forma, params, w, ini, fin in self._propios():
            g = self.grad[ini:fin].view(len(params), *forma)
            opt._muon_owned(grupo, forma, g, w, opt.state[grupo["params"][0]])
        self.franjas.reunir(True)
        self.franjas.cerrar()

    @torch.no_grad()
    def restaurar(self) -> None:
        self.franjas.abrir()
        self.franjas.reunir(True)
        self.franjas.cerrar()


def crear(grupos, rango: int, mundo: int, franjas: int, activo: bool = False):
    clase = RepartoMaestro if activo else RepartoFranjas
    return clase(grupos, rango, mundo, franjas)


def leer(valor: str) -> bool:
    if valor not in ("", "0", "1"):
        raise ValueError("QF2_MUON_BF16 admite vacio,0 o1")
    return valor == "1"


def matrices(model):
    salida = []
    for block in model.transformer.h:
        salida.extend(block.attn.parameters())
        salida.extend(block.mlp.parameters())
    if model.config.memoria:
        salida.append(model.memoria_proy.weight)
    if model.config.memhost:
        salida.extend(model.memhost_sitios.matrices())
    if model.pkm is not None:
        salida.extend([model.pkm.consulta, model.pkm.salida])
    if model.resumen is not None:
        salida.append(model.resumen.proy.weight)
    if model.mos is not None:
        salida.append(model.mos.w2)
    if model.delta is not None:
        salida.extend(model.delta.matrices())
    return salida


def validar(mundo, dtype, muon, franjas, gather, rs, opt, incompatibles):
    if mundo <= 1:
        raise ValueError("QF2_MUON_BF16 exige entrenamiento distribuido")
    if dtype != torch.bfloat16 or not muon or not franjas or not gather or rs or opt:
        raise ValueError("QF2_MUON_BF16 exige computeBF16, Muon, franjas, gatherBF16, reduceFP32 y opt base")
    if any(incompatibles):
        raise ValueError("QF2_MUON_BF16 no admite fork, reanudar, puntos ni promedio")


@torch.no_grad()
def preparar_carga(model, state):
    if not model.config.muon_bf16:
        return
    ids = {id(p) for p in matrices(model)}
    for nombre, p in model.named_parameters():
        if id(p) in ids:
            if nombre not in state or state[nombre].dtype != torch.bfloat16:
                raise ValueError(f"Checkpoint MuonBF16 incompatible: {nombre}")
            p.data = p.data.to(torch.bfloat16)

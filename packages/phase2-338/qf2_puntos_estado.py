"""What a resumable point holds, besides the loader: model, optimizer, RNG and the small QF2 objects (QF2_PUNTOS_S).

Model. Without QF2_COM_GATHER_BF16 every rank holds the same fp32 parameters, so rank 0 alone writes them. With it, each
rank keeps the fp32 master of what it owns and the bf16 rounding of the rest. Rank 0 then writes the bf16 rounding of its
parameters as a shared base, and every rank writes its own values on ranges that cover every element that is not bf16
exact. bf16(base) is what any rank holds wherever its value is bf16 exact, so base plus own ranges rebuilds each rank's
parameters bit for bit, whoever owns what. Ranges merge gaps below HUECO elements: an own value is always right.
Optimizer: every rank writes the state of its shard, keyed by the position of the parameter in param_groups.
"""
from __future__ import annotations

import torch

HUECO = 4096
CAMPOS = {"guarda": ("saltos", "seguidos"), "cola": ("forma",), "promedio": ("pendientes", "pasos")}


def tramos(plano: torch.Tensor) -> list:
    malos = (plano != plano.bfloat16().float()).nonzero().flatten()
    if malos.numel() == 0:
        return []
    cortes = ((malos[1:] - malos[:-1]) > HUECO).nonzero().flatten()
    inicios = [int(malos[0])] + (malos[cortes + 1]).tolist()
    fines = (malos[cortes]).tolist() + [int(malos[-1])]
    return [[a, plano[a:b + 1].clone()] for a, b in zip(inicios, fines)]


@torch.no_grad()
def modelo_base(modelo, bf16: bool) -> dict:
    return {k: v.detach().to("cpu", copy=True).to(torch.bfloat16 if bf16 else v.dtype) for k, v in modelo.state_dict().items()}


@torch.no_grad()
def modelo_propio(modelo) -> dict:
    return {k: tramos(v.detach().to("cpu", copy=True).float().reshape(-1)) for k, v in modelo.state_dict().items()}


@torch.no_grad()
def poner_modelo(modelo, base: dict, propio: dict | None) -> None:
    estado = {}
    for k, t in modelo.state_dict().items():
        v = base[k].to(t.dtype)
        if propio is not None:
            plano = v.reshape(-1).clone()
            for a, valores in propio[k]:
                plano[a:a + valores.numel()] = valores
            v = plano.view(t.shape)
        estado[k] = v
    modelo.load_state_dict(estado, strict=True)


def parametros(opt) -> list:
    return [p for g in opt.param_groups for p in g["params"]]


def estado_opt(opt) -> dict:
    salida = {}
    for i, p in enumerate(parametros(opt)):
        s = opt.state.get(p)
        if s:
            salida[i] = {k: {"t": v.detach().to("cpu", copy=True), "cpu": v.device.type == "cpu"} if torch.is_tensor(v) else v
                         for k, v in s.items()}
    return salida


def poner_opt(opt, estado: dict) -> None:
    params = parametros(opt)
    if estado and max(estado) >= len(params):
        raise ValueError(f"qf2_puntos: optimizer state for parameter {max(estado)} but only {len(params)} parameters")
    for i, s in estado.items():
        p = params[int(i)]
        opt.state[p] = {k: (v["t"] if v["cpu"] else v["t"].to(p.device)) if isinstance(v, dict) else v for k, v in s.items()}


def estado_rng(device) -> dict:
    salida = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        salida["cuda"] = torch.cuda.get_rng_state(device)
    return salida


def poner_rng(estado: dict, device) -> None:
    torch.set_rng_state(estado["cpu"])
    if "cuda" in estado:
        torch.cuda.set_rng_state(estado["cuda"], device)


def estado_extras(extras: dict) -> dict:
    return {k: {c: getattr(o, c) for c in CAMPOS[k]} for k, o in extras.items() if o is not None}


def poner_extras(extras: dict, estado: dict) -> None:
    activos = {k for k, o in extras.items() if o is not None}
    if activos != set(estado):
        raise ValueError(f"qf2_puntos: point saved with {sorted(estado)} but this run has {sorted(activos)}")
    for k, campos in estado.items():
        for c, v in campos.items():
            setattr(extras[k], c, v)


@torch.no_grad()
def poner_promedio(promedio, suma: list) -> None:
    if len(suma) != len(promedio.suma):
        raise ValueError(f"qf2_puntos: {len(suma)} saved averaging tensors, run has {len(promedio.suma)}")
    for s, t in zip(promedio.suma, suma):
        s.copy_(t.to(s.device))

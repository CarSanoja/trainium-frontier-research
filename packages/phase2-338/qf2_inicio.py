"""QF2_INICIO_DESDE y QF2_BANCO_LRM: continuar una corrida desde un final.pt del kit (banco CORE bifurcado).

abrir() mapea el final.pt con torch.load(mmap=True) (qf2_memoria_anfitrion.abrir, la lectura de la evaluación) y exige
que su model_config sea el del modelo construido. cargar_tronco() carga en cada rango el model_state sin las tablas del
anfitrión con el mismo filtro estricto que load_for_eval_memhost (qf2_memhost_ckpt.cargar_resto), sobre el modelo de CPU,
antes de model.to(device) y de crear el optimizador, que empieza nuevo.

La única diferencia de model_config que abrir() admite es la vía del sitio 11 (QF2_MEMHOST_VIA_SITIO11): un modelo con
via=v11 desde un final.pt con la vía res (el 274 o el 293). Las claves son las mismas, porque memhost_sitios.key.11 y
val.11 tienen la misma forma [n_embd, S*W] en las dos vías: la puerta y la proyección que llevaban lo recuperado al
residual tras el bloque 11 pasan a llevarlo al valor de la atención del bloque 11, el tronco carga entero y nada empieza
a cero. Cualquier otra diferencia, también v11 -> res o res -> v, sigue siendo un error.

tablas() da el gancho que qf2_memhost_hilo.Memhost llama con su Tabla antes de arrancar el hilo de tablas, así que ni un
gather ni su plazo ESPERA_S (300 s) corren durante la carga. cargar_tablas() escribe en los maestros fp32 compartidos del
anfitrión las filas propias del rango (Tabla.propias) descuantizadas de memhost.q.weight y memhost.s.weight, pone a cero
acc y cnt (y m, v con el optimizador Adam) en esas filas, deja una marca y espera la de todos los rangos: ningún gather
del primer lote lee una fila a medio cargar. Las lecturas del entreno aplican ida_vuelta a los maestros, que sobre valores
ya descuantizados es la identidad, así que el primer paso ve exactamente lo que la evaluación lee del checkpoint.

leer_lrm('a:b') y multiplicador(): el multiplicador de LR baja en línea recta de a a b con el progreso de la corrida
(reloj_progress con QF2_LR_SCHEDULE=reloj, paso / pasos objetivo con steps), sin calentamiento; vacío deja el calendario.

shards_banco(): con QF2_BANCO_SHARDS_DESDE=d (train.py lo deja en SHARDS_DESDE), los ids de los shards de entreno presentes en disco con id >= d, para que el
cargador base y los documentos de recolocado de todos los brazos lean datos que la corrida fuente no vio; vacío da None
y el cargador del kit queda como estaba.
"""
from __future__ import annotations

import json
import math
import time

import torch

import qf2_memhost_cfg as cfgm
import qf2_memhost_ckpt as ckptm
import qf2_memhost_cuantiza as cuant
import qf2_memhost_sincro as sincro
import qf2_memoria_anfitrion as anfitrion

TROZO_FILAS = 1 << 18
PLAZO_S = 1800.0


def leer_lrm(texto: str) -> tuple[float, float] | None:
    texto = (texto or "").strip()
    if not texto:
        return None
    partes = texto.split(":")
    try:
        valores = [float(p) for p in partes]
    except ValueError:
        valores = []
    if len(partes) != 2 or len(valores) != 2 or not all(math.isfinite(v) and v >= 0 for v in valores):
        raise ValueError(f"QF2_BANCO_LRM={texto!r}: se espera 'a:b' con a y b finitos y >= 0")
    return valores[0], valores[1]


def multiplicador(tramo: tuple[float, float], progreso: float) -> float:
    a, b = tramo
    return a + (b - a) * min(1.0, max(0.0, progreso))


VIA_V11 = ";via=v11"


def via_guardada(config: dict) -> dict:
    texto = config.get("memhost") or ""
    return {**config, "memhost": texto[:-len(VIA_V11)]} if texto.endswith(VIA_V11) else config


def abrir(ruta: str, config: dict) -> dict:
    payload = anfitrion.abrir(ruta)
    propio = payload.get("model_config") or {}
    if json.dumps(propio, sort_keys=True) not in (json.dumps(config, sort_keys=True), json.dumps(via_guardada(config), sort_keys=True)):
        distintas = sorted(k for k in set(propio) | set(config) if propio.get(k) != config.get(k))
        raise ValueError(f"QF2_INICIO_DESDE={ruta}: model_config distinto del modelo construido en {distintas}")
    return payload


def cargar_tronco(model: torch.nn.Module, payload: dict, tramo: tuple[float, float] | None) -> str:
    estado = payload["model_state"]
    ckptm.cargar_resto(model, estado)
    propias = set(ckptm.claves())
    tensores = [v for k, v in estado.items() if k not in propias]
    lrm = f"{tramo[0]:g}:{tramo[1]:g} lineal sobre el progreso" if tramo is not None else "el del calendario"
    propia = getattr(model, "memhost_cfg", None)
    guardada = (payload.get("model_config") or {}).get("memhost") or ""
    via = ("; sitio 11 de la vía res a la v con memhost_sitios.key.11 y val.11 del final.pt"
           if propia is not None and propia.via == "v11" and ";via=" not in guardada else "")
    return (f"qf2_inicio: tronco de {len(tensores)} tensores ({sum(v.numel() for v in tensores):,} valores) del final.pt de "
            f"step={payload.get('meta', {}).get('step')}; optimizador nuevo; lrm {lrm}{via}")


def tablas(payload: dict | None, imprimir):
    if payload is None:
        return None
    return lambda tabla: imprimir(cargar_tablas(tabla, payload))


def cargar_tablas(tabla, payload: dict, plazo: float = PLAZO_S) -> str:
    estado = payload["model_state"]
    q8, s16 = estado[cfgm.CLAVE_Q], estado[cfgm.CLAVE_S]
    if tuple(q8.shape) != (tabla.r, tabla.w) or q8.dtype != torch.int8 or tuple(s16.shape) != (tabla.r,) or s16.dtype != torch.float16:
        raise RuntimeError(f"qf2_inicio: tablas con forma/dtype inesperados q={tuple(q8.shape)} {q8.dtype} s={tuple(s16.shape)} "
                           f"{s16.dtype}, la tabla es ({tabla.r}, {tabla.w})")
    t0 = time.monotonic()
    lo, hi = tabla.propias
    for a in range(lo, hi, TROZO_FILAS):
        b = min(hi, a + TROZO_FILAS)
        tabla.q[a:b] = cuant.descuantizar(q8[a:b], s16[a:b])
    for t in (tabla.acc, tabla.cnt, tabla.m, tabla.v):
        if t is not None:
            t[lo:hi].zero_()
    cargado = time.monotonic() - t0
    (tabla.ruta / f"inicio.{tabla.rango}").write_text(tabla.id)
    for r in range(tabla.mundo):
        sincro.esperar_marca(tabla.ruta / f"inicio.{r}", tabla.id, plazo)
    return (f"qf2_inicio: tablas memhost filas [{lo:,}, {hi:,}) de {tabla.r:,} descuantizadas a los maestros del anfitrión en "
            f"{cargado:.1f} s, acc y cnt a cero; {tabla.mundo} rangos listos a los {time.monotonic() - t0:.1f} s")


SHARDS_DESDE = ""


def shards_banco():
    desde = SHARDS_DESDE
    if not desde:
        return None
    import prepare
    ids = [i for i in prepare.PUBLIC_TRAIN_SHARDS if i >= int(desde) and (prepare.DATA_DIR / prepare.shard_filename(i)).exists()]
    if not ids:
        raise RuntimeError(f"QF2_BANCO_SHARDS_DESDE={desde}: ningún shard de entreno en disco")
    return ids

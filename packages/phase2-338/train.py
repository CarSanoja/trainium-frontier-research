"""
Neuron Competition Round 2 baseline training script.

Participants edit this file. It is intentionally self-contained and readable.
The fixed data/tokenizer/eval utilities live in prepare.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import qf2_diff
import qf2_memoria
import qf2_memoria_anfitrion
import qf2_memoria_ckpt
import qf2_corpus_constructor
import qf2_corpus_mezcla
import qf2_guardado_anticipado
import qf2_memhost_canon
import qf2_memhost_ckpt
import qf2_memhost_cfg
import qf2_hilos_aten
import qf2_hilos_nice
import qf2_memhost_hilo
import qf2_memhost_opt
import qf2_memhost_hojas
import qf2_memhost_sitios
import qf2_memoria_opt
import qf2_mos
import qf2_ngramas
import qf2_peso_contexto
import qf2_pkm
import qf2_resumen

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    PUBLIC_TRAIN_SHARDS,
    Tokenizer,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    list_shard_paths,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 2048
# Best measured baseline package at MBS1/T2048 on all 64 LNC2 ranks.
DEPTH = int(os.environ.get("QF2_DEPTH", "24"))
ASPECT_RATIO = int(os.environ.get("QF2_ASPECT_RATIO", "64"))
HEAD_DIM = 128

DEVICE_BATCH_SIZE = int(os.environ.get("QF2_DEVICE_BATCH_SIZE", "4"))
TOTAL_BATCH_SIZE = int(os.environ.get("QF2_TOTAL_BATCH_SIZE", "1048576"))
NUM_STEPS = -1
TARGET_PARAM_DATA_RATIO = 10.30
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = float(os.environ.get("QF2_MATRIX_LR", "0.02"))
SCALAR_LR = 0.20
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 40
WARMDOWN_RATIO = float(os.environ.get("QF2_WARMDOWN_RATIO", "").strip() or 0.65)
FINAL_LR_FRAC = float(os.environ.get("QF2_FINAL_LR_FRAC", "").strip() or 0.05)
LR_SHAPE = os.environ.get("QF2_LR_SHAPE", "lin").strip()
LR_SCHEDULE = os.environ.get("QF2_LR_SCHEDULE", "reloj")
RELOJ_LOG = os.environ.get("QF2_RELOJ_LOG", "")
RELOJ_TRAMOS = os.environ.get("QF2_RELOJ_TRAMOS", "0") != "0"
RELOJ_STEP_CEILING = 1_000_000
CHECKPOINT_RESERVE_SECONDS = 5.0
MUON_MOMENTUM = 0.95
GRAD_CLIP = 0.0
SEED = int(os.environ.get("QF2_SEED", "1").strip() or 42)
WD_COS = os.environ.get("QF2_WD_COS", "1") != "0"
WD_PICO = float(os.environ.get("QF2_WD_PICO", "0.06").strip() or WEIGHT_DECAY)
SUELO_A = os.environ.get("QF2_SUELO_A", "1") != "0"
SUELO_MOM = SUELO_A or os.environ.get("QF2_SUELO_MOM", "0") != "0"
SUELO_ESC = SUELO_A or os.environ.get("QF2_SUELO_ESC", "0") != "0"
SUELO_LR = os.environ.get("QF2_SUELO_LR", "0") != "0"
SUELO_INIT = SUELO_A or os.environ.get("QF2_SUELO_INIT", "0") != "0"
OPT = os.environ.get("QF2_OPT", "")
OPT_BETA2 = float(os.environ.get("QF2_OPT_BETA2", "0.95"))
OPT_EPS = float(os.environ.get("QF2_OPT_EPS", "1e-8"))
OPT_VALORES = ("", "m2")
VALUE_EMBEDS = os.environ.get("QF2_VE", "1") != "0"
VE_PASO = int(os.environ.get("QF2_VE_PASO", "8"))
MLP_EXP = int(os.environ.get("QF2_MLP_EXP", "4"))
NGRAM_FILAS = int(os.environ.get("QF2_NGRAM_FILAS", "0"))
MEM_FILAS = int(os.environ.get("QF2_MEM_FILAS", "0"))
MEMHOST_VIA_SITIO11 = os.environ.get("QF2_MEMHOST_VIA_SITIO11", "").strip()
MEMHOST = qf2_memhost_cfg.desde_banderas(
    os.environ.get("QF2_MEMHOST", "1"), os.environ.get("QF2_MEMHOST_ANCHO", "128"),
    os.environ.get("QF2_MEMHOST_CABEZAS", "2x2:64000000,3x4:64000000"), os.environ.get("QF2_MEMHOST_SITIOS", "in,1,11"),
    os.environ.get("QF2_MEMHOST_SIGNOS", "1"), os.environ.get("QF2_MEMHOST_CANON", "1"),
    os.environ.get("QF2_MEMHOST_PUERTA", "sig"),
    qf2_memhost_cfg.via_sitio11(os.environ.get("QF2_MEMHOST_VIA", "res"), MEMHOST_VIA_SITIO11))
if MEMHOST_VIA_SITIO11 and not MEMHOST:
    raise RuntimeError("QF2_MEMHOST_VIA_SITIO11 necesita QF2_MEMHOST=1")
MEMHOST_RETRASO = int(os.environ.get("QF2_MEMHOST_RETRASO", "1"))
MEMHOST_MASCARA = int(os.environ.get("QF2_MEMHOST_MASCARA", "0"))
qf2_hilos_nice.NICE = int(os.environ.get("QF2_HILOS_NICE", "0"))
qf2_memhost_hilo.traza.TRAZA = int(os.environ.get("QF2_MEMHOST_TRAZA", "0"))
qf2_hilos_aten.ATEN = int(os.environ.get("QF2_HILOS_ATEN", "1"))
qf2_hilos_aten.HILO = int(os.environ.get("QF2_HILOS_HILO", "0"))
MEMHOST_X0_ESCALA = float(os.environ.get("QF2_MEMHOST_X0_ESCALA", "1"))
GIL_MS = float(os.environ.get("QF2_GIL_MS", "0"))
if GIL_MS > 0:
    import sys
    sys.setswitchinterval(GIL_MS / 1e3)
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_gil: switchinterval={sys.getswitchinterval()}", flush=True)
MEMHOST_OPT = os.environ.get("QF2_MEMHOST_OPT", "rms")
MEMHOST_LR = float(os.environ.get("QF2_MEMHOST_LR", "0"))
MEMHOST_RUTA = os.environ.get("QF2_MEMHOST_RUTA", "/dev/shm/memhost")
MEMHOST_VAL_ALEATORIO = os.environ.get("QF2_MEMHOST_VAL", "1") == "1"
MEMHOST_RESERVA_S = float(os.environ.get("QF2_MEMHOST_RESERVA_S", "250"))
PESO_CTX = qf2_peso_contexto.desde_entorno()
MEMCORPUS = os.environ.get("QF2_MEMCORPUS", "").strip()
MEMCORPUS_RUTA = os.environ.get("QF2_MEMCORPUS_RUTA", "/dev/shm/memcorpus")
MEMCORPUS_PROCESOS = int(os.environ.get("QF2_MEMCORPUS_PROCESOS", "96"))
MEMCORPUS_EXCLUIR = int(os.environ.get("QF2_MEMCORPUS_EXCLUIR", "256"))
MEMCORPUS_RESERVA_S = float(os.environ.get("QF2_MEMCORPUS_RESERVA_S", "300"))
MEMCORPUS_ANTICIPADO_S = float(os.environ.get("QF2_MEMCORPUS_ANTICIPADO_S", "0"))
MEMCORPUS_TRONCO = float(os.environ.get("QF2_MEMCORPUS_TRONCO", "0"))
MEMHOST_DECAY = os.environ.get("QF2_MEMHOST_DECAY", "0")
MEM_ANCHO = int(os.environ.get("QF2_MEM_ANCHO", "0"))
MEM_TRANSPORTE = os.environ.get("QF2_MEM_TRANSPORTE", "b")
PKM_FILAS = int(os.environ.get("QF2_PKM_FILAS", "0"))
PKM_K = int(os.environ.get("QF2_PKM_K", "4"))
PKM_VALOR = int(os.environ.get("QF2_PKM_VALOR", "256"))
PKM_DK = int(os.environ.get("QF2_PKM_DK", "128"))
PKM_LR = float(os.environ.get("QF2_PKM_LR", "").strip() or EMBEDDING_LR)
RESUMEN = os.environ.get("QF2_RESUMEN", "6:12").strip()
DIFF_CAPAS = os.environ.get("QF2_DIFF_CAPAS", "")
DIFF_SIGNO = int(os.environ.get("QF2_DIFF_SIGNO", "-1"))
MOS = int(os.environ.get("QF2_MOS", "0"))
MEM_EVAL_ANFITRION_BYTES = int(float(os.environ.get("QF2_MEM_EVAL_ANFITRION_BYTES", "8e9")))
ROPE_MM = os.environ.get("QF2_ROPE_MM", "1") != "0"
ROPE_ROLES = os.environ.get("QF2_ROPE_ROLES", "").strip()
QKNORM_MM = os.environ.get("QF2_QKNORM_MM", "1") != "0"
QK_ESCALA = float(os.environ.get("QF2_QK_ESCALA", "1.2"))
FP8_MLP = os.environ.get("QF2_FP8_MLP", "1") != "0"
os.environ["QF2_FP8_DYSCALE"] = os.environ.get("QF2_FP8_DYSCALE", "18")
os.environ["QF2_FP8_RECORTE_GRAD"] = os.environ.get("QF2_FP8_RECORTE_GRAD", "1")
os.environ["QF2_FP8_BWDK"] = os.environ.get("QF2_FP8_BWDK", "0")
if FP8_MLP:
    from kernels.fp8_mlp_op import RESUMEN as FP8_RESUMEN, mlp_fp8
    from kernels.fp8_mlp_comun import _H_CLAMP, _SCALE_H
EVAL_TOPE = os.environ.get("QF2_EVAL_TOPE", "t61").strip()
EMB_SCATTER =os.environ.get("QF2_EMB_SCATTER", "1") != "0"
EMB_SCATTER_GATE = os.environ.get("QF2_EMB_SCATTER_GATE", "0") != "0"
EMB_DETACH = os.environ.get("QF2_EMB_DETACH", "0") != "0"
EMB_RUTA = EMB_SCATTER or EMB_DETACH
if EMB_RUTA:
    from kernels.emb_op import emb_descripcion, emb_preparar, emb_tabla
if EMB_SCATTER_GATE:
    from kernels.emb_puerta import emb_puerta
CEFA = os.environ.get("QF2_CEFA", "0") != "0"
if CEFA:
    from cefa_perdida import cefa_descripcion, cefa_perdida
CE_LSE = os.environ.get("QF2_CE_LSE", "0") != "0"
CE_TROZOS = int(os.environ.get("QF2_CE_TROZOS", "1024"))
if CE_TROZOS:
    from ce_trozos import ce_trozos, ce_trozos_tokens
PROMEDIO_MODO = os.environ.get("QF2_PROMEDIO", "0").strip()
PROMEDIO = PROMEDIO_MODO not in ("", "0")
if PROMEDIO:
    from qf2_promedio import Promedio
GUARDA = os.environ.get("QF2_GUARDA", "1") != "0"
if GUARDA:
    from qf2_guarda import Guarda
REGISTRO = os.environ.get("QF2_REGISTRO", "0") != "0"
PREFIJO_BOS = os.environ.get("QF2_PREFIJO_BOS", "0") != "0"
COM_ESCALARES = os.environ.get("QF2_COM_ESCALARES", "1") != "0"
COM_MUON = os.environ.get("QF2_COM_MUON", "1") != "0"
COM_GATHER_BF16 = os.environ.get("QF2_COM_GATHER_BF16", "1") != "0"
COM_RS_BF16 = os.environ.get("QF2_COM_RS_BF16", "0") != "0"
PLANO_TRANSITORIO = os.environ.get("QF2_PLANO_TRANSITORIO", "0") != "0"
COM_FRANJAS = int(os.environ.get("QF2_COM_FRANJAS", "8").strip() or 0)
OPT_COMPILADO = os.environ.get("QF2_OPT_COMPILADO", "1") != "0"
if OPT_COMPILADO:
    import qf2_opt_compilado
if COM_ESCALARES or COM_MUON or COM_GATHER_BF16 or COM_RS_BF16:
    import qf2_reparto
if COM_FRANJAS:
    import qf2_franjas
if COM_GATHER_BF16:
    CHECKPOINT_RESERVE_SECONDS += 25.0
COLA_BANDERAS = (
    os.environ.get("QF2_SEQ_MAX", "").strip(),
    os.environ.get("QF2_CARGADOR_LARGO", "1").strip(),
    os.environ.get("QF2_COLA_FRAC", "").strip(),
    os.environ.get("QF2_CALIENTA_LARGO", "0").strip(),
)
COLA4096 = any(v not in ("", "0") for v in COLA_BANDERAS)
COLA_LARGA = any(v not in ("", "0") for v in (COLA_BANDERAS[0], COLA_BANDERAS[2], COLA_BANDERAS[3]))
if COLA4096:
    import qf2_cola
    qf2_cola.configurar(*COLA_BANDERAS)
FLUJO_FRAC = os.environ.get("QF2_FLUJO_FRAC", "").strip()
if FLUJO_FRAC not in ("", "0"):
    import qf2_flujo
    qf2_flujo.configurar(FLUJO_FRAC, COLA4096)
FLUJO = FLUJO_FRAC not in ("", "0") and qf2_flujo.activo()
PUNTOS_S = os.environ.get("QF2_PUNTOS_S", "").strip()
REANUDA = os.environ.get("QF2_REANUDA", "0") != "0"
PUNTOS = PUNTOS_S not in ("", "0") or REANUDA
if PUNTOS:
    import qf2_flujo
    import qf2_puntos
PRECARGA = os.environ.get("QF2_PRECARGA", "1") != "0" and not PUNTOS and not FLUJO
if PRECARGA:
    import qf2_precarga
RECOLOCADO = float(os.environ.get("QF2_RECOLOCADO", "0").strip() or 0)
RECOLOCADO_CE = os.environ.get("QF2_RECOLOCADO_CE", "todo").strip()
RECOLOCADO_RAFAGA = os.environ.get("QF2_RECOLOCADO_RAFAGA", "").strip()
if RECOLOCADO > 0 and not PRECARGA:
    raise RuntimeError("QF2_RECOLOCADO necesita QF2_PRECARGA=1")
if PRECARGA:
    import qf2_recolocado
RECOLOCADO_FORMATO = os.environ.get("QF2_RECOLOCADO_FORMATO", "ejemplos").strip()
PARES_TRAMO = os.environ.get("QF2_PARES_TRAMO", "8:40").strip()
PARES_ETIQUETAS = os.environ.get("QF2_PARES_ETIQUETAS", "Input/Output").strip()
INICIO_DESDE = os.environ.get("QF2_INICIO_DESDE", "").strip()
BANCO_LRM = os.environ.get("QF2_BANCO_LRM", "").strip()
if INICIO_DESDE and not PRECARGA:
    raise RuntimeError("QF2_INICIO_DESDE necesita QF2_PRECARGA=1")
import qf2_inicio
BANCO_LRM_TRAMO = qf2_inicio.leer_lrm(BANCO_LRM)
qf2_inicio.SHARDS_DESDE = os.environ.get("QF2_BANCO_SHARDS_DESDE", "").strip()
import qf2_agrupa
AGRUPA = qf2_agrupa.leer(os.environ.get("QF2_AGRUPA", "0.7:4096:relacionados").strip())
import qf2_agrupa_fuente
qf2_agrupa_fuente.CONFIG = qf2_agrupa_fuente.leer(os.environ.get("QF2_AGRUPA_FUENTE", "grupos:0:240").strip())
if qf2_agrupa_fuente.CONFIG is not None and (AGRUPA is None or RECOLOCADO > 0 or INICIO_DESDE or BANCO_LRM or qf2_inicio.SHARDS_DESDE):
    raise ValueError("QF2_AGRUPA_FUENTE necesita AGRUPA desde cero, RECOLOCADO=0 y calendario ordinario")
if AGRUPA is not None and not PRECARGA:
    raise RuntimeError("QF2_AGRUPA necesita QF2_PRECARGA=1")
import qf2_filas
qf2_filas.validar(AGRUPA, RECOLOCADO)
import qf2_objetivos
DIFICILES = qf2_objetivos.leer_objetivo(os.environ.get("QF2_CE_DIFICILES", "").strip(), "")
TRAMO = qf2_objetivos.leer_objetivo("", os.environ.get("QF2_CE_TRAMO", "").strip())
OBJETIVO = TRAMO if DIFICILES is None else DIFICILES
RUIDO = qf2_objetivos.Ruido(os.environ.get("QF2_RUIDO_ENTRADA", "").strip(), SEED)
if OBJETIVO is not None and (not CE_TROZOS or CEFA or MOS):
    raise RuntimeError("QF2_CE_DIFICILES y QF2_CE_TRAMO van en la CE por trozos: necesitan QF2_CE_TROZOS sin QF2_CEFA ni QF2_MOS")
if RUIDO.activo and PESO_CTX.activo:
    raise RuntimeError("QF2_RUIDO_ENTRADA no llega a las pasadas propias de QF2_PESO_CTX")
import qf2_brier
BRIER = qf2_brier.leer(os.environ.get("QF2_BRIER", "").strip())
if BRIER is not None and (not CE_TROZOS or CEFA or MOS or PESO_CTX.activo):
    raise RuntimeError("QF2_BRIER va en la CE por trozos: necesita QF2_CE_TROZOS, sin QF2_CEFA, QF2_MOS ni QF2_PESO_CTX")
import qf2_equilibrio
EQUILIBRIO = qf2_equilibrio.leer(os.environ.get("QF2_EQUILIBRIO", "").strip())
EQUILIBRIO = qf2_filas.equilibrio(EQUILIBRIO, AGRUPA)
if EQUILIBRIO is not None and (not CE_TROZOS or CEFA or MOS or REGISTRO or MEMCORPUS or PESO_CTX.activo):
    raise RuntimeError("QF2_EQUILIBRIO va solo en la CE por trozos: sin QF2_CEFA, QF2_MOS, QF2_REGISTRO, QF2_MEMCORPUS ni QF2_PESO_CTX")
import qf2_union
UNION = qf2_union.leer(DIFICILES, TRAMO, EQUILIBRIO, BRIER)
VE_GATE_CHANNELS = 12
SUELO_B_REF = 2**19
SUELO_UNEMBEDDING_LR = 0.008
SUELO_MATRIX_LR = 0.02
SUELO_RESID_LR = 0.005
SUELO_X0_LR = 0.5
SUELO_MOMENTUM_WARMUP_FRAC = 0.0625
SUELO_MOMENTUM_START = 0.85
SUELO_MOMENTUM_PEAK = 0.97
SUELO_MOMENTUM_END = 0.90


# -----------------------------------------------------------------------------
# Device / distributed setup

def neuron_available() -> bool:
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        return "neuron"
    return "cpu"


def init_runtime(device_type: str):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(SEED)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    elif ddp and device_type == "neuron":
        dist.init_process_group("neuron")
        device = torch.device("neuron")
    elif ddp:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return ddp, rank, local_rank, world_size, device


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def forward_mem(model, mem_obj, x, y, mem=None, corpus=None, reduccion="mean"):
    extra = {"corpus": corpus} if corpus is not None else {}
    if reduccion != "mean":
        extra["loss_reduction"] = reduccion
    if RUIDO.activo and model.training:
        extra["ruido"] = RUIDO.muestra(x.size(0), model.config.n_embd, x.device, COMPUTE_DTYPE)
    if mem is not None:
        return model(x, y, mem=mem, **extra)
    if mem_obj is None:
        return model(x, y, **extra)
    return model(x, y, mem=mem_obj.consultar(x, COMPUTE_DTYPE), **extra)


def perdida_micro(model, mem_obj, x, y, mem, corpus, paso, micro):
    if not PESO_CTX.activo:
        return forward_mem(model, mem_obj, x, y, mem, corpus)
    nll = forward_mem(model, mem_obj, x, y, mem, corpus, "none")
    return qf2_peso_contexto.ponderar(PESO_CTX, lambda a, b: model(a, b, loss_reduction="none"), x, y, nll, paso, micro)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "neuron":
        return torch.bfloat16
    return torch.float32


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    value_embeds: bool = False
    ve_paso: int = 2
    registro: bool = False
    prefijo_bos: bool = False
    ngramas: int = 0
    memoria: int = 0
    memoria_ancho: int = 0
    memhost: str = ""
    memcorpus: str = ""
    pkm: int = 0
    pkm_k: int = 4
    pkm_valor: int = 256
    pkm_dk: int = 128
    resumen: str = ""
    diff_capas: str = ""
    diff_signo: int = -1
    mos: int = 0
    mlp_exp: int = 4
    rope_roles: str = ""


def has_ve(layer_idx: int, n_layer: int, paso: int = 2) -> bool:
    return (n_layer - 1 - layer_idx) % paso == 0


def config_dict(config: GPTConfig) -> dict:
    return {k: v for k, v in asdict(config).items()
            if (config.pkm or not k.startswith("pkm")) and (config.diff_capas or not k.startswith("diff")) and (config.mos or k != "mos") and (k not in ("value_embeds", "registro", "prefijo_bos", "ngramas", "memoria", "memoria_ancho", "memhost", "memcorpus", "resumen") or v) and (k != "ve_paso" or (config.value_embeds and v != 2)) and (k != "mlp_exp" or v != 4) and (k != "rope_roles" or v)}


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


def tabla(mod: nn.Embedding, idx: torch.Tensor) -> torch.Tensor:
    return emb_tabla(idx, mod.weight, COMPUTE_DTYPE, EMB_DETACH) if EMB_RUTA else mod(idx)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


_BOS_ID: list[int] = []


def bos_token_id() -> int:
    if not _BOS_ID:
        _BOS_ID.append(ensure_tokenizer(build_if_missing=False).get_bos_token_id())
    return _BOS_ID[0]


def anteponer(primero: torch.Tensor, v: torch.Tensor, recorta: bool) -> torch.Tensor:
    cabeza = primero.to(v.dtype).view(1, 1, -1).expand(v.size(0), 1, v.size(-1))
    return torch.cat([cabeza, v[:, :-1] if recorta else v], dim=1)


def prefijo_bos(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bos = idx.new_full((idx.size(0), 1), bos_token_id())
    sin_bos = idx[:, :1] != bos
    return torch.where(sin_bos, torch.cat([bos, idx], dim=1), torch.cat([idx, bos], dim=1)), sin_bos


# -----------------------------------------------------------------------------
# Example: a custom NKI kernel (optional, default OFF)
#
# This is an ILLUSTRATION to get you started writing your own Trainium kernels.
# It implements the MLP activation `relu(x)**2` as a fused NKI kernel. The eager
# baseline does not use it; enable it with `--nki-relu2` (Neuron only).
#
# Three things worth copying from this example:
#   1. The kernel body: load a tile from HBM into SBUF, compute, store back.
#   2. Tiling over the 128-partition limit: a NeuronCore processes at most 128
#      rows (the "partition" dim) at once, so we loop over chunks of 128 rows.
#   3. Autograd: a raw NKI kernel is NOT differentiable. To use one inside the
#      training forward pass you must wrap it in a torch.autograd.Function and
#      provide the backward yourself, or gradients silently stop flowing.
#
# To write your own: change the per-tile math in `_relu_sq_fwd` (and the matching
# derivative in `_relu_sq_bwd`), or fuse a different elementwise op.

_NKI_PARTITION = 128  # NeuronCore SBUF partition limit (== nki.language.tile_size.pmax)

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _relu_sq_fwd(x):
        """Fused y = relu(x)**2 for a 2-D [M, N] tensor, tiled over rows."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)  # relu
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)     # square
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    @nki.jit
    def _relu_sq_bwd(x, grad_out):
        """Backward of relu(x)**2: grad_in = grad_out * 2 * relu(x)."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            gb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=grad_out[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)   # relu(x)
            two_r = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=two_r, data=rb, op0=nl.multiply, operand0=2.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=two_r, data2=gb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    class _ReluSquaredNKI(torch.autograd.Function):
        """Makes the NKI relu**2 kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x):
            y = _relu_sq_fwd(x.reshape(-1, x.shape[-1])).reshape(x.shape)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)

    _HAS_NKI = True
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) — eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the example NKI kernel when enabled, else eager."""
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def mascara_secuencias(idx: torch.Tensor, por_ciento: int) -> torch.Tensor:
    viva = torch.remainder(idx[:, 1:9].float().sum(dim=1), 100.0) >= por_ciento
    return viva.view(-1, 1, 1) | (torch.arange(idx.size(1), device=idx.device) < 8).view(1, -1, 1)


def qknorm(x: torch.Tensor, media: torch.Tensor | None = None) -> torch.Tensor:
    if media is None:
        return norm(x) * QK_ESCALA if QK_ESCALA != 1.0 else norm(x)
    return x * torch.rsqrt(torch.matmul(x * x, media.to(x.dtype)) + torch.finfo(torch.float32).eps)


def rot_matrix(d: int) -> torch.Tensor:
    h = d // 2
    z = torch.zeros(h, h, dtype=torch.float32)
    i = torch.eye(h, dtype=torch.float32)
    return torch.cat((torch.cat((z, -i), dim=1), torch.cat((i, z), dim=1)), dim=0)


def rope_roles_mask(roles: str, n_head: int, pairs: int) -> torch.Tensor | None:
    if not roles:
        return None
    tipo, sep, arg = roles.partition(":")
    if tipo not in ("nope", "pares") or not sep or not arg.isascii() or not arg.isdigit():
        raise ValueError(f"rope_roles {roles!r}: se esperaba '', 'nope:N' o 'pares:K'")
    n = int(arg)
    mask = torch.zeros(n_head, pairs, dtype=torch.bool)
    if tipo == "nope":
        if n > n_head:
            raise ValueError(f"rope_roles {roles!r}: N={n} supera n_head={n_head}")
        mask[:n] = True
        return mask
    if n < 1 or pairs % n:
        raise ValueError(f"rope_roles {roles!r}: K={n} debe dividir {pairs} pares")
    for h in range(n_head):
        s = (h % (pairs // n)) * n
        mask[h, s:s + n] = True
    return mask


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor | None = None) -> torch.Tensor:
    if rot is not None:
        return x * cos + torch.matmul(x, rot.to(x.dtype)) * sin
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        if ROPE_MM:
            self.register_buffer("rot", rot_matrix(self.head_dim).to(COMPUTE_DTYPE), persistent=False)
        else:
            self.rot = None
        if QKNORM_MM:
            self.register_buffer("media", torch.full((self.head_dim, self.head_dim), 1.0 / (self.head_dim * QK_ESCALA ** 2)).to(COMPUTE_DTYPE), persistent=False)
        else:
            self.media = None
        if config.value_embeds and has_ve(layer_idx, config.n_layer, config.ve_paso):
            with torch.random.fork_rng(devices=[]):
                self.ve_gate = Linear(VE_GATE_CHANNELS, config.n_head, bias=False)
        else:
            self.ve_gate = None
        self.diff = qf2_diff.RamaDiff(config.n_embd, config.n_head, config.diff_signo) if layer_idx in qf2_diff.indices(config.diff_capas, config.n_layer) else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None = None, mem_v: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve.view(b, t, self.n_head, self.head_dim)
        if mem_v is not None:
            v = v + mem_v.view(b, t, self.n_head, self.head_dim)
        q = qknorm(apply_rotary_emb(q, cos, sin, self.rot), self.media)
        k = qknorm(apply_rotary_emb(k, cos, sin, self.rot), self.media)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.diff is not None:
            q2 = qknorm(apply_rotary_emb(self.diff.c_q2(x).view(b, t, self.n_head, self.head_dim), cos, sin, self.rot), self.media).transpose(1, 2)
            y = self.diff.combinar(y, F.scaled_dot_product_attention(q2, k, v, is_causal=True), x)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


def oculta_mlp(config: GPTConfig) -> int:
    return config.mlp_exp if config.mlp_exp > 16 else config.mlp_exp * config.n_embd


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, oculta_mlp(config), bias=False)
        self.c_proj = Linear(oculta_mlp(config), config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FP8_MLP and self.training and torch.is_grad_enabled():
            return mlp_fp8(x, self.c_fc.weight, self.c_proj.weight)
        if FP8_MLP and EVAL_TOPE == "t61":
            return self.c_proj(torch.clamp(F.relu(self.c_fc(x)), max=_H_CLAMP / _SCALE_H).square())
        return self.c_proj(relu_squared(self.c_fc(x)))


def mix_residual(
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_lambda: torch.Tensor,
    x0_lambda: torch.Tensor,
) -> torch.Tensor:
    """Mix residual streams without promoting BF16 activations to FP32."""
    return resid_lambda.to(dtype=x.dtype) * x + x0_lambda.to(dtype=x.dtype) * x0


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None = None, mem_v: torch.Tensor | None = None) -> torch.Tensor:
        x = mix_residual(x, x0, self.resid_lambda, self.x0_lambda)
        x = x + self.attn(norm(x), cos, sin, ve, mem_v)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, repartida: bool = False, memoria_dtype: torch.dtype = torch.float32, memoria_device=None):
        super().__init__()
        self.config = config
        self.repartida = repartida
        self.memoria_anfitrion = False
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        with torch.random.fork_rng(devices=[]):
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded_vocab_size, config.n_embd)
                for i in range(config.n_layer) if config.value_embeds and has_ve(i, config.n_layer, config.ve_paso)
            })
        if config.ngramas and (config.registro or config.prefijo_bos):
            raise ValueError("ngramas es incompatible con registro y prefijo_bos")
        self.ngramas = qf2_ngramas.tablas(config.ngramas, config.n_embd) if config.ngramas else nn.ModuleDict()
        if config.ngramas:
            self.ngrama_escala = qf2_ngramas.escala()
            self.ngrama_bos = bos_token_id()
        self.memoria = nn.ModuleDict()
        if config.memoria:
            if config.ngramas or config.registro or config.prefijo_bos:
                raise ValueError("memoria es incompatible con ngramas, registro y prefijo_bos")
            if not repartida:
                self.memoria = qf2_memoria.tablas(config.memoria, config.memoria_ancho, memoria_dtype, memoria_device)
            with torch.random.fork_rng(devices=[]):
                self.memoria_proy = Linear(2 * config.memoria_ancho, config.n_embd, bias=False)
            self.memoria_escala = qf2_ngramas.escala()
            self.memoria_bos = bos_token_id()
        self.memhost_cfg = None
        if config.memhost:
            if config.ngramas or config.memoria or config.registro or config.prefijo_bos:
                raise ValueError("memhost es incompatible con ngramas, memoria, registro y prefijo_bos")
            self.memhost_cfg = qf2_memhost_cfg.desde_texto(config.memhost)
            self.memhost_bos = bos_token_id()
            self.memhost_sitios = qf2_memhost_sitios.Sitios(self.memhost_cfg, config.n_embd, Linear)
            self.memhost = qf2_memhost_sitios.Tablas(self.memhost_cfg, config.vocab_size)
        self.memcorpus_cfg = None
        if config.memcorpus:
            if config.mos or config.registro or config.prefijo_bos:
                raise ValueError("memcorpus es incompatible con mos, registro y prefijo_bos")
            self.memcorpus_cfg = qf2_corpus_mezcla.desde_texto(config.memcorpus)
            self.memcorpus_bos = bos_token_id()
            self.memcorpus_puerta = qf2_corpus_mezcla.Puerta(self.memcorpus_cfg, config.n_embd)
        if config.registro:
            self.registro = nn.Parameter(torch.zeros(config.n_embd))
        self.pkm = qf2_pkm.PKM(config.pkm, config.n_embd, config.pkm_k, config.pkm_valor, config.pkm_dk) if config.pkm else None
        if config.resumen and config.registro:
            raise ValueError("resumen es incompatible con registro")
        self.resumen = qf2_resumen.Resumen(config.resumen, config.n_layer, config.n_embd, bos_token_id(), Linear) if config.resumen else None
        self.mos = qf2_mos.Mezcla(config.n_embd, config.mos) if config.mos else None
        cos, sin = self._precompute_rotary_embeddings(self.rotary_len(), config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, memoria: bool = True) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        depth_span = max(self.config.n_layer - 1, 1)
        for i, block in enumerate(self.transformer.h):
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if SUELO_INIT:
                block.resid_lambda.fill_(1.15 - 0.10 * i / depth_span)
                block.x0_lambda.fill_(0.20 - 0.15 * i / depth_span)
            else:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        if self.config.ngramas:
            qf2_ngramas.iniciar(self.ngramas, self.config.n_embd)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_len(), self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        if self.config.registro:
            self.registro.copy_(self.transformer.wte.weight[bos_token_id()])
        if self.config.memhost:
            self.memhost_sitios.iniciar(MEMHOST_VAL_ALEATORIO)
        if self.config.memcorpus:
            self.memcorpus_puerta.w.fill_(qf2_corpus_mezcla.LAM_INICIAL)
            if self.memcorpus_puerta.v is not None:
                self.memcorpus_puerta.v.zero_()
        if self.config.memoria:
            torch.nn.init.uniform_(self.memoria_proy.weight, -s, s)
            if memoria and not self.repartida:
                qf2_memoria.iniciar(self.memoria, SEED)
        if self.pkm is not None:
            self.pkm.iniciar()
        if self.resumen is not None:
            self.resumen.iniciar()
        if self.mos is not None:
            self.mos.iniciar()
        for block in self.transformer.h:
            if block.attn.diff is not None:
                block.attn.diff.iniciar()

    def rotary_len(self) -> int:
        return self.config.sequence_len + int(self.config.registro) + int(self.config.prefijo_bos)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 100000):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        mask = rope_roles_mask(self.config.rope_roles, self.config.n_head, head_dim // 2)
        if mask is not None:
            mask = mask.to(device)[None, None]
            cos = torch.where(mask, torch.ones_like(cos), cos).contiguous()
            sin = torch.where(mask, torch.zeros_like(sin), sin).contiguous()
        if ROPE_MM:
            return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer  # per-block scalars
        non_matmul += sum(p.numel() for p in self.value_embeds.parameters())
        non_matmul += sum(p.numel() for p in self.ngramas.parameters())
        non_matmul += sum(p.numel() for p in self.memoria.parameters())
        if self.pkm is not None:
            non_matmul += self.pkm.valores.weight.numel() + self.pkm.escala.numel()
        non_matmul += self.registro.numel() if self.config.registro else 0
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T: 2*h*d*T per token, AV: 2*h*d*T per token, x3 for fwd+bwd
        attn_flops = (self.config.n_layer + len(qf2_diff.indices(self.config.diff_capas, self.config.n_layer))) * 12 * h * d * t
        mos_flops = 6 * self.lm_head.weight.numel() if self.mos is not None else 0
        return 6 * (nparams - non_matmul) + attn_flops + mos_flops

    def num_scaling_params(self) -> dict[str, int]:
        """Parameter count used for nanochat-style data:parameter scaling."""
        transformer_matrices = sum(
            p.numel()
            for block in self.transformer.h
            for name, p in block.named_parameters()
            if p.ndim == 2 and "ve_gate" not in name and "diff." not in name
        )
        lm_head = self.lm_head.weight.numel()
        return {
            "transformer_matrices": transformer_matrices,
            "lm_head": lm_head,
            "total": transformer_matrices + lm_head,
        }

    def memoria_lr(self, total_batch_size: int) -> float:
        batch_lr_scale = (total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0
        return EMBEDDING_LR * (self.config.n_embd / 768) ** -0.5 * 0.5 * batch_lr_scale

    def setup_optimizer(self, total_batch_size: int = TOTAL_BATCH_SIZE):
        scalar_params = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        if self.config.memoria:
            matrix_params.append(self.memoria_proy.weight)
        if self.config.memhost:
            matrix_params.extend(self.memhost_sitios.matrices())
        if self.pkm is not None:
            matrix_params.extend([self.pkm.consulta, self.pkm.salida])
        if self.resumen is not None:
            matrix_params.append(self.resumen.proy.weight)
        if self.mos is not None:
            matrix_params.append(self.mos.w2)
        embedding_params = list(self.transformer.wte.parameters()) + ([self.registro] if self.config.registro else [])
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        batch_lr_scale = (total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0
        unembedding_lr = SUELO_UNEMBEDDING_LR if SUELO_LR else UNEMBEDDING_LR
        matrix_lr = SUELO_MATRIX_LR if SUELO_LR else MATRIX_LR
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        ]
        if SUELO_ESC:
            param_groups.append(dict(kind="adamw", params=[b.resid_lambda for b in self.transformer.h], lr=SUELO_RESID_LR * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05))
            param_groups.append(dict(kind="adamw", params=[b.x0_lambda for b in self.transformer.h], lr=SUELO_X0_LR * batch_lr_scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        else:
            param_groups.append(dict(kind="adamw", params=scalar_params, lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        value_embeds_params = list(self.value_embeds.parameters())
        if value_embeds_params:
            param_groups.append(dict(kind="adamw", params=value_embeds_params, lr=EMBEDDING_LR * dmodel_lr_scale * 0.5 * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
        ngramas_params = list(self.ngramas.parameters())
        if ngramas_params:
            param_groups.append(dict(kind="adamw", params=ngramas_params, lr=EMBEDDING_LR * dmodel_lr_scale * 0.5 * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind="adamw", params=[self.ngrama_escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        memoria_params = list(self.memoria.parameters())
        if memoria_params:
            param_groups.append(dict(kind="adamw", params=memoria_params, lr=self.memoria_lr(total_batch_size), betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        if self.pkm is not None:
            param_groups.append(dict(kind="adamw", params=[self.pkm.valores.weight, self.pkm.subclaves], lr=PKM_LR * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind="adamw", params=[self.pkm.escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.mos is not None:
            param_groups.append(dict(kind="adamw", params=[self.mos.pi_w, self.mos.pi_b], lr=qf2_mos.pi_lr(self.config.n_embd) * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memhost and self.memhost_sitios.escalares():
            param_groups.append(dict(kind="adamw", params=self.memhost_sitios.escalares(), lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memoria:
            param_groups.append(dict(kind="adamw", params=[self.memoria_escala], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if self.config.memcorpus:
            param_groups.append(dict(kind="adamw", params=[self.memcorpus_puerta.w], lr=SCALAR_LR * batch_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
            if self.memcorpus_puerta.v is not None:
                param_groups.append(dict(kind="adamw", params=[self.memcorpus_puerta.v], lr=unembedding_lr * dmodel_lr_scale * batch_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=matrix_lr * batch_lr_scale, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=wd_muon(0.0)))
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        opt = DistMuonAdamW(param_groups) if distributed else MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def consulta_memoria(self, idx: torch.Tensor):
        if self.memoria_anfitrion:
            return qf2_memoria_anfitrion.consultar(self.memoria, idx, self.config.memoria, self.memoria_bos)
        return qf2_memoria.consulta_replicada(self.memoria, idx, self.config.memoria, self.memoria_bos)

    def memoria_externa(self, idx: torch.Tensor):
        if self.config.memhost:
            return qf2_memhost_ckpt.consultar(self, idx, COMPUTE_DTYPE)
        return self.consulta_memoria(idx) if self.memoria_anfitrion else None

    def salida_memcorpus(self, x, idx, targets, loss_reduction, corpus):
        if CEFA or CE_TROZOS or CE_LSE:
            raise ValueError("QF2_MEMCORPUS calcula su propia pérdida: incompatible con QF2_CEFA, QF2_CE_TROZOS y QF2_CE_LSE")
        cand, celda = corpus if corpus is not None else qf2_corpus_mezcla.consultar_modelo(self, idx)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        s = self.memcorpus_puerta.desplazamiento(x)
        if targets is None:
            return qf2_corpus_mezcla.log_probs(self.memcorpus_puerta, logits, cand, celda, s)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction="none")
        return qf2_corpus_mezcla.reducir(self.memcorpus_puerta, nll, targets, cand, celda, loss_reduction, s,
                                         MEMCORPUS_TRONCO if self.training else 0.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", mem=None, corpus=None, ruido=None):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        registro = self.config.registro
        entrena = targets is not None
        prefijo = self.config.prefijo_bos and not entrena
        if prefijo:
            idx, sin_bos = prefijo_bos(idx)
            t = idx.size(1)
        n = t + int(registro and not entrena)
        cos, sin = self.cos[:, :n], self.sin[:, :n]
        x = tabla(self.transformer.wte, idx).to(COMPUTE_DTYPE)
        if registro:
            x = anteponer(self.registro, x, entrena)
            if entrena:
                targets = torch.cat([targets.new_full((targets.size(0), 1), -1), targets[:, :-1]], dim=1)
        x = norm(x)
        if ruido is not None and self.training:
            x = x + ruido
        if self.config.ngramas:
            x = qf2_ngramas.sumar(self.ngramas, self.ngrama_escala, idx, self.config.ngramas, self.ngrama_bos, tabla, x)
        if self.config.memoria:
            if mem is None:
                if self.repartida:
                    raise ValueError("el modelo repartido necesita mem")
                mem = self.consulta_memoria(idx)
            x = qf2_memoria.combinar(mem[0], mem[1], self.memoria_proy, self.memoria_escala, x)
        if self.config.memhost:
            if mem is None:
                if not hasattr(self.memhost, "q"):
                    raise ValueError("el modelo memhost necesita mem")
                mem = qf2_memhost_ckpt.consultar(self, idx, COMPUTE_DTYPE)
            if MEMHOST_MASCARA and self.training:
                mem = mem * mascara_secuencias(idx, MEMHOST_MASCARA).to(mem.dtype)
            x_sin = x
            if self.memhost_cfg.entrada:
                x = self.memhost_sitios.entrada(mem, x)
        x0 = x
        if self.config.memhost and MEMHOST_X0_ESCALA != 1:
            x0 = x_sin + MEMHOST_X0_ESCALA * (x - x_sin)
        resumen_doc = None
        for i, block in enumerate(self.transformer.h):
            if self.resumen is not None and i == self.resumen.capa_salida:
                x = x + self.resumen.inyectar(resumen_doc)
            ve = tabla(self.value_embeds[str(i)], idx).to(x.dtype) if str(i) in self.value_embeds else None
            if ve is not None and registro:
                ve = anteponer(ve.new_zeros(ve.size(-1)), ve, entrena)
            gated = self.config.memhost and i in self.memhost_cfg.gated
            mem_v = self.memhost_sitios.delta(i, mem, x) if gated and self.memhost_cfg.via_de(i) == "v" else None
            x = block(x, x0, cos, sin, ve, mem_v)
            if gated and self.memhost_cfg.via_de(i) == "res":
                x = self.memhost_sitios.compuerta(i, mem, x)
            if self.pkm is not None and i == self.config.n_layer // 2:
                x = x + self.pkm(norm(x), tabla)
            if self.resumen is not None and i == self.resumen.capa_entrada:
                resumen_doc = self.resumen.media(norm(x), idx)
        x = norm(x)
        if registro and not entrena:
            x = x[:, 1:]
        if prefijo:
            x = torch.where(sin_bos.unsqueeze(-1), x[:, 1:], x[:, :-1])
        if self.mos is not None:
            return self.mos(x, self.lm_head, self.config.vocab_size, LOGIT_SOFTCAP, targets, loss_reduction, CEFA and self.training)
        if self.config.memcorpus:
            return self.salida_memcorpus(x, idx, targets, loss_reduction, corpus)
        if CEFA and self.training and targets is not None and loss_reduction == "mean":
            return cefa_perdida(self, x, targets, LOGIT_SOFTCAP)
        if CE_TROZOS and self.training and targets is not None and loss_reduction == "mean":
            if UNION is not None:
                return UNION(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets, idx == bos_token_id(), LOGIT_SOFTCAP,
                             self.config.vocab_size, CE_TROZOS)
            if BRIER is not None:
                return qf2_brier.ce_brier_trozos(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1),
                                                 LOGIT_SOFTCAP, self.config.vocab_size, CE_TROZOS, BRIER.lam)
            if OBJETIVO is not None:
                nll = ce_trozos_tokens(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP,
                                       self.config.vocab_size, CE_TROZOS)
                return OBJETIVO(nll.view_as(targets), targets != -1)
            if EQUILIBRIO is not None:
                nll = ce_trozos_tokens(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP,
                                       self.config.vocab_size, CE_TROZOS)
                return EQUILIBRIO(nll.view_as(targets), targets != -1, idx == bos_token_id())
            return ce_trozos(x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1), LOGIT_SOFTCAP, self.config.vocab_size, CE_TROZOS)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        if CE_LSE and self.training and loss_reduction == "mean":
            z, y = logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            vale = y != -1
            nll = torch.logsumexp(z, dim=-1) - z.gather(1, y.clamp_min(0).unsqueeze(1)).squeeze(1)
            return torch.where(vale, nll, 0.0).sum() / vale.sum()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config(
    depth: int = DEPTH,
    aspect_ratio: int = ASPECT_RATIO,
    head_dim: int = HEAD_DIM,
    sequence_len: int = SEQ_LEN,
) -> GPTConfig:
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        value_embeds=VALUE_EMBEDS,
        ve_paso=VE_PASO,
        rope_roles=ROPE_ROLES,
        registro=REGISTRO,
        prefijo_bos=PREFIJO_BOS,
        ngramas=NGRAM_FILAS,
        memoria=MEM_FILAS,
        memoria_ancho=(MEM_ANCHO or n_embd // 2) if MEM_FILAS else 0,
        memhost=MEMHOST,
        memcorpus=MEMCORPUS,
        pkm=PKM_FILAS,
        pkm_k=PKM_K,
        pkm_valor=PKM_VALOR,
        pkm_dk=PKM_DK,
        resumen=RESUMEN,
        diff_capas=DIFF_CAPAS,
        diff_signo=DIFF_SIGNO,
        mos=MOS,
        mlp_exp=MLP_EXP,
    )


# -----------------------------------------------------------------------------
# Optimizer

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        if OPT not in OPT_VALORES:
            raise ValueError(f"QF2_OPT must be one of {OPT_VALORES}, got {OPT!r}")
        if OPT_COMPILADO and (OPT or type(self) is MuonAdamW):
            raise ValueError("QF2_OPT_COMPILADO=1 needs the distributed optimizer (world > 1) and QF2_OPT unset")
        super().__init__(param_groups, defaults={})
        # Changing Python scalar arguments retraces Neuron eager optimizer ops.
        # Keep the changing schedule values in persistent 0-D CPU tensors so the
        # same device graphs can be reused throughout warmup and training.
        for group in self.param_groups:
            group["_lr_t"] = torch.tensor(float(group["lr"]), dtype=torch.float32)
            group["_wd_t"] = torch.tensor(float(group["weight_decay"]), dtype=torch.float32)
            if group["kind"] == "adamw":
                beta1, beta2 = group["betas"]
                group["_one_minus_beta1_t"] = torch.tensor(1 - beta1, dtype=torch.float32)
                group["_one_minus_beta2_t"] = torch.tensor(1 - beta2, dtype=torch.float32)
                group["_eps_t"] = torch.tensor(float(group["eps"]), dtype=torch.float32)
            elif group["kind"] == "muon":
                group["_momentum_t"] = torch.tensor(float(group["momentum"]), dtype=torch.float32)
                group["_one_minus_momentum_t"] = torch.tensor(1 - float(group["momentum"]), dtype=torch.float32)
                group["_one_minus_beta2_t"] = torch.tensor(1 - float(group["beta2"]), dtype=torch.float32)
                if OPT == "m2":
                    group["_m2_one_minus_beta2_t"] = torch.tensor(1 - OPT_BETA2, dtype=torch.float32)
                    group["_m2_eps_t"] = torch.tensor(OPT_EPS, dtype=torch.float32)

    @staticmethod
    def _m2_denominator(g, state, group):
        if "m2_v" not in state:
            state["m2_v"] = torch.zeros_like(g)
            state["m2_step"] = 0
            state["m2_bias_t"] = torch.tensor(0.0, dtype=torch.float32)
        state["m2_step"] += 1
        state["m2_bias_t"].fill_(1 - OPT_BETA2 ** state["m2_step"])
        state["m2_v"].lerp_(g.square(), group["_m2_one_minus_beta2_t"])
        return (state["m2_v"] / state["m2_bias_t"]).sqrt_().add_(group["_m2_eps_t"])

    @staticmethod
    def _adamw_update(p, grad, state, group):
        beta1, beta2 = group["betas"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        one_minus_beta1_t = group["_one_minus_beta1_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        eps_t = group["_eps_t"]
        lr_t.fill_(group["lr"])
        wd_t.fill_(group["weight_decay"])
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["bias1_t"] = torch.tensor(0.0, dtype=torch.float32)
            state["bias2_t"] = torch.tensor(0.0, dtype=torch.float32)
        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        bias1_t = state["bias1_t"]
        bias2_t = state["bias2_t"]
        bias1_t.fill_(1 - beta1 ** state["step"])
        bias2_t.fill_(1 - beta2 ** state["step"])
        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, one_minus_beta1_t)
        exp_avg_sq.lerp_(grad.square(), one_minus_beta2_t)
        update = exp_avg / bias1_t / ((exp_avg_sq / bias2_t).sqrt() + eps_t)
        p.sub_(lr_t * update)

    @torch.no_grad()
    def _adamw_step(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            self._adamw_update(p, p.grad, self.state[p], group)

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if "stacked_grads" not in state:
            state["stacked_grads"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "stacked_params" not in state:
            state["stacked_params"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)

        g = state["stacked_grads"]
        w = state["stacked_params"]
        for i, p in enumerate(params):
            if p.grad is None:
                g[i].zero_()
            else:
                g[i].copy_(p.grad)
            w[i].copy_(p)
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]

        momentum_t = group["_momentum_t"]
        one_minus_momentum_t = group["_one_minus_momentum_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        momentum_t.fill_(group["momentum"])
        one_minus_momentum_t.fill_(1 - group["momentum"])
        one_minus_beta2_t.fill_(1 - group["beta2"])
        lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd_t.fill_(group["weight_decay"])

        m2_den = self._m2_denominator(g, state, group) if OPT == "m2" else None
        momentum_buffer.lerp_(g, one_minus_momentum_t)
        g = g.lerp_(momentum_buffer, momentum_t)
        if m2_den is not None:
            g = g.div_(m2_den)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        g = x

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, one_minus_beta2_t)
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        mask = (g * w) >= 0
        w.sub_(lr_t * g + lr_t * wd_t * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


class DistMuonAdamW(MuonAdamW):
    """ZeRO-2-style distributed variant with replicated FP32 parameters.

    AdamW tensors are sharded along their first dimension. Muon tensors are
    sharded by whole matrices within each equal-shape group. Gradients use
    reduce-scatter, only the owned shard carries optimizer state, and updated
    parameters are all-gathered back to every rank.

    Muon groups execute serially and share one communication slab. This is the
    memory-critical part of the port: allocating one padded slab per matrix
    shape would erase most of the optimizer-state saving on Trainium.
    """

    handles_gradient_sync = True

    def __init__(self, param_groups):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DistMuonAdamW requires an initialized process group")
        if COM_GATHER_BF16 and COMPUTE_DTYPE != torch.bfloat16:
            raise ValueError(f"QF2_COM_GATHER_BF16=1 needs bf16 compute (the forward casts weights to bf16), got {COMPUTE_DTYPE}")
        if COM_RS_BF16 and not COM_MUON:
            raise ValueError("QF2_COM_RS_BF16=1 needs QF2_COM_MUON=1 (the per-shape Muon path reduces in fp32)")
        if PLANO_TRANSITORIO and not COM_MUON:
            raise ValueError("QF2_PLANO_TRANSITORIO=1 needs QF2_COM_MUON=1 (the slab belongs to the batched Muon path)")
        if COM_FRANJAS and (not COM_MUON or COM_RS_BF16):
            raise ValueError("QF2_COM_FRANJAS needs QF2_COM_MUON=1 and QF2_COM_RS_BF16=0 (strips move the fp32 batched Muon slab)")
        super().__init__(param_groups)
        self._compilado = qf2_opt_compilado.Compilado(self.param_groups[0]["params"][0].device) if OPT_COMPILADO else None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._adamw_grad_slices = {}
        self._shared_muon_buffers = {}
        muon_groups = [g for g in self.param_groups if g["kind"] == "muon"]
        if COM_MUON and COM_FRANJAS:
            self._reparto = qf2_franjas.RepartoFranjas(muon_groups, self.rank, self.world_size, COM_FRANJAS)
        else:
            self._reparto = qf2_reparto.RepartoMuon(muon_groups, self.rank, self.world_size, PLANO_TRANSITORIO) if COM_MUON else None

    def restaurar_fp32(self):
        qf2_reparto.restaurar_fp32(self, self._reparto)

    def _shared_muon_buffer(self, name, rows, shape, dtype, device, *, zero=False):
        need = rows * shape[0] * shape[1]
        buf = self._shared_muon_buffers.get(name)
        if buf is None or buf.numel() < need or buf.dtype != dtype or buf.device != device:
            factory = torch.zeros if zero else torch.empty
            buf = factory(need, dtype=dtype, device=device)
            self._shared_muon_buffers[name] = buf
        return buf.flatten()[:need].view(rows, *shape)

    @staticmethod
    def _copy_muon_params(params, stacked):
        if not params:
            return
        if params[0].device.type == "neuron":
            per = params[0].numel()
            torch.split_with_sizes_copy(
                stacked.reshape(-1)[:len(params) * per],
                [per] * len(params),
                out=[p.data.view(-1) for p in params],
            )
        else:
            torch._foreach_copy_(params, list(stacked[:len(params)].unbind(0)))

    @torch.no_grad()
    def _distributed_adamw_step(self, group):
        if self._compilado is not None:
            return self._distributed_adamw_step_compilado(group)
        for param_index, p in enumerate(group["params"]):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grad = p.grad
            if p.numel() < 1024:
                if not COM_ESCALARES:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                self._adamw_update(p, grad, self.state[p], group)
                p.grad = None
                continue

            if p.shape[0] % self.world_size:
                raise ValueError(
                    f"AdamW parameter first dimension {p.shape[0]} must be divisible "
                    f"by world size {self.world_size}"
                )
            rows = p.shape[0] // self.world_size
            grad_slice = self._adamw_grad_slices.get(id(p))
            if grad_slice is None or grad_slice.shape != grad[:rows].shape:
                grad_slice = torch.empty_like(grad[:rows])
                self._adamw_grad_slices[id(p)] = grad_slice
            if COM_RS_BF16:
                qf2_reparto.reducir_tabla_bf16(grad_slice, grad)
            else:
                dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG)

            start = self.rank * rows
            p_slice = p[start:start + rows]
            self._adamw_update(p_slice, grad_slice, self.state[p], group)
            if COM_GATHER_BF16:
                qf2_reparto.reunir_tabla_bf16(p, p_slice, grad, start, rows)
            else:
                dist.all_gather_into_tensor(p, p_slice)
            p.grad = None

    @torch.no_grad()
    def _distributed_adamw_step_compilado(self, group):
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        pequenos = [p for p in group["params"] if p.numel() < 1024]
        if pequenos:
            if not COM_ESCALARES:
                for p in pequenos:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            self._compilado.adamw_paso(pequenos, [p.grad for p in pequenos], [self.state[p] for p in pequenos], group,
                                       ("adamw", id(group)))
            for p in pequenos:
                p.grad = None
        for p in group["params"]:
            if p.numel() < 1024:
                continue
            if p.shape[0] % self.world_size:
                raise ValueError(f"AdamW parameter first dimension {p.shape[0]} must be divisible by world size {self.world_size}")
            grad = p.grad
            rows = p.shape[0] // self.world_size
            grad_slice = self._adamw_grad_slices.get(id(p))
            if grad_slice is None or grad_slice.shape != grad[:rows].shape:
                grad_slice = torch.empty_like(grad[:rows])
                self._adamw_grad_slices[id(p)] = grad_slice
            if COM_RS_BF16:
                qf2_reparto.reducir_tabla_bf16(grad_slice, grad)
            else:
                dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG)
            start = self.rank * rows
            p_slice = p[start:start + rows]
            self._compilado.adamw_paso([p_slice], [grad_slice], [self.state[p]], group, ("adamw", id(group), id(p)))
            if COM_GATHER_BF16:
                qf2_reparto.reunir_tabla_bf16(p, p_slice, grad, start, rows)
            else:
                dist.all_gather_into_tensor(p, p_slice)
            p.grad = None

    @torch.no_grad()
    def _distributed_muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        shape = tuple(p0.shape)
        if any(tuple(p.shape) != shape for p in params):
            raise ValueError("Muon groups must contain equal-shape matrices")

        chunk_size = (len(params) + self.world_size - 1) // self.world_size
        padded_count = chunk_size * self.world_size
        dtype, device = p0.dtype, p0.device
        stacked = self._shared_muon_buffer(
            "stacked", padded_count, shape, dtype, device, zero=True
        )
        grad_chunk = self._shared_muon_buffer(
            "grad_chunk", chunk_size, shape, dtype, device
        )
        updated = self._shared_muon_buffer(
            "updated", chunk_size, shape, dtype, device
        )

        for i, p in enumerate(params):
            if p.grad is None:
                stacked[i].zero_()
            else:
                stacked[i].copy_(p.grad)
            p.grad = None
        dist.reduce_scatter_tensor(grad_chunk, stacked, op=dist.ReduceOp.AVG)

        start = self.rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start))
        if num_owned:
            for i in range(num_owned):
                updated[i].copy_(params[start + i])
            self._muon_owned(group, shape, grad_chunk[:num_owned], updated[:num_owned], self.state[p0])

        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        if COM_GATHER_BF16:
            qf2_reparto.reunir_muon_bf16(stacked, updated, params, start, num_owned)
        else:
            dist.all_gather_into_tensor(stacked, updated)
            self._copy_muon_params(params, stacked)

    @torch.no_grad()
    def _muon_owned(self, group, shape, g, w, state):
        num_owned, dtype, device = g.shape[0], g.dtype, g.device
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_owned, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_owned, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_owned, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=torch.float32, device=device
            )
        if self._compilado is not None:
            self._compilado.muon_paso(group, shape, g, w, state["momentum_buffer"], state["second_momentum_buffer"],
                                      POLAR_EXPRESS_COEFFS[:group["ns_steps"]], COMPUTE_DTYPE == torch.bfloat16)
            return

        momentum_t = group["_momentum_t"]
        one_minus_momentum_t = group["_one_minus_momentum_t"]
        one_minus_beta2_t = group["_one_minus_beta2_t"]
        lr_t = group["_lr_t"]
        wd_t = group["_wd_t"]
        momentum_t.fill_(group["momentum"])
        one_minus_momentum_t.fill_(1 - group["momentum"])
        one_minus_beta2_t.fill_(1 - group["beta2"])
        lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd_t.fill_(group["weight_decay"])

        momentum = state["momentum_buffer"]
        second = state["second_momentum_buffer"]
        m2_den = self._m2_denominator(g, state, group) if OPT == "m2" else None
        momentum.lerp_(g, one_minus_momentum_t)
        g = g.lerp_(momentum, momentum_t)
        if m2_den is not None:
            g = g.div_(m2_den)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        # A one-matrix shard would lower the 3-D products below as batch-1
        # bmm, which the Trainium lazy prologue cannot merge. Use ordinary
        # 2-D matmul for that case and restore the batch dimension after the
        # Polar Express iterations.
        squeezed = x.dim() == 3 and x.size(0) == 1
        if squeezed:
            x = x.squeeze(0)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        if squeezed:
            x = x.unsqueeze(0)
        g = x

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, one_minus_beta2_t)
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        mask = (g * w) >= 0
        w.sub_(lr_t * g + lr_t * wd_t * w * mask)

    @torch.no_grad()
    def step(self):
        # AdamW state is small after row sharding, so process it first. Muon
        # groups remain serial to reuse the large padded communication slab.
        if COM_ESCALARES:
            qf2_reparto.promediar_pequenos(self.param_groups)
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._distributed_adamw_step(group)
        if self._reparto is not None:
            self._reparto.paso(self, COM_GATHER_BF16, COM_RS_BF16)
            return
        for group in self.param_groups:
            if group["kind"] == "muon":
                self._distributed_muon_step(group)


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def optimizer_handles_gradient_sync(optimizer: torch.optim.Optimizer) -> bool:
    return bool(getattr(optimizer, "handles_gradient_sync", False))


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict, extra: dict | None = None, anticipado=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    estado = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if model.config.memhost and qf2_memhost_cfg.CLAVE_Q not in estado and qf2_memhost_cfg.CLAVE_Q not in (extra or {}):
        extra = {**(extra or {}), **qf2_memhost_ckpt.vacias(model.memhost_cfg)}
    if model.config.memcorpus and not all(k in (extra or {}) for k in qf2_corpus_mezcla.claves(model.memcorpus_cfg)):
        extra = {**(extra or {}), **qf2_corpus_mezcla.vacias(model.memcorpus_cfg)}
    payload = {
        "model_state": {**estado, **(extra or {})},
        "model_config": config_dict(model.config),
        "meta": meta,
    }
    if anticipado is not None and anticipado.empezado:
        payload["model_state"] = anticipado.ordenar(payload["model_state"])
        anticipado.terminar(payload)
        return
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    cabecera = qf2_memoria_anfitrion.abrir(checkpoint_path)
    if qf2_memhost_ckpt.usa_memhost(cabecera) or cabecera["model_config"].get("memcorpus"):
        return load_for_eval_memhost(checkpoint_path, device)
    if qf2_memoria_anfitrion.usa_anfitrion(checkpoint_path, MEM_EVAL_ANFITRION_BYTES):
        return load_for_eval_anfitrion(checkpoint_path, device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    tabla_mem = payload["model_state"].get("memoria.2.weight")
    model = GPT(config, memoria_dtype=tabla_mem.dtype if tabla_mem is not None else torch.float32)
    model.init_weights(memoria=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_for_eval_memhost(checkpoint_path: str, device: torch.device) -> nn.Module:
    qf2_memoria_anfitrion.precargar(checkpoint_path)
    payload = qf2_memoria_anfitrion.abrir(checkpoint_path)
    state = payload["model_state"]
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    propias = set(qf2_memhost_ckpt.claves() if config.memhost else []) | set(qf2_corpus_mezcla.claves(model.memcorpus_cfg) if config.memcorpus else [])
    res = model.load_state_dict({k: v for k, v in state.items() if k not in propias}, strict=False)
    if res.unexpected_keys or res.missing_keys:
        raise RuntimeError(f"claves inesperadas al cargar sin tablas: faltan={res.missing_keys} sobran={res.unexpected_keys}")
    model.to(device)
    if config.memhost:
        qf2_memhost_ckpt.asignar(model, state)
    if config.memcorpus:
        qf2_corpus_mezcla.asignar(model, state)
    model.eval()
    return model


def load_for_eval_anfitrion(checkpoint_path: str, device: torch.device) -> nn.Module:
    qf2_memoria_anfitrion.precargar(checkpoint_path)
    payload = qf2_memoria_anfitrion.abrir(checkpoint_path)
    state = payload["model_state"]
    config = GPTConfig(**payload["model_config"])
    model = GPT(config, memoria_dtype=state["memoria.2.weight"].dtype, memoria_device="meta")
    model.init_weights(memoria=False)
    qf2_memoria_anfitrion.cargar_resto(model, state)
    tablas, model.memoria = model.memoria, nn.ModuleDict()
    model.to(device)
    model.memoria = tablas
    qf2_memoria_anfitrion.asignar(model, state)
    model.memoria_anfitrion = True
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Training

def lr_bajada(remaining: float) -> float:
    if LR_SHAPE == "sqrt":
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * (1.0 - math.sqrt(max(0.0, 1.0 - remaining)))
    return remaining + (1.0 - remaining) * FINAL_LR_FRAC


def lr_multiplier(step: int, target_steps: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    warmdown_steps = round(WARMDOWN_RATIO * target_steps)
    if step <= target_steps - warmdown_steps:
        return 1.0
    return lr_bajada((target_steps - step) / max(1, warmdown_steps))


def lr_multiplier_reloj(step: int, progress: float) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    remaining = max(0.0, 1.0 - progress) / WARMDOWN_RATIO
    if remaining >= 1.0:
        return 1.0
    return lr_bajada(remaining)


def suelo_momentum(warmup_frac: float, warmdown_remaining: float) -> float:
    if warmup_frac < 1.0:
        return SUELO_MOMENTUM_START + (SUELO_MOMENTUM_PEAK - SUELO_MOMENTUM_START) * max(0.0, warmup_frac)
    if warmdown_remaining < 1.0:
        return SUELO_MOMENTUM_END + (SUELO_MOMENTUM_PEAK - SUELO_MOMENTUM_END) * max(0.0, warmdown_remaining)
    return SUELO_MOMENTUM_PEAK


def muon_momentum(step: int, target_steps: int) -> float:
    if not SUELO_MOM:
        return MUON_MOMENTUM
    warmdown_steps = round(WARMDOWN_RATIO * target_steps)
    warmup_frac = step / max(1.0, SUELO_MOMENTUM_WARMUP_FRAC * target_steps)
    return suelo_momentum(warmup_frac, (target_steps - step) / max(1, warmdown_steps))


def muon_momentum_reloj(progress: float) -> float:
    if not SUELO_MOM:
        return MUON_MOMENTUM
    return suelo_momentum(progress / SUELO_MOMENTUM_WARMUP_FRAC, max(0.0, 1.0 - progress) / WARMDOWN_RATIO)


def wd_muon(progress: float) -> float:
    if not WD_COS:
        return WD_PICO
    return WD_PICO * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def reloj_progress_from(control: torch.Tensor, max_train_seconds: float) -> float:
    elapsed_ms, slowest_ms = control[2:].tolist()
    return ((elapsed_ms + 2 * slowest_ms) / 1000.0 + CHECKPOINT_RESERVE_SECONDS) / max_train_seconds


def main() -> None:
    qf2_hilos_aten.fijar_proceso()
    parser = argparse.ArgumentParser(description="Neuron Competition Round 2 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--aspect-ratio", type=int, default=ASPECT_RATIO)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--device-batch-size", type=int, default=DEVICE_BATCH_SIZE)
    parser.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_NUM_STEPS", NUM_STEPS)),
                        help="explicit step override; <=0 derives steps from --target-param-data-ratio")
    parser.add_argument("--target-param-data-ratio", type=float,
                        default=float(os.environ.get("NEURON_COMPETITION_R2_TARGET_PARAM_DATA_RATIO", TARGET_PARAM_DATA_RATIO)),
                        help="target token IDs per scaling parameter (Transformer matrices + lm_head)")
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R2_OUT_DIR", "out"))
    parser.add_argument(
        "--eval-public",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NEURON_COMPETITION_R2_EVAL_PUBLIC", "0") != "0",
        help="run the optional development-shard BPB evaluation after checkpointing",
    )
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R2_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action="store_true", default=os.environ.get("NEURON_COMPETITION_R2_COMPILE", "0") != "0",
                        help="enable torch.compile on the full model")
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic training step without reading training shards")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R2_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    args = parser.parse_args()
    if LR_SCHEDULE not in ("steps", "reloj"):
        raise ValueError(f"QF2_LR_SCHEDULE must be 'steps' or 'reloj', got {LR_SCHEDULE!r}")
    if OPT not in OPT_VALORES:
        raise ValueError(f"QF2_OPT must be one of {OPT_VALORES}, got {OPT!r}")
    if EMB_SCATTER and EMB_DETACH:
        raise ValueError("QF2_EMB_SCATTER=1 and QF2_EMB_DETACH=1 both replace the table backward")
    if EMB_SCATTER_GATE and not EMB_SCATTER:
        raise ValueError("QF2_EMB_SCATTER_GATE=1 needs QF2_EMB_SCATTER=1")

    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2

    device_type = detect_device_type(args.device_type)
    constructor = None
    if MEMCORPUS and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tok_mc = ensure_tokenizer(build_if_missing=False)
        shards_mc = qf2_corpus_constructor.shards_del_constructor(list_shard_paths(PUBLIC_TRAIN_SHARDS, require_all=False), MEMCORPUS_EXCLUIR)
        constructor = qf2_corpus_constructor.Constructor(qf2_corpus_mezcla.desde_texto(MEMCORPUS), Path(MEMCORPUS_RUTA), shards_mc,
                                                         MEMCORPUS_PROCESOS, tok_mc.get_bos_token_id(), tok_mc)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE, CHECKPOINT_RESERVE_SECONDS
    COMPUTE_DTYPE = compute_dtype_for(device)
    if MEMHOST:
        CHECKPOINT_RESERVE_SECONDS = max(CHECKPOINT_RESERVE_SECONDS, MEMHOST_RESERVA_S)
    if MEMCORPUS:
        CHECKPOINT_RESERVE_SECONDS += MEMCORPUS_RESERVA_S

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    if CEFA:
        print0(cefa_descripcion())
    if SUELO_MOM or SUELO_ESC or SUELO_LR or SUELO_INIT or VALUE_EMBEDS or OPT or SEED != 42:
        print0(f"qf2_suelo: a={int(SUELO_A)} mom={int(SUELO_MOM)} esc={int(SUELO_ESC)} opt={f'{OPT}:{OPT_BETA2}:{OPT_EPS}' if OPT else '-'} lr={int(SUELO_LR)} init={int(SUELO_INIT)} ve={int(VALUE_EMBEDS)} seed={SEED} lr_scale={(args.total_batch_size / SUELO_B_REF) ** 0.5 if SUELO_LR else 1.0:.6f}")

    print0(f"qf2_lr: forma={LR_SHAPE} bajada={WARMDOWN_RATIO!r} final={FINAL_LR_FRAC!r} lrm(r=1,0.5,0)={lr_bajada(1.0):.6g},{lr_bajada(0.5):.6g},{lr_bajada(0.0):.6g}")
    print0(f"qf2_wd: cos={int(WD_COS)} pico={WD_PICO!r} muon_wd(p=0,0.5,1)={wd_muon(0.0):.6g},{wd_muon(0.5):.6g},{wd_muon(1.0):.6g} p={'reloj_progress' if LR_SCHEDULE == 'reloj' else 'step/target_steps'}")
    if FP8_MLP:
        print0(f"qf2_fp8_mlp: {FP8_RESUMEN}")
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        sequence_len=qf2_cola.maximo(args.seq_len) if COLA4096 else args.seq_len,
    )
    print0(f"model_config={json.dumps(config_dict(config), indent=2)}")
    repartida = bool(config.memoria) and dist.is_available() and dist.is_initialized()
    if repartida and (PROMEDIO or PUNTOS or args.eval_public):
        raise ValueError("QF2_MEM_FILAS repartida es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S y --eval-public")
    if config.memhost and (PROMEDIO or PUNTOS or args.eval_public or COLA_LARGA or not PRECARGA or repartida):
        raise ValueError("QF2_MEMHOST necesita QF2_PRECARGA=1 y es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S, --eval-public, la cola larga (QF2_SEQ_MAX, QF2_COLA_FRAC, QF2_CALIENTA_LARGO) y la memoria repartida")
    if config.memcorpus and (PROMEDIO or PUNTOS or FLUJO or COLA_LARGA or not PRECARGA or repartida):
        raise ValueError("QF2_MEMCORPUS necesita QF2_PRECARGA=1 y es incompatible con QF2_PROMEDIO, QF2_PUNTOS_S, QF2_FLUJO, la cola larga y la memoria repartida")
    if PESO_CTX.activo and (config.memoria or config.memhost or config.memcorpus or config.mos or config.registro or config.prefijo_bos or config.ngramas
                            or CEFA or CE_TROZOS or CE_LSE or COLA_LARGA or args.seq_len % PESO_CTX.k):
        raise ValueError("QF2_PESO_CTX es incompatible con memoria, memhost, memcorpus, mos, registro, prefijo_bos, ngramas, QF2_CEFA, QF2_CE_TROZOS, QF2_CE_LSE y la cola larga, y exige seq_len multiplo de K")
    model = GPT(config, repartida=repartida)
    model.init_weights()
    tabla_mc = None
    if config.memcorpus:
        qf2_corpus_constructor.esperar_creada(Path(MEMCORPUS_RUTA))
        tabla_mc = qf2_corpus_constructor.abrir(Path(MEMCORPUS_RUTA), model.memcorpus_cfg)
        model.memcorpus_tabla = tabla_mc
        if constructor is not None:
            print0(constructor.descripcion())
            print0(f"qf2_memcorpus: tronco={MEMCORPUS_TRONCO} contexto={model.memcorpus_cfg.contexto}")
    anticipado = None
    if constructor is not None and MEMCORPUS_ANTICIPADO_S > 0:
        anticipado = qf2_guardado_anticipado.Anticipado(Path(args.out_dir) / "final.pt")
    if config.memhost and model.memhost_cfg.canon:
        model.memhost.canon.copy_(torch.from_numpy(qf2_memhost_canon.construir(tokenizer.decode, config.vocab_size, model.memhost_bos)))
    inicio = qf2_inicio.abrir(INICIO_DESDE, config_dict(config)) if INICIO_DESDE else None
    if inicio is not None:
        print0(qf2_inicio.cargar_tronco(model, inicio, BANCO_LRM_TRAMO))
    model.to(device)
    mem_obj = qf2_memoria.Memoria(config.memoria, config.memoria_ancho, device, SEED, model.memoria_bos, MEM_TRANSPORTE) if repartida else None
    mem_opt = qf2_memoria_opt.AdamShard(mem_obj.parametros(), model.memoria_lr(args.total_batch_size), MuonAdamW._adamw_update) if repartida else None
    num_params = sum(p.numel() for p in model.parameters())
    scaling_param_counts = model.num_scaling_params()
    num_scaling_params = scaling_param_counts["total"]
    flops_per_token = model.estimate_flops()
    if repartida:
        print0(f"qf2_memoria: repartida filas={config.memoria} ancho={config.memoria_ancho} rangos={world_size} filas_por_rango={mem_obj.r} params_globales={2 * config.memoria * config.memoria_ancho:,} transporte={MEM_TRANSPORTE}")
    print0(f"num_params={num_params:,} scaling_params={num_scaling_params:,} flops_per_token={flops_per_token:e}")
    if EMB_RUTA:
        print0(emb_descripcion(device, EMB_DETACH, 1 + len(model.value_embeds) + len(model.ngramas), model.transformer.wte.weight.shape, COMPUTE_DTYPE))
        if EMB_SCATTER and device.type == "neuron":
            emb_preparar()

    world_tokens_per_micro = args.device_batch_size * args.seq_len * world_size
    if args.total_batch_size % world_tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_size={args.total_batch_size} must divide micro tokens={world_tokens_per_micro}"
        )
    grad_accum_steps = args.total_batch_size // world_tokens_per_micro
    local_step_batch_size = args.device_batch_size * grad_accum_steps
    cola = qf2_cola.Cola(args.seq_len, args.device_batch_size, args.total_batch_size, world_size, config.sequence_len, print0) if COLA4096 else None
    loader_rows, loader_len = cola.cargador(local_step_batch_size, args.seq_len) if cola else (local_step_batch_size, args.seq_len)
    mem_forma = (loader_rows * loader_len // args.seq_len, args.seq_len) if cola else (loader_rows, loader_len)
    if cola:
        print0(f"{qf2_cola.descripcion()} loader={loader_rows}x{loader_len} grad_accum_split={cola.micro_partido} grad_accum_long={cola.micro_largo}")
    if args.num_steps > 0:
        target_steps = args.num_steps
        horizon_source = "explicit steps"
    elif LR_SCHEDULE == "reloj":
        target_steps = RELOJ_STEP_CEILING
        horizon_source = "reloj"
    else:
        if args.target_param_data_ratio <= 0:
            raise ValueError("--target-param-data-ratio must be positive when --num-steps is not set")
        target_tokens = int(args.target_param_data_ratio * num_scaling_params)
        target_steps = target_tokens // args.total_batch_size
        if target_steps <= 0:
            raise ValueError("target ratio produces zero optimizer steps")
        horizon_source = "data:param ratio"
    target_tokens = target_steps * args.total_batch_size
    actual_target_ratio = target_tokens / num_scaling_params
    print0(
        f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} "
        f"local_step_batch_size={local_step_batch_size}"
    )
    print0(
        f"training_horizon={horizon_source} target_steps={target_steps:,} "
        f"target_tokens={target_tokens:,} target_scaling_ratio={actual_target_ratio:.4f} "
        f"max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s"
    )

    orig_model = model
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    if args.compile_only:
        # Match the real loader's packed local-step input and int32 token dtype so
        # the timed process can reuse this exact graph specialization.
        synthetic_host_x = torch.zeros((loader_rows, loader_len), dtype=torch.int32)
        synthetic_host_y = torch.zeros((loader_rows, loader_len), dtype=torch.long)
        synthetic_x = synthetic_host_x.to(device)
        synthetic_y = synthetic_host_y.to(device)
        optimizer = orig_model.setup_optimizer(args.total_batch_size)
        synthetic_e = torch.zeros(*mem_forma, orig_model.memhost_cfg.ancho_e, dtype=COMPUTE_DTYPE, device=device) if config.memhost else None
        hojas_sinteticas = qf2_memhost_hojas.Hojas() if config.memhost else None
        for synthetic_step in (cola.pasos_compilacion() if cola else (0,)):
            model.zero_grad(set_to_none=True)
            xs, ys, n_micro, filas = cola.lote(synthetic_x, synthetic_y, synthetic_step, 0.0, grad_accum_steps) if cola else (synthetic_x, synthetic_y, grad_accum_steps, args.device_batch_size)
            if EQUILIBRIO is not None:
                qf2_equilibrio.fijar(cola is not None and qf2_cola.CARGADOR_LARGO and cola.forma is False)
            if mem_opt is not None:
                mem_opt.zero_grad()
            hs = hojas_sinteticas.repartir(synthetic_e, n_micro, filas) if hojas_sinteticas is not None else None
            for micro_step in range(n_micro):
                start = micro_step * filas
                end = start + filas
                loss = perdida_micro(model, mem_obj, xs[start:end], ys[start:end], hs[micro_step] if hs is not None else None, None, synthetic_step, micro_step)
                (loss / n_micro).backward()
            if not optimizer_handles_gradient_sync(optimizer):
                sync_gradients(orig_model)
            optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    # Transfer one complete local optimizer-step batch before launching any
    # microbatch graph. Reusing and overwriting a single device input between
    # asynchronous Neuron graph calls forces a synchronization per microbatch.
    memhost, memhost_lr = None, 0.0
    if FLUJO:
        print0(qf2_flujo.descripcion(SEED, loader_len))
    if PUNTOS:
        train_loader = qf2_flujo.make_flujo_dataloader(tokenizer, loader_rows, loader_len, device, SEED, print0 if FLUJO else (lambda *_: None),
                                                       p=None if FLUJO else 0, reanudable=True)
    elif PRECARGA:
        if config.memhost:
            memhost = qf2_memhost_hilo.Memhost(orig_model.memhost_cfg, MEMHOST_RUTA, rank, world_size, mem_forma, COMPUTE_DTYPE,
                                               orig_model.memhost_bos, MEMHOST_RETRASO, MEMHOST_OPT,
                                               orig_model.memhost.canon.cpu().numpy() if orig_model.memhost_cfg.canon else None,
                                               inicio=qf2_inicio.tablas(inicio, print0))
            memhost_lr = MEMHOST_LR or orig_model.memoria_lr(args.total_batch_size)
            if MEMHOST_DECAY:
                qf2_memhost_opt.DECAY = float(MEMHOST_DECAY)
            print0(f"qf2_memhost: cfg={config.memhost} filas={orig_model.memhost_cfg.filas:,} retraso={MEMHOST_RETRASO} opt={MEMHOST_OPT} lr={memhost_lr} decay={qf2_memhost_opt.DECAY} ruta={MEMHOST_RUTA} reserva_s={CHECKPOINT_RESERVE_SECONDS}")
        corpus_mc = (lambda x: qf2_corpus_mezcla.consultar(tabla_mc, x.numpy(), orig_model.memcorpus_bos, orig_model.memcorpus_cfg.minimo)) if tabla_mc is not None else None
        train_loader = qf2_precarga.Precarga(lambda: qf2_filas.cargador(make_dataloader(tokenizer, loader_rows, loader_len, "train", "cpu", shard_ids=qf2_agrupa_fuente.shards(qf2_inicio.shards_banco())), tokenizer, rank, world_size, SEED, args.seq_len, AGRUPA, RECOLOCADO, RECOLOCADO_CE, RECOLOCADO_RAFAGA, RECOLOCADO_FORMATO, PARES_TRAMO, PARES_ETIQUETAS), device, memhost=memhost,
                                            forma_mem=mem_forma, corpus=corpus_mc)
        print0(f"qf2_precarga: hilo con cola de {qf2_precarga.PROFUNDIDAD} lotes de {loader_rows}x{loader_len} en CPU, copia al dispositivo en el hilo principal")
    else:
        train_loader = qf2_flujo.make_flujo_dataloader(tokenizer, loader_rows, loader_len, device, SEED, print0) if FLUJO else make_dataloader(tokenizer, loader_rows, loader_len, "train", device)
    step_x, step_y, loader_state = next(train_loader)
    optimizer = orig_model.setup_optimizer(args.total_batch_size)
    restaurar_fp32 = optimizer.restaurar_fp32 if COM_GATHER_BF16 and isinstance(optimizer, DistMuonAdamW) else None
    if COM_ESCALARES or COM_MUON or COM_GATHER_BF16 or COM_RS_BF16:
        print0(qf2_reparto.descripcion(COM_ESCALARES, COM_MUON, COM_GATHER_BF16, optimizer, COM_RS_BF16))
    if OPT_COMPILADO and getattr(optimizer, "_compilado", None) is not None:
        print0(optimizer._compilado.descripcion())
    promedio = Promedio(orig_model.parameters(), PROMEDIO_MODO == "host", antes=restaurar_fp32) if PROMEDIO else None
    if promedio is not None:
        print0(promedio.descripcion())
    guarda = Guarda(list(orig_model.parameters()) + (mem_obj.parametros() if mem_obj else []), TRAIN_STARTUP_STEPS_EXCLUDED, rank) if GUARDA else None
    if guarda is not None:
        print0(guarda.descripcion())
    puntos = qf2_puntos.Puntos(PUNTOS_S, REANUDA, args.out_dir, rank, world_size, args, print0) if PUNTOS else None
    puntos_extras = {"guarda": guarda, "cola": cola, "promedio": promedio}
    if puntos is not None:
        print0(puntos.descripcion())
    reanudado = puntos.reanudar(orig_model, optimizer, train_loader, puntos_extras, restaurar_fp32 is not None, device) if puntos else None
    if reanudado:
        step_x, step_y, loader_state = reanudado["lote"]
    if EMB_SCATTER_GATE and rank == 0:
        if not emb_puerta(device, step_x[:args.device_batch_size], orig_model.transformer.wte.weight.shape, COMPUTE_DTYPE):
            raise SystemExit(3)
    if UNION is not None:
        print0(UNION.descripcion(RUIDO if RUIDO.activo else None))
    if UNION is None and (OBJETIVO is not None or RUIDO.activo):
        print0(qf2_objetivos.descripcion(OBJETIVO, RUIDO))
    if UNION is None and BRIER is not None:
        print0(BRIER.descripcion())
    if UNION is None and EQUILIBRIO is not None:
        print0(EQUILIBRIO.descripcion())
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    reloj_progress = 0.0
    paso_inicial = 0
    if reanudado:
        step, smooth_loss, total_train_tokens = reanudado["step"], reanudado["smooth_loss"], reanudado["total_train_tokens"]
        startup_allowance_est, reloj_progress = reanudado["startup_allowance_est"], reanudado["reloj_progress"]
        recent_step_times = list(reanudado["recent_step_times"])
        budget_started = time.monotonic() - reanudado["entrenado_s"]
        paso_inicial = step
    hojas = qf2_memhost_hojas.Hojas() if memhost is not None else None
    reloj_log = open(f"{RELOJ_LOG}.rank{rank}", "a" if reanudado else "w", encoding="utf-8") if RELOJ_LOG else None

    while step < target_steps:
        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if mem_opt is not None:
            mem_opt.zero_grad()
        last_loss = None
        xs, ys, n_micro, filas = cola.lote(step_x, step_y, step, reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps, grad_accum_steps) if cola else (step_x, step_y, grad_accum_steps, args.device_batch_size)
        if EQUILIBRIO is not None:
            qf2_equilibrio.fijar(cola is not None and qf2_cola.CARGADOR_LARGO and cola.forma is False)
        hs = hojas.repartir(train_loader.e, n_micro, filas) if hojas is not None else None
        cs = (train_loader.c.reshape(xs.shape), train_loader.k.reshape(xs.shape)) if tabla_mc is not None else None
        for micro_step in range(n_micro):
            start = micro_step * filas
            end = start + filas
            loss = perdida_micro(model, mem_obj, xs[start:end], ys[start:end], hs[micro_step] if hs is not None else None,
                                 (cs[0][start:end], cs[1][start:end]) if cs is not None else None, step, micro_step)
            last_loss = loss.detach()
            (loss / n_micro).backward()

        saltar = guarda is not None and guarda.saltar(step)
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not saltar and not optimizer_handles_gradient_sync(optimizer):
            sync_gradients(orig_model)
        if RELOJ_TRAMOS:
            synchronize(device)
            t_fb = time.monotonic()

        if LR_SCHEDULE == "reloj":
            lrm = lr_multiplier_reloj(step, reloj_progress)
            momentum = muon_momentum_reloj(reloj_progress)
        else:
            lrm = lr_multiplier(step, target_steps)
            momentum = muon_momentum(step, target_steps)
        if BANCO_LRM_TRAMO is not None:
            lrm = qf2_inicio.multiplicador(BANCO_LRM_TRAMO, reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = wd_muon(reloj_progress if LR_SCHEDULE == "reloj" else step / target_steps)
        if mem_opt is not None:
            for group in mem_opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        if memhost is not None:
            g_mem = None if saltar else hojas.gradiente(train_loader.e.shape[0])
            if not saltar and g_mem is None:
                raise RuntimeError("qf2_memhost: el gradiente de los vectores recuperados no llego a las hojas")
            t_copia = time.monotonic()
            g_mem = None if saltar else g_mem.detach().to("cpu")
            memhost.anotar("copia_d2h", time.monotonic() - t_copia)
        if not saltar:
            optimizer.step()
            if mem_opt is not None:
                mem_opt.step()
        if memhost is not None:
            memhost.postear(g_mem, memhost_lr * lrm, lrm)

        synchronize(device)
        if RELOJ_TRAMOS:
            t_opt = time.monotonic()
        if not saltar:
            record_step_boundary(step)
        step_x, step_y, loader_state = next(train_loader)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        if RELOJ_TRAMOS and rank == 0:
            print(f"RELOJ_TRAMOS {step} fb={t_fb - t0:.4f} opt={t_opt - t_fb:.4f} resto={t0 + dt - t_opt:.4f} dt={dt:.4f}", flush=True)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped), so
            # restart the budget clock here to use the full charged budget.
            # If startup exceeded the cap, the excess is already charged:
            # back-date the clock by that excess, with a margin for the
            # launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(
                0.0,
                observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
            )
            budget_started = now - excess
            if constructor is not None:
                constructor.empezar()
        if puntos is not None and puntos.calienta():
            recent_step_times.pop()
            budget_started = time.monotonic() - puntos.entrenado_s
        elapsed_after_step = time.monotonic() - budget_started
        if anticipado is not None and not anticipado.empezado and step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED and (
                constructor.acabado() or elapsed_after_step >= max(args.max_train_seconds - MEMCORPUS_ANTICIPADO_S, 0.5 * args.max_train_seconds)):
            print(f"qf2_memcorpus: anticipado en el paso {step} a {elapsed_after_step:.0f} s, constructor {constructor.parar()}", flush=True)
            anticipado.empezar(qf2_corpus_mezcla.para_guardar(tabla_mc, orig_model.memcorpus_cfg.minimo))
        # Fold the deadline decision into the existing health collective. A
        # local wall-clock check can otherwise make one rank begin another
        # optimizer step while its peers enter the checkpoint barrier.
        # The budget clock is only meaningful once it has been reset at the
        # excluded-startup boundary; before that, elapsed_after_step is measured
        # from loop start and a slow first-step compile could trip the predictive
        # stop and break before five steps complete — leaving the run unscorable.
        # Gate the time-based stop behind reaching the excluded-step count so the
        # run is always scorable; an explicit target_steps budget still stops early.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= target_steps
            or (
                reached_min_steps
                and (
                    elapsed_after_step >= args.max_train_seconds
                    or elapsed_after_step
                    + max(recent_step_times[-20:])
                    + CHECKPOINT_RESERVE_SECONDS
                    >= args.max_train_seconds
                )
            )
        )
        stop_after_step = stop_after_step or (guarda is not None and guarda.agotada())
        reloj_clock = []
        if LR_SCHEDULE == "reloj" or puntos is not None:
            reloj_clock = [round(elapsed_after_step * 1000), round(max(recent_step_times[-20:]) * 1000)]
        control = torch.tensor(
            [
                1 if (not saltar and (math.isnan(train_loss) or train_loss > 100)) else 0,
                1 if stop_after_step else 0,
                *reloj_clock,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")
        if reloj_log is not None:
            suelo_campo = f" {momentum!r}" if SUELO_MOM else ""
            reloj_log.write(f"{step} {lrm!r} {reloj_progress!r} {elapsed_after_step!r}{suelo_campo}\n")
        if LR_SCHEDULE == "reloj" and reached_min_steps:
            reloj_progress = max(reloj_progress, reloj_progress_from(control, args.max_train_seconds))
        if promedio is not None and not saltar:
            promedio.tras_paso(step + 1, reloj_progress if LR_SCHEDULE == "reloj" else (step + 1) / target_steps,
                               int(control[1].item()) != 0)

        if not saltar:
            total_train_tokens += args.total_batch_size
            smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = args.total_batch_size / max(dt, 1e-6)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = target_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | "
                f"remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}"
                + (f" | mem_desbordes {int(mem_obj.desbordes.item())}" if mem_obj is not None and MEM_TRANSPORTE == "a2a" else ""),
                flush=True,
            )

        if step == paso_inicial:
            # After first step all persistent objects are allocated; freeze to avoid GC pauses
            gc.collect()
            gc.freeze()
            gc.disable()
        if not saltar:
            step += 1
        if int(control[1].item()) != 0:
            break
        if puntos is not None and puntos.toca(control):
            bucle = {"step": step, "smooth_loss": smooth_loss, "total_train_tokens": total_train_tokens,
                     "recent_step_times": recent_step_times[-20:], "startup_allowance_est": startup_allowance_est,
                     "reloj_progress": reloj_progress, "entrenado_ms": int(control[2].item()),
                     "max_train_seconds": args.max_train_seconds}
            budget_started += puntos.guardar(step, bucle, orig_model, optimizer, train_loader, puntos_extras,
                                             restaurar_fp32 is not None, device)

    if reloj_log is not None:
        reloj_log.close()
    if guarda is not None:
        print0(f"qf2_guarda: saltos={guarda.saltos}")
    if constructor is not None:
        leidos = total_train_tokens / qf2_corpus_constructor.TOKENS_POR_SHARD
        print(f"qf2_memcorpus: parado {constructor.parar()} ocupacion~{tabla_mc.ocupacion(4096)} la red leyó ~{leidos:.0f} shards de tokens"
              f" (el constructor no baja del {MEMCORPUS_EXCLUIR}){' AVISO: SOLAPE POSIBLE' if 2 * leidos > MEMCORPUS_EXCLUIR else ''}", flush=True)
    extra_memhost = None
    if memhost is not None:
        n_memhost = memhost.fin()
        print0(memhost.resumen())
        extra_memhost = qf2_memhost_ckpt.cuantizar_propias(memhost.tabla, 2 * n_memhost + 4)
    if PRECARGA:
        print0(train_loader.resumen())
        train_loader.cerrar()
    if restaurar_fp32 is not None:
        restaurar_fp32()
    if promedio is not None:
        print0(f"qf2_promedio: pasos={promedio.instalar()}")
    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    extra_mem = qf2_memoria_ckpt.consolidar(mem_obj, COMPUTE_DTYPE if COMPUTE_DTYPE == torch.bfloat16 else torch.float32) if mem_obj else extra_memhost
    if tabla_mc is not None and rank == 0:
        extra_mem = {**(extra_mem or {}), **qf2_corpus_mezcla.para_guardar(tabla_mc, orig_model.memcorpus_cfg.minimo)}
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        }, extra_mem, anticipado)
        if anticipado is not None and anticipado.empezado:
            print(f"qf2_memcorpus: tablas escritas antes en {anticipado.segundos:.0f} s; guardado final {time.monotonic() - budget_started - budget_elapsed_before_save:.1f} s", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    if constructor is not None:
        constructor.limpiar()
    if memhost is not None and rank == 0:
        memhost.tabla.cerrar()

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, args.device_batch_size, args.seq_len, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            args.device_batch_size,
            args.seq_len,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"horizon_source:    {horizon_source}")
        print(f"lr_schedule:       {LR_SCHEDULE} last_lrm={lrm:.4f}")
        print(f"target_steps:      {target_steps}")
        print(f"target_tokens_M:   {target_tokens / 1e6:.1f}")
        print(f"target_ratio:      {args.target_param_data_ratio:.4f}")
        print(f"actual_target_ratio: {actual_target_ratio:.4f}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > args.max_train_seconds}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f"depth:             {args.depth}")
        print(f"aspect_ratio:      {args.aspect_ratio}")
        print(f"seq_len:           {args.seq_len}")
        print(f"device_batch_size: {args.device_batch_size}")
        print(f"total_batch_size:  {args.total_batch_size}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        print(f"scaling_params_M:  {num_scaling_params / 1e6:.1f}")
        print(f"actual_ratio:      {total_train_tokens / num_scaling_params:.4f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")

    cleanup_runtime()


if __name__ == "__main__":
    main()

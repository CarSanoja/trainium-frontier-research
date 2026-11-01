"""
Neuron Competition R1 baseline training script.

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
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    get_token_bytes,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    Tokenizer,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 1024
DEPTH = int(os.environ.get("ARCH_DEPTH", "12"))
ASPECT_RATIO = 64
HEAD_DIM = int(os.environ.get("ARCH_HEAD_DIM", "128"))

DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = int(os.environ.get("ARCH_TBS", "262144"))
NUM_STEPS = 100000
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.025
SCALAR_LR = 0.20
WEIGHT_DECAY = float(os.environ.get("ARCH_WD", "0.05"))
WARMUP_STEPS = int(os.environ.get("ARCH_WARMUP_STEPS", "20"))
GRAD_CLIP = 0.0
import torch.utils.checkpoint

# -----------------------------------------------------------------------------
# Toggles (experiments/arch). Apagados = baseline exacto. Ver docs/PLAN.md y D18/D19.
#   (Reloj: desde el PR #6 del kit, 3 sep, el runner excluye los 5 primeros pasos y cobra el
#    resto; train.py reinicia su reloj en el 5.º paso y usa 1800 s con reserva de 5 s. Nuestro
#    antiguo ARCH_CLOCK sobra y se eliminó. Para enviar: NUM_STEPS alto, manda el reloj.)
#   ARCH_DEPTH, ARCH_TBS, ARCH_SEED, ARCH_WARMDOWN=f (LR lineal a 0 en la última fracción
#                     f del reloj cobrado; usa budget_started, que el kit reinicia en el paso 5)
#   ARCH_LR_TAIL=1    añade peldaños 0.075/0.05/0.02/0 al final de la escalera: la tasa termina en 0
#                     y no en 0.1 (el ceil nunca baja del primer peldaño). 4 kernels más.
#   ARCH_LR_LEVELS=L  cuantiza lrm a L niveles (ceil): acota las recompilaciones por valor de LR
#   ARCH_LR_MULT, ARCH_MUON_MOM (final), ARCH_WD, ARCH_WARMUP_STEPS, ARCH_EMB_LR_MULT (solo embeddings): receta
#   ARCH_EMA=d       media móvil exponencial de los pesos (decay d); se vuelca al checkpoint
#   ARCH_DW_SAMPLE_MODE=split  (14 sep, tras dos sondas que no compilaron) el mismo gradiente muestreado
#                     SIN funcion autograd a medida: el MLP parte las secuencias del lote en una mitad superior
#                     y otra inferior (vistas), la superior pasa por c_fc/c_proj con los pesos normales y la
#                     inferior con los pesos .detach(). Forward exacto para todas, dX exacto para todas, y dW
#                     solo de la mitad superior: el modo 'bloque' con ventana fija expresado con matmul,
#                     detach y un cat por bloque, que el compilador si digiere.
#   ARCH_MASTER_FP32=1  copia maestra fp32 de los pesos (candidato RSI MW1, 15 sep): Muon actualiza una pila fp32 por
#                     grupo y AdamW una copia fp32 por parametro; el modelo recibe la copia bf16 tras cada paso. El momento
#                     de Muon sigue en bf16. Exige ADAMW_FOREACH=1; no admite STACKED_PARAMS ni MUON_SHARD. +~500 MB/rango.
#   ARCH_MLP_SAVED_RELU=2  MLP ReLU2 SIN activacion ancha guardada (candidato RSI MS3, 15 sep): el backward
#                     recalcula r = relu(x W1^T) desde x con un GEMM mas por bloque; aritmetica identica a =1
#   ARCH_MLP_SAVED_RELU=1  MLP ReLU2 con UNA activacion ancha guardada (candidato RSI MS1, 14 sep): se guarda
#                     r=relu(XW1) y el backward reconstruye r^2 y 2r. Misma aritmetica, una activacion de
#                     16384x3584 menos escrita y leida por bloque. NB1 midio que la ineficiencia del paso esta
#                     en materializar activaciones, no en los GEMM. Contrato CPU: research/mlp_saved_relu_reference.py.
#   ARCH_ATTN_TAU=1  concentracion aprendible por cabeza (propuesta de Carlos, 13 sep): q' = exp(theta)*q
#                     DESPUES de la norma QK, con theta inicializado a cero, asi que arranca reproduciendo
#                     P24 exactamente. Siete escalas por cada una de las seis atenciones: 42 escalares.
#                     Controla QUE posiciones se seleccionan; la puerta por cabeza ya refutada modulaba la
#                     salida DESPUES de seleccionar, y las frecuencias RoPE aprendibles tocan la geometria
#                     posicional. Son ejes distintos. No es gratis: es una multiplicacion elemental de
#                     ancho completo sobre q en cada atencion, y eso se mide con tools/pasadas.py antes de
#                     gastar chip. Grupo AdamW propio sin decaimiento.
#   ARCH_RANK_SOUP=1|2 SOLO se sincronizan los GRADIENTES (sync_gradients); los PESOS nunca. Con
#                     ARCH_BF16=1 y NEURON_RT_STOCHASTIC_ROUNDING_EN=1 cada rango redondea su propia
#                     actualización con su propia moneda, así que los 8 rangos terminan con 8 pesos
#                     distintos alrededor de la misma trayectoria fp32, y sólo guardamos el del rango 0.
#                     1 = sólo diagnóstico: imprime la divergencia entre rangos (si sale 0, la premisa
#                     es falsa y la idea muere ahí). 2 = además promedia los 8 en fp32 antes de guardar,
#                     lo que sigue siendo UN modelo consolidado y reduce el ruido de redondeo por raíz de 8.
#   ARCH_WAVG=f      promedio uniforme de los pesos sobre la última fracción f del reloj, con
#                     ARCH_WAVG_K=k instantáneas (6 por defecto) acumuladas en fp32 en el HOST.
#                     La sombra fp32 en el dispositivo de ARCH_EMA agotó la HBM el 5 sep (+415 MB
#                     por rango con PACK 16); ésta no vive en el dispositivo y sólo cruza k veces.
#                     Todos los rangos hacen la copia para que el programa sea idéntico en SPMD;
#                     sólo el rango 0 acumula. El promedio va a final.pt y los pesos crudos a
#                     final_raw.pt, así que la calidad se mide DENTRO DE UNA SOLA CORRIDA, sin
#                     ruido de semilla: es la misma trayectoria evaluada dos veces.
#   ARCH_VE="0,1,10,11"  value embeddings compartidos en esas capas; ARCH_SKIP_SRC/DST=2/9 salto
#                     aprendido; ARCH_SMEAR=1 mezcla causal del token anterior. Van al GPTConfig.
#   ARCH_UNET=1       saltos U-net: la salida de la capa i (i < L/2) se suma a la entrada de la
#                     capa L-1-i con un peso aprendido (init 0). ARCH_TIMING=1 imprime fwd+bwd /
#   ARCH_MASK_SPECIAL=1 logits de los 4 tokens reservados a -1e4 y sus targets ignorados (no
#                     puntúan: 0 bytes). ARCH_WIDTH=n fija n_embd sin tocar DEPTH. ARCH_EMA guarda
#                     la sombra fp32 directamente en final.pt y los pesos crudos en final_raw.pt.
#   ARCH_FLASH=1      (NO compila: los kernels legados neuronxcc.nki.kernels.attention usan dtypes numpy y una
#                     dataclass que el front-end nuevo de NKI rechaza; se conserva como referencia).
#   ARCH_MUDD_TAIL=1   mezclas dinamicas tipo MUDD (PR259): enrutador 1152→64→6 tras la capa 5 mezcla {x0,h3,h5}
#                     en el residual y en la V de la capa 6; enrutador 1152→64→3 tras la capa 7 mezcla en el residual
#                     final. Segunda matriz a cero (salida identica al inicio). Campo de GPTConfig; asume 8 capas.
#   ARCH_LOGIT_ASYM=1  soft-cap asimetrico 11.5*tanh((z+5)/15) (campo de GPTConfig).
#   ARCH_PARTIAL_ROPE=1 ARCH_KEY_SHIFT_LAYERS=0,6  RoPE parcial (frecuencias 32:64 nulas) y, en esas capas, los
#                     canales estacionarios de K (32:64, 96:128) toman la posicion anterior salvo en pos 0 / tras BOS
#                     (campos de GPTConfig; head_dim 128).
#   ARCH_HOURGLASS=2,3,4,5 reloj de arena causal: esas capas (contiguas) procesan la secuencia a media resolucion
#                     s[j]=(h[2j-1]+h[2j])/2 (h[-1]=0), x0 y VE agrupados igual, RoPE 0..t/2-1; a la salida
#                     x = h + repetir(z - s, 2). Campo de GPTConfig (cambia el forward: la eval lo necesita).
#   ARCH_ATTN_WIDTH=512 anchura de la atencion (Q/K/V/VE) independiente del residual: cabezas = aw/head_dim
#                     (512 → 4×128); c_proj aw→n_embd. Ataca el coste heads×T² del kernel de atencion.
#   ARCH_HEAD_DIM=64  dimensión de cabeza (kit: 128); con anchura 1152 pasa de 9 a 18 cabezas.
#   ARCH_PROFILE_DIR=/ruta ARCH_PROFILE_WAIT=30  perfil NRT (dispositivo+runtime) del paso WAIT+2 en el rank 0
#                     vía torch.profiler + torch_neuronx.profiling; ver con neuron-profile view -d.
#   ARCH_ROPE_MM=1     RoPE como x*cos + (x@P)*sin con P permutacion con signo (rotate_half en el motor tensorial) y
#                     cos/sin de ancho completo: evita los cortes y el cat que el compilador baja mal (P64: RoPE = 22 % del paso).
#   ARCH_NORM_MANUAL=1..6  RMSNorm escrita a mano (1: estadisticos en fp32; 2: todo en bf16) con eps = finfo(dtype).eps
#                     como F.rms_norm: mide si la bajada de F.rms_norm es el coste (P65: sin normas −290 ms/paso).
#                     3: media de x² por matmul con ones(d,d)/d (la reduccion cruza particiones: va al motor tensorial)
#                     en todas las normas; 4: matmul solo en la norma QK (d=128) y modo 2 en las residuales;
#                     5: como 4 y las residuales con ones(d,128)/d y corte [..., :1] (reduccion en el motor tensorial sin d×d);
#                     6: norma QK sobre la vista plana [b,t,H*D] con matriz diagonal por bloques (buffer), antes de la vista
#                     por cabezas y de RoPE (equivalente: RoPE es ortogonal por cabeza); residuales como el modo 2.
#   ARCH_QKNORM_KERNEL=1  RoPE + norma QK en un kernel NKI propio fusionado (tokens en las particiones; fwd y bwd
#                     via torch_neuronx.nki_op con autograd); solo en bf16 con t=1024 en el dispositivo, si no cae al camino torch.
#   ARCH_ATTN_FUSED_BWD=1  atención fusionada NKI con backward NKI propio (ATTF H1, research/edge30/H1): op arch::attfb_fwd (kernel
#                     de tools/attn_fused_nki.py copiado abajo) sobre las vistas [N, 896] de c_q(x), c_k(x) y c_v(x)+ve, y su
#                     backward arch::attfb_bwd (kernel de research/edge30/H1/attn_fused_bwd_nki.py copiado abajo): dQ, dK, dV con P
#                     recomputada por bloques desde lse, norma QK y RoPE hacia atrás dentro, entradas y salidas [N, 896] con los tokens
#                     en filas. Rejilla wrap_nki(k)[ARCH_ATTN_FUSED_LNC] (2 por defecto). Solo en entrenamiento (la evaluación va por
#                     el camino torch, como el MLP fp8), con t = 1024, lote múltiplo de la
#                     rejilla (si no, p. ej. una evaluación con lote impar, camino torch), bf16, head_dim 128 y
#                     dispositivo (ARCH_ATTN_FUSED_SIM=1: también en CPU con el simulador; solo pruebas); si no, camino torch.
#                     Incompatible (assert): NO_ROPE, NO_NORM, NO_QKNORM, NO_ATTN, FLASH, MANUAL_ATTN, QKV_FUSED, QKNORM_KERNEL,
#                     NORM_MANUAL=6; por capa cae al camino torch con attn_tau, xsa, head_gate, lga, doc_mask, hier_window, key_shift
#                     o ventana.
#   ARCH_ATTN_FUSED_BWD_GATE=1  puerta de exactitud en el rango 0 antes del reloj del bucle: forward y backward en el dispositivo
#                     (torch.compile backend neuron) sobre entradas deterministas a N = 16384 frente al gradiente fp64 del anfitrión
#                     (reglas F, E y S de research/edge30/H1/h1_check.py); imprime ATTFB_GATE y sale con código 3 si falla. En el chip
#                     el backward llama al kernel envuelto directamente (el patrón de mlp_fp8_backward_kernel); en el simulador, por la
#                     op arch::attfb_bwd (con su register_fake, para trazar con aot_eager).
#   ARCH_ATTN_DW8=1  (ADW8, 30 sep; exige ARCH_ATTN_FUSED_BWD=1) en el camino de entrenamiento _attfb_forward, el dW de c_q, c_k, c_v y
#                     c_proj sale del kernel NKI arch::adw8_dw en fp8 double_row (generaliza _dw8_fase de ARCH_MLP_DW8) por _ADW8Lin:
#                     forward F.linear y dX = g · W siguen en bf16 del compilador. g [N, 896] (gradiente de la salida) y a [N, 896] (entrada)
#                     llegan en bf16 de HBM por trozos de 256 tokens (token 256·s + 2·p + r en la partición p, fila r de double_row) y se
#                     convierten en SBUF: g a e5m2 por S = 2^ARCH_ATTN_DW8_K y a a e4m3 saturada a ±240. Los tokens se contraen en la
#                     partición sin trasponer operandos (estacionario g por teselas de 128 filas, móvil a), cada tesela acumula en su banco
#                     de PSUM fp32 y el epílogo aplica 1/S: salida bf16 contigua [salida, entrada], la disposición del peso. Rejilla
#                     wrap_nki(k)[ARCH_ATTN_DW8_LNC] (2): cada programa hace 448 de las 896 columnas de entrada. ARCH_ATTN_DW8_K=24: |g|
#                     máximo medido 3.1e-6 (research/adw830/adw8_captura.py, MLPW2_qctl, 24 proyecciones) -> 53 frente a 57344 (margen 2^10,
#                     como ARCH_MLP_FP8_DYSCALE), aplanado a cero <= 0.3 %. ARCH_ATTN_DW8_ACT=1 (defecto) convierte g en el motor escalar
#                     (tensor_scalar con engine=scalar; el planificador del compilador lo estima 25 % más rápido que todo en el vectorial,
#                     research/adw830/bir) y =0 deja las dos conversiones en el motor por defecto (las formas de DW8). Evaluación y
#                     conmutador a 0 sin cambios.
#   ARCH_MLP_EPIPASS=1|2  (EPIPASS, 30 sep; exige ARCH_MLP_DW8=1 y actua con ARCH_MLP_FP8_DYSCALE > 0) epilogo de cada tesela de
#                     dpre^T en mlp_fp8_bwd_act: kgs = sc[:, 2]·sc[:, 4] = 2/(_SCALE_H·s2) una vez por llamada (potencias de dos de
#                     bwd_scales: producto exacto) y dps = e5m2(ps · kgs · rT) sin el paso por bf16 de gb (con ARCH_MLP_DW8=1 gb solo
#                     alimentaba a dps). 1 (brazo A): una sola scalar_tensor_tensor de PSUM a e5m2 en vez de tensor_tensor + dos
#                     tensor_scalar; 2 (brazo A'): g = ps · rT y un tensor_scalar por kgs a e5m2 (quita 1 de 3 pases sin forma de
#                     instruccion nueva). Los dos brazos dan los mismos bytes; frente a 0 solo desaparece el doble redondeo por bf16
#                     (research/frontera6/fix_epipass.py). Solo backward: evaluacion y conmutador a 0 sin cambios.
#   ARCH_KA_MLP=1  (KA30, 30 sep; research/ka30/parche_ka_mlp.py, candidatos LT512 y PK1 de research/kernaudit) mismas cuentas con
#                     menos instrucciones en los kernels MLP fp8. LT512: en mlp_fp8_fwd_r (x^T y su cola TLD de 64 columnas) y en
#                     mlp_fp8_bwd_act (dy^T a dys y dytl), una load_transpose2d de 512 tokens y una conversion por (c, j) en vez de
#                     cuatro de 128. PK1 (con ARCH_MLP_DW8=1): las cuatro nc_transpose de dps y e4m3(rT^2) escriben una sola pk
#                     [128, 4, 128] bf16 de PSUM (1 KiB por particion, un banco) y una tensor_copy desentrelaza dpk y hpk. Bit a bit
#                     igual que 0 con ARCH_MLP_EPIPASS=0, 1 y 2 (simulador NKI, research/ka30/ka_mlp_k/ka_mlp_sim.py). Toca el forward
#                     sin estado: la evaluacion da lo mismo con 0 o 1.
#   ARCH_MLP_FWDEPI=1  (FWDEPI, 30 sep; exige ARCH_MLP_EPIPASS=1 y ARCH_MLP_DW8=1; research/fwdepi/parche_fwdepi.py) epilogo de
#                     cada tesela de pre^T en mlp_fp8_fwd_r en dos pasadas: rT' = min(max(ps, 0), 61.75*s1) directo de PSUM a bf16
#                     (sin la ReLU escalada de ACT; 61.75*s1 = _H_CLAMP/(_SCALE_H/s1), exacto) y hs = e4m3(rT'*q*rT') con
#                     q = (_SCALE_H/s1)^2 en una tesela [128, 1] por llamada. rT se guarda sin escala (rT' = rT*s1/_SCALE_H) y el
#                     backward pliega _SCALE_H/s1 = _SCALE_H*sc[:, 3]*sc[:, 4] en kgs de EPIPASS y q en h8 (scalar_tensor_tensor);
#                     las referencias torch desescalan rT' por 1/s1. Escalas potencias de dos: mismos bytes que 0 salvo el signo de
#                     los ceros. La evaluacion no pasa por el kernel fp8; con 0 el texto que traza cada kernel es el del portador.
#                     2: la misma cuenta con la tensor_scalar de PSUM fijada al motor escalar (engine=nisa.engine.scalar): ACT hace la
#                     pasada de PSUM a bf16 como en 0 y la DVE solo la scalar_tensor_tensor (1 en vez de 2 pasadas de DVE por tesela).
#   ARCH_MOE_LAYERS=1,3,5,7 ARCH_MOE_E=8 ARCH_MOE_K=2 ARCH_MOE_HIDDEN=2304 ARCH_MOE_CAP=1.25 ARCH_MOE_AUX=0.01
#                     MLP de mezcla de expertos en esas capas: E expertos ReLU² de anchura HIDDEN, top-K por token con
#                     gating softmax sobre los K elegidos (Mixtral), capacidad fija por experto (formas estaticas, sin
#                     recompilar; los tokens que exceden la capacidad pasan solo por el residual), expertos en bmm por
#                     lotes (pesos apilados de Linear normales: Muon los agrupa por forma), perdida auxiliar de balanceo
#                     (Switch: E·sum_e f_e·P_e) sumada en un envoltorio del forward. Con E=8, K=2, HIDDEN=2304: los mismos
#                     FLOPs por token que el MLP denso ×4 y 4× sus parametros (eficiencia de muestra en regimen multi-epoca).
#   ARCH_MOE_DISPATCH=mm (defecto) despacho y combinacion por matmul con matriz one-hot POR FILA (GShard: sin indices
#                     ni scatter en fwd ni bwd; +2·E·C·T·d FLOPs por fila); =gather usa gather de filas + combinacion por gather.
#                     Gating: prob softmax completa del experto elegido (Switch; con K=1 el router recibe gradiente),
#                     renormalizada entre los K elegidos si K>1.
#   ARCH_MOE_LOOP=1 (defecto) expertos como matmuls 2D por experto en vez de bmm; ARCH_MOE_SCATTER=1 combina con index_add
#                     (defecto: por gather de las K filas de cada token, sin scatter); ARCH_MOE_CUMSUM_MM=1 cumsum por bloques
#                     (M2: compila 1196 s y no acelera; defecto torch.cumsum).
#   ARCH_DOC_MASK=1    atencion causal restringida al documento (mascara bloque-diagonal por BOS, [b,1,t,t] bool via kmask);
#                     no cruza documentos empaquetados en la misma fila. Coste: SDPA con mascara densa en vez de is_causal.
#   ARCH_XSA=1        Exclusive Self Attention con intensidad aprendida por cabeza (campo xsa de GPTConfig; 9 alphas fp32 por
#                     atencion, init 0 = identidad; grupo Adam escalar, fuera de Muon): y' = y - tanh(a_h)·(y·v)/(v·v)·v con el v
#                     real de la SDPA (incluye value embeddings); las dos reducciones sobre d por matmul con ones(d,d)/d,
#                     todo en fp32 con eps 1e-6 sumado (AX0 en bf16 con clamp 1e-8 dio NaN en el paso 1 del chip).
#   ARCH_CTX_RANK=128 ARCH_CTX_SOURCE=attn|resid  canal de contexto separado (propuesta de Carlos, 8 sep): en cada bloque
#                     con atencion, c = norm(P·a) (attn: la actualizacion de atencion; resid: el residual ya mezclado s+a) con
#                     P [1152→r]; el MLP del bloque siguiente sin atencion recibe concat(norm(x), c) (c_fc [4608, 1152+r], columnas
#                     del canal a cero en init = identidad); campos ctx_rank/ctx_source en GPTConfig; Muon compensa el factor de LR
#                     de la W1 ampliada.
#   ARCH_PREFIX_BETA=b  objetivo auxiliar del prefijo lexico (modded-nanogpt PR337): para cada objetivo y, el token del
#                     prefijo propio de bytes mas largo (tabla fija del tokenizador 8192; 86 % de los tokens de texto) se suma
#                     al CE con peso b reutilizando el mismo log_softmax, mezclado como el MTP ((CE + w·CE2 + b·CEp)/(1+w+b));
#                     solo en entrenamiento (aux_alpha no nulo); la tabla es un buffer no persistente creado en main.
#   ARCH_NO_BWD=1    ablacion SOLO de tiempo (rompe el modelo): forward en modo entrenamiento sin backward ni
#                     optimizador. dt con y sin ella, en el paso compilado real, es la medida D1 que las sondas eager
#                     no pudieron dar el 14 sep (fallos de memoria alrededor de la medida).
#   ARCH_NO_ROPE=1 / ARCH_NO_NORM=1 / ARCH_NO_QKNORM=1  ablaciones SOLO de tiempo (rompen el modelo): miden el coste
#                     de RoPE, de todas las RMSNorm y de la norma QK.
#   ARCH_PREFETCH=1    hilo que precarga el siguiente micro-lote (cola de 2): oculta la carga de datos (~20 ms/paso).
#   ARCH_TBS_RAMP=0.3  rampa de lote: la primera fraccion del reloj entrena con la mitad de la acumulacion
#                     (mitad de lote, mismo LR), luego el lote completo; los tokens se contabilizan por micro-lote.
#   ARCH_TIE_UNTIL=0.667 atado temporal: cabeza con std 0.005 y embedding = cabeza; hasta esa fraccion del reloj el
#                     gradiente del embedding se suma al de la cabeza y el embedding se copia de la cabeza tras cada paso;
#                     despues se separan (el embedding hereda el estado Adam de la cabeza y sus hiperparametros).
#   ARCH_OPT_COMPILE=1 el núcleo de Muon por grupo de formas (momento, Polar Express, NorMuon, decaimiento) se compila
#                     como función pura con LR y momento como tensores; los copiados a los parámetros siguen en eager.
#   ARCH_FWD_PROBE=1  en el paso 8 cronometra (sincronizado, 3 repeticiones) forward sin perdida, forward con
#                     perdida y forward+backward sobre el micro-lote real, e imprime "fwdprobe:".
#   ARCH_BIGRAM_N=32749 ARCH_BIGRAM_DIM=256  tabla de bigramas con hash (prev*8192+tok mod N), proyección
#                     a n_embd inicializada a cero (OFF exacto en el paso 0); tabla en Adam, proyección en Muon.
#   ARCH_MTP=1|2      objetivo auxiliar log p_t(y_{t+1}) sobre los mismos logits (sin cabeza ni forward extra);
#                     1: alfa=0.10 la primera mitad del reloj, 0 despues; 2: alfa 0.50 lineal a 0 a los 2/3.
#                     ARCH_MTP_ALPHA cambia el peso; ARCH_MTP_PROBE=1 varia alfa cada paso (prueba de recompilacion).
#   ARCH_MTP_HEAD=256  cabeza auxiliar independiente (perdida por bloques con recomputacion, ARCH_MTP_HEAD_CHUNKS=4) de rango r (n_embd→r→vocab) para el token t+2 con el mismo
#                     horario; la cabeza principal no comparte la distribucion. Requiere ARCH_MTP=1|2.
#                     ARCH_MTP_END=0.667 fraccion del reloj donde alfa llega a 0 (MTP=2); ARCH_MTP_SHAPE=lin|const;
#                     ARCH_MTP_ALPHA3=0.x anade el token t+3 (tercera columna del mismo gather) con el mismo horario.
#   ARCH_MLP_GATED="1,3,5,7" esas capas usan SwiGLU con inner=2·mult·n_embd/3 (mismos pesos y FLOPs
#                     que ReLU² ×mult); c_fc fusiona puerta y subida. Vacío = ReLU² en todas.
#   ARCH_MLP_MULT=6   oculta del MLP = 6×n_embd (kit: 4). Calidad por token sin más datos.
#   ARCH_ATTN_LAYERS="0,2,4" solo esas capas llevan atención; el resto es solo MLP (la atención es ~50 % del paso).
#   ARCH_MANUAL_ATTN=1 atención causal manual: matmul + máscara aditiva precalculada + softmax fp32 (=2: softmax
#                     en bf16). Evita la bajada de F.scaled_dot_product_attention del compilador.
#   ARCH_MANUAL_ATTN=4 atención manual estilo flash: forward por bloques que guarda solo q,k,v,o,lse; backward
#                     recomputa las probabilidades por bloque (autograd.Function, sin checkpoint). Memoria como SDPA.
#   ARCH_MANUAL_ATTN_CKPT=1 recomputa la atención manual en el backward (no guarda los mapas T×T): menos memoria.
#   ARCH_MANUAL_ATTN=3 atención causal manual por bloques de consultas (ARCH_ATTN_CHUNKS=4): cada bloque
#                     solo atiende a las claves hasta su fin (salta los bloques enmascarados), softmax fp32.
#   ARCH_ATTN_WINDOW=256 ARCH_ATTN_WINDOW_LAYERS=2,4  (solo con ARCH_FLASH=2) ventana deslizante causal en esas
#                     capas via nkilib attention_cte/attention_bwd(sliding_window); las demas capas, causal completo.
#                     Con ventana el kernel exige scale=1.0: q se preescala por head_dim^-0.5 fuera del kernel.
#   ARCH_CTE_MM_BF16=1 / ARCH_CTE_SOFTMAX_BF16=1 / ARCH_CTE_BWD_MIXED=0 / ARCH_CTE_TPQ=1  (con ARCH_FLASH=2) opciones
#                     del kernel nkilib: salida del QK en bf16, softmax en bf16, backward sin fp32, layout de Q sin
#                     transponer (tp_q=True). Barrido de velocidad del 8 sep.
#   ARCH_FLASH=2      igual con los kernels de nkilib (attention_cte con softmax cacheado + attention_bwd).
#   ARCH_NO_ATTN=1    DIAGNÓSTICO: la atención devuelve v (sin softmax ni QK^T); solo para medir cuánto
#                     tiempo del paso es atención. Nunca en un envío.
#   ARCH_MUON_SHARD=1 cada rango calcula Newton-Schulz solo para su parte de cada pila de matrices
#                     y difunde el resultado (broadcast): el optimizador deja de ser redundante ×8.
#   ARCH_ADAMW_FOREACH=1 AdamW con ops multi-tensor (torch._foreach_*): ~8 lanzamientos por grupo en vez de
#                     ~8 por tensor (P43: los 113 ms del optimizador son lanzamientos, no matemática).
#   ARCH_MLP_FP8_EAGER=1 la op corre FUERA del grafo compilado (torch._dynamo.disable + despacho eager de los kernels): el
#                     grafo se parte alrededor de cada MLP; evita el HOP, que en el bucle real de entrenamiento falla (P99b-n).
#   ARCH_VOCAB_PAD=K   la cabeza (y las tablas) tienen K columnas más que el vocabulario, enmascaradas con un sesgo fijo de -1e4
#                     (campo vocab_pad de GPTConfig, persiste en el checkpoint; la evaluación devuelve logits recortados a
#                     vocab_size): el buffer del scatter del backward del gather es físicamente más ancho; F.pad se pliega
#                     (MD19) y no sirve. Remedio del fallo NRT 1006 con la op MLP fp8.
#   ARCH_PAIR_PHASE=f  entrenamiento a dos resoluciones (apuesta del 10 sep): mientras el reloj cobrado va por debajo de f,
#                     cada pareja de tokens ocupa una posición interna (1024 tokens → 512 posiciones): z[j] = (E(x[2j+1]) +
#                     R(E(x[2j]))) / sqrt(2) con R giro fijo de 90° por mitades (también en VE), RoPE en las coordenadas
#                     originales 1,3,...,T-1, y la pérdida MTP de P19 en las fronteras impares: principal y[2j+1], auxiliar
#                     y[2j+2] (un solo gather de dos columnas). Un solo cambio, sincronizado por rangos en una frontera de
#                     paso (broadcast desde el rango 0) y solo con el reloj ya reiniciado (paso ≥5); sin reiniciar pesos,
#                     optimizador ni schedules. Después, forward y pérdida normales. La evaluación siempre usa el forward normal.
#   ARCH_GROW=f        crecimiento de capacidad (contrato del 10 sep): los bloques ARCH_GROW_LAYERS (por defecto 2,3,6,7) no se
#                     ejecutan mientras el reloj cobrado va por debajo de f; sus parámetros viven en grupos propios del optimizador,
#                     congelados (sin gradiente, momento ni weight decay) y con estado reservado; x0_lambda=0 desde el inicio y
#                     proyecciones de salida a cero (init del kit), así que al activarse son identidades exactas. Un solo cambio,
#                     difundido desde el rango 0 en una frontera de paso con el reloj ya reiniciado. Grafo completo precalentado en el paso 0.
#   ARCH_LEXSETS=b0    supervisión de conjuntos léxicos: 128 grupos de 64 ids ordenados por (bytes, id) con el tokenizador del kit
#                     (ARCH_LEXSETS_RANDOM=semilla: permutación aleatoria como control de mecanismo); por posición válida,
#                     pérdida = (L_base_MTP + β·(−logsumexp de las log-probabilidades de los 64 miembros del grupo del objetivo))/(1+β),
#                     β = b0·max(0, 1 − u/ARCH_LEXSETS_END) (u = fracción del reloj cobrado; fin por defecto 2/3). Un solo gather de
#                     66 columnas (principal, MTP, miembros preparados fuera del grafo como targets_prefix). β=0 reproduce la base.
#   ARCH_LEXSETS_MM=1  la masa del grupo del objetivo se calcula como exp(logp) @ M (M [V,128] de pertenencia, un GEMM de
#                     N×8192×128) y se selecciona con un one-hot de 128 columnas, en vez del gather de 66 columnas (GR3: ese
#                     gather costaba ~+0.8 s por paso en Neuron). Misma pérdida (igualdad verificada en CPU). Los kwargs pasan
#                     el id de grupo [B,T] en lugar de los 64 miembros.
#   ARCH_HIER_WIDTH=w  jerarquía local/global (apuesta de Carlos, research/2026-09-10-ventaja-estructural.md): un encoder
#                     causal de anchura w sobre las 1024 posiciones, empaquetado de cada pareja de estados finos a un estado
#                     del tronco (512 posiciones, RoPE en las fronteras impares, VE promediado por pareja), y un lector causal
#                     de anchura w que combina cada estado fino con el último estado global permitido (cero, H0, H0, H1, H1, ...,
#                     duplicación estática sin gather); salida = norm(emb + estado global + proyección del lector) → cabeza
#                     completa en cada posición. Siempre activa (entrenamiento y evaluación). Cabezas locales = w/128.
#   ARCH_TWOSTREAM=1  memoria residual de dos vías (apuesta de Carlos, research/2026-09-11-investigacion-memoria-residual.md):
#                     cada posición lleva dos estados u, v; por bloque un router (proyección 5×2d de la concatenación
#                     normalizada, sesgos y tres escalas) da lectura a0,a1 (sigmoide), escritura b0,b1 (2·sigmoide) y retención p
#                     (sigmoide, sesgo log 99); z = a0·u + a1·v entra al bloque; delta = incremento del bloque sin restar tras
#                     el redondeo; u' = p·u + (1−p)·v + b0·delta, v' = (1−p)·u + p·v + b1·delta. Expansión inicial u = e·(c+d),
#                     v = e·(c−d) con c=1 y d ~ U(−0.01, 0.01) aprendibles; salida = norm((u+v)/2). El salto 1→10 se aplica
#                     a cada vía; MTP y el resto de la receta sin cambios. Router en AdamW fp32 (lr 0.25·UNEMBEDDING_LR·(d/768)^-0.5,
#                     betas 0.9/0.99, WD 0). ARCH_TWOSTREAM_DIFF=0 desactiva la diferenciación inicial (control de mecanismo).
#   ARCH_MUON_VS=1    Muon precondicionado por la dispersión del gradiente (apuesta de Carlos, research/2026-09-11-nueva-via-
#                     optimizacion.md, adaptación de Muon-VS): por grupo, estado Γ (misma forma que el momento, fp32) y masa c;
#                     Γ_t = β·Γ + β(1−β)(M_{t−1} − G)², c_t = β·c + (1−β); el Nesterov se divide por sqrt(max(Γ/c, 0)) + 1e−8
#                     ANTES de Polar Express; el resto (normalización por fila/columna, LR por forma, WD cauteloso) sin cambio.
#                     Solo en el camino eager de Muon (ARCH_OPT_COMPILE=0). +367.5 MiB fp32 por rango a 12×896.
#   ARCH_HIER_ATTN=0   (jerarquía) el encoder y el lector locales son solo MLP (sin atención causal de 1024×1024):
#                     el contexto fino lo aporta el tronco; ahorra la atención densa local (3.2 G de 10.5 G MAC locales).
#   ARCH_HIER_ADAMW=1  (jerarquía) las matrices de encoder, lector y puentes van a un grupo AdamW multi-tensor en vez de
#                     a grupos de Muon por forma (HR2: +0.059 s de optimizador por la docena de formas nuevas).
#   ARCH_HIER_WINDOW=w (jerarquía, con ARCH_HIER_ATTN=1) la atención de encoder y lector locales se limita a una banda
#                     causal de w tokens (máscara aditiva cacheada, SDPA): contexto fino a coste ~w/T de la densa.
#   ARCH_REUSE_RANK=r  reutilización de activaciones entre MLP (apuesta de Carlos, research/2026-09-11-reutilizacion-y-
#                     rutas-de-profundidad.md, adaptación de CR-Net): el MLP de la primera capa es denso (ancla) y los
#                     siguientes transportan la preactivación ancha u y la salida o de la capa anterior dentro de la MISMA
#                     llamada: u_l = β_up·u_{l−1} + B_up(A_up z_l); o_l = β_down·o_{l−1} + B_down(A_down ReLU²(u_l)), con
#                     A/B de rango r, β iniciales 0.5 entrenables, B_down a cero y B_up escalado por sqrt(1−β²). Sin detach,
#                     sin estado entre llamadas. Los factores van a un grupo AdamW propio (lr 0.001, betas 0.9/0.95, WD 0.05).
#   ARCH_REUSE_BETA0=1 control de mecanismo: β fijos a cero (mismos factores e init; mide la factorización sin historial).
#   ARCH_PAIR_MODE=ordered|mean|last  control de mecanismo: promedio reescalado (a+b)/sqrt(2) o solo el último token b.
#   ARCH_PAIR_WARM=1   (por defecto) en el paso 0 compila el grafo normal (fwd+bwd con gradientes descartados) dentro de la
#                     ventana de arranque excluida, para que el cambio de fase no pague una compilación fría en los 1800 s.
#                     Antes rellena las cachés globales (_MEANM, máscaras) que el forward consulta con `if key not in`: la primera
#                     traza de dynamo guarda 'key not in' y la siguiente llamada recompila un grafo distinto (NEFF nuevo, ~190 s
#                     en el cambio de fase: PR1/PR2, 10 sep). Con las cachés llenas, calentamiento y régimen trazan el mismo grafo.
#   ARCH_STACKED_PARAMS=1 los pesos de cada grupo de Muon (misma forma) viven en un solo tensor apilado y cada peso es una vista:
#                     el paso de Muon no copia N gradientes y N pesos a los buffers apilados ni escribe N copias de vuelta
#                     (~100 lanzamientos por paso, ~0.03 s = 3 % del paso ≈ 0.003 bpb al cambio de T1860). Matemática idéntica.
#   ARCH_LOSS_PADROWS=R añade R filas (copias de las últimas, con objetivo -1 = ignoradas) a las log-probabilidades, a los
#                     objetivos y a las entradas de la cabeza MTP: el último objetivo real deja de estar al final del buffer
#                     del scatter (fallo NRT 1006 con la op MLP fp8 cuando ese objetivo es ≥8185). NO es gratis en el chip:
#                     ~7 % del paso (copia de la matriz de log-probabilidades) y G0 = +0.0045 bpb frente a S22 (11 sep). Nunca en un paquete.
#   ARCH_LP_PAD=K      rellena las log-probabilidades con K columnas a -1e4 antes del gather (pérdida principal y trozos MTP):
#                     misma pérdida; el buffer del scatter del backward queda K columnas más ancho y el vector de 8 elementos
#                     no se sale con objetivos ≥8185 en la última fila (fallo NRT 1006 con la op MLP fp8; MD17 lo acotó).
#   ARCH_CE_ONEHOT=1   -log p del objetivo por producto con one-hot en vez de gather (pérdida principal y trozos MTP): evita el
#                     scatter indirecto del backward del gather, sospechoso del fallo NRT 1006 con la op MLP fp8 (MD13 descartó
#                     las tablas de embedding). Coste ~1 % del paso.
#   ARCH_EMB_CHUNKS=N  gradiente de las tablas de embedding (tokens y VE) acumulado por N trozos de tokens con index_add_
#                     en fp32 (autograd propio): esquiva la acumulación indirecta del compilador que desborda con ids de
#                     >512 apariciones cuando la op MLP fp8 está en el grafo (P99b-p). N=0: nn.Embedding normal.
#   ARCH_EMB_SCATTER=1 (EMBK, 28 sep) wte y value_embed pasan por _EmbScatter: forward = F.embedding (bit a bit el de
#                     nn.Embedding); backward = dW fp32 [V, D] con el kernel NKI arch::embk_dw en el dispositivo (index_add_ fp32 en
#                     CPU) y el gradiente vuelve con el dtype del parámetro. El kernel pone dW a cero, recorre los ids en tiles de
#                     128 filas (nl.sequential_range) y en cada tile suma los duplicados en el PE (matriz de igualdad 128x128,
#                     simétrica, por dy: PSUM fp32), deja solo la primera aparición de cada id con su índice real (cada repetición
#                     va a la fila basura V + m, con m su fila en el tile: 128 filas extra de dW que se descartan con [:V]; el
#                     salto fuera de rango, oob_mode=skip, abortó el DMA en el chip el 28 sep) y acumula con nisa.dma_compute
#                     (DMA read-modify-write indirecto, fp32): cada instrucción ve 128 índices únicos y en rango, que es lo que
#                     exige unique_indices=True en nki 0.5.0; el
#                     dma_copy(dst_rmw_op=...) del kernel experimental del SDK no existe en nki 0.5.0. Rejilla wrap_nki(k)[2]:
#                     cada programa escribe solo sus D/2 columnas. ARCH_EMB_SCATTER_LNC=1 lanza sin rejilla. Excluye ARCH_EMB_CHUNKS.
#   ARCH_EMB_SCATTER_GATE=1 puerta de exactitud en el rango 0 antes del reloj del bucle: dW del kernel sobre los ids del primer
#                     micro-lote (y los mismos con las 5000 primeras posiciones al id más frecuente) con dy = 1e-3·randn (semilla 0,
#                     bf16) frente a index_add_ fp32 en CPU; imprime EMBK_GATE y sale con 3 si falla
#                     (max|dW − ref| / max|ref| ≤ 1e-3 y ‖dW − ref‖F / ‖ref‖F ≤ 1e-4 en los dos casos). Con =2 añade a la
#                     línea la comparación con index_add_ compilado en el dispositivo, solo informativa (no decide).
#   ARCH_DIAG_BATCH=1  DIAGNÓSTICO: imprime min/max/-1 de cada lote por micro-paso y guarda el último lote en
#                     <out-dir>/lastbatch_rank<r>.pt (para reproducir el fallo NRT 1006 de la op fp8 con el lote exacto).
#   ARCH_STATIC_INPUTS=1 x e y del paso se copian en dos buffers de dispositivo fijos (misma dirección en cada paso): con la
#                     op MLP fp8 en el grafo, el segundo lote (direcciones nuevas por el prefetch) disparaba un acceso fuera de
#                     rango en la acumulación del gradiente de embeddings (P99e); el diagnóstico con un lote fijo pasaba.
#   ARCH_SCHED_BY_STEP=1 rampa de lote, alfa MTP, warmdown y atado por fracción de PASOS (step/num_steps) en vez de
#                     reloj: comparaciones a igual exposición (mismo número de pasos y horarios); solo diagnóstico.
#   ARCH_MLP_FP8=1    MLP ReLU² fusionado en fp8 double_row (kernel NKI propio, forward; backward en torch) en el
#                     dispositivo; en CPU cae al camino torch. ARCH_MLP_FP8_LNC=2 lanza con rejilla wrap_nki(kernel)[2]
#                     (dos núcleos físicos, reparto por program_id). ARCH_MLP_FP8_BWDK=1: gradiente de activación (dpre^T, dx) también en NKI fp8.
#   ARCH_MLP_FP8_DYSCALE=22  (con ARCH_MLP_FP8_BWDK=1) escala estática S = 2^k de los gradientes de activación antes de copiarlos a
#                     e5m2 en mlp_fp8_bwd_act. dy real (p50 5.6e-9, máximo 4.9e-7) y dpre (máximo 1.3e-5) quedan bajo el subnormal
#                     mínimo de e5m2 (2^-16): sin escala se aplanaba el 100 % y dx y dW1 del MLP salían a cero exacto (P99s, F8s;
#                     research/fp8-dy/resultado.md). Con S: dys = e5m2(S·dy) y dps = e5m2(S·dpre^T) con tensor_scalar sobre los
#                     tiles de SBUF (columna 4 de sc) en lugar de tensor_copy; el 1/S va en los multiplicadores de los epílogos que ya
#                     existían (columnas 2 y 3 de sc), así que dpre^T (el que usa dW1 en torch) y dx salen sin escala y no se añade
#                     ninguna pasada. amax·2^22 medido: 2.0 en dy y 52.6 en dpre frente a 57344 (margen 2^10). 0 = kernel sin escala.
#                     FP8Z (27 sep): hb = _SCALE_H·relu(pre) se acota a _H_CLAMP = 15.4375 (exacto en bf16) con una tensor_scalar en SBUF
#                     antes del cuadrado: hs = hc² ≤ 238.3 < 240 (máximo de e4m3). Sin el clamp, (0.25·relu(pre))² desbordaba a inf con
#                     relu(pre) > 62 y FP8X_s42 murió con loss NaN en el paso 1765; rT guarda hc (acotado) y el backward no cambia. La
#                     variante FP8Y (clamp desde PSUM + activation(square) a e4m3) compiló pero corrió 4× más lenta (FP8Y_1nucleo).
#   ARCH_LAST_TARGET_GUARD=1  el ultimo objetivo del micro-lote (by[-1, -1]) pasa a -1 si es >= 8185 (BOS o reservado): el gather de
#                     la perdida con MTP lee una ventana de 8 columnas y su scatter se sale del buffer [N, 8192] solo en la ultima
#                     fila (NRT 1006 con la op fp8 en el grafo; bitacora 20260910T070115Z). Un objetivo de 16384 por micro-paso, solo
#                     en entrenamiento; sustituye a ARCH_VOCAB_PAD=64 (+0.017 s por paso). Evaluacion sin cambios.
#   ARCH_FP8_SIM=1|2  numérica fp8 emulada en los dos GEMM del MLP (política e4m3 legacy/e5m2 con escala por tensor,
#                     subnormales y saturación, en punto flotante; el gradiente entrante se cuantiza ANTES de dX y dW): 1 = e4m3 (3 bits de mantisa) en x, W1, h, W2 con STE; 2 = además e5m2 en los gradientes de
#                     activación. Mide el coste en bpb antes de construir el backward fp8 (P96/Q8). Solo diagnóstico.
#   ARCH_STALL_TEST=s DIAGNÓSTICO: duerme s segundos tras cada frontera de los pasos 5-19 (reproduce el atasco de
#                     fsync de S19: 22 s) para medir su efecto en val_bpb con horarios por tiempo. Nunca en un envío.
#   ARCH_SYNC_STARTUP=1 os.sync() antes de las 5 primeras fronteras de paso (ventana excluida): vacía las
#                     páginas sucias de los NEFF para que el fsync por paso del harness no atasque los pasos 5-20.
#   ARCH_STATIC_CE=1  pérdida con log_softmax + gather + máscara (formas estáticas) en vez de
#                     F.cross_entropy(ignore_index): evita la ruptura de grafo en la pérdida.
#                     all_reduce / optimizador por paso. ARCH_FLAT_SYNC=1 un all_reduce por dtype.
#   ARCH_PACK_MULT=8  filas de 8×SEQ_LEN en el loader y reshape a (·, SEQ_LEN):
#                     recupera los documentos largos que el best-fit descartaba y alinea
#                     la distribución con la de evaluación. Legal por escrito (1 sep).
#   ARCH_BF16=1       pesos en bf16 (como autoresearch/train.py); acompañar de
#                     NEURON_RT_STOCHASTIC_ROUNDING_EN=1 en el comando de lanzamiento.
#   ARCH_HEAD_BF16=1  cabeza + softcap + CE sin el .float(): la cabeza es el 7.8% de los
#                     FLOPs y en fp32 corre a 20 TFLOPS en vez de 79. Validar NaN en chip.
#   ARCH_CE_FUSED=1   (CEF, 30 sep) pérdida de ENTRENAMIENTO de la rama STATIC_CE (principal + MTP compartido) sin tensores
#                     [N, V] de log-probabilidades: z = x @ W[:V]^T (las 64 filas de relleno no entran: su sesgo -1e4 les da
#                     probabilidad 0 y gradiente 0, igual que antes); el kernel NKI arch::cef_fwd lee z por tiles [128, 2048],
#                     escribe t = tanh(z/15) en bf16 (único tensor ancho guardado para el backward; antes se guardaban lp y tanh)
#                     y devuelve lse = log(sum exp(15 t)) por fila (sin máximo: |15 t| <= 15); arch::cef_bwd devuelve
#                     dz = g (1 - t^2) exp(15 t - lse) en bf16. El término del objetivo, -15 tanh(x·W[y]/15), va aparte con las
#                     filas W[y] y W[y2] reunidas por _EmbScatter (dos llamadas EMBK de N filas en el backward, la misma forma que
#                     la de wte), así que ningún kernel CEF ve índices.
#                     Evaluación y rutas no cubiertas (MTP_HEAD, H3, prefijo, LEXSETS, parejas, LOSS_PADROWS, mask_special,
#                     logit_asym) siguen el camino original. En CPU los kernels se sustituyen por su emulación en torch.
#                     ARCH_CE_FUSED_LNC=2 rejilla wrap_nki(k)[2] (cada programa, la mitad de las filas).
#   ARCH_CE_FUSED_GATE=1 puerta en el rango 0 antes del reloj del bucle: los dos kernels sobre z = 4·randn [512, 8192] bf16
#                     (semilla 0) y g = 1e-3·rand con una fila a cero, frente a la emulación en CPU; imprime CEF_GATE y sale con 3
#                     si falla (|Δt| <= 2^-7, lse rel <= 1e-3, dz rel-Frobenius <= 2e-2, todo finito, fila g = 0 a cero).

WARMDOWN_FRAC = float(os.environ.get("ARCH_WARMDOWN", "0.5"))
LR_LEVELS = int(os.environ.get("ARCH_LR_LEVELS", "10"))
LR_TAIL = os.environ.get("ARCH_LR_TAIL", "1") == "1"
WARMDOWN_SHAPE = os.environ.get("ARCH_WARMDOWN_SHAPE", "sqrt")
assert WARMDOWN_SHAPE in ("lin", "sqrt"), WARMDOWN_SHAPE
_LR_LADDER = sorted({0.0, 0.02, 0.05, 0.075} | {(_i + 1) / max(LR_LEVELS, 1) for _i in range(max(LR_LEVELS, 1))})
LR_MULT = float(os.environ.get("ARCH_LR_MULT", "1.0"))
MUON_MOM_FINAL = float(os.environ.get("ARCH_MUON_MOM", "0.95"))
EMB_LR_MULT = float(os.environ.get("ARCH_EMB_LR_MULT", "1.0"))
EMA_DECAY = float(os.environ.get("ARCH_EMA", "0"))
DW_SAMPLE = float(os.environ.get("ARCH_DW_SAMPLE", "0"))
DW_SAMPLE_MODE = os.environ.get("ARCH_DW_SAMPLE_MODE", "bloque")
DW_SAMPLE_RESCALE = os.environ.get("ARCH_DW_SAMPLE_RESCALE", "0") == "1"
_DW_PASO = [0]
MLP_SAVED_RELU = int(os.environ.get("ARCH_MLP_SAVED_RELU", "1"))
ATTN_TAU = os.environ.get("ARCH_ATTN_TAU", "0") == "1"
RANK_SOUP = int(os.environ.get("ARCH_RANK_SOUP", "0"))
WAVG_FRAC = float(os.environ.get("ARCH_WAVG", "0"))
WAVG_K = int(os.environ.get("ARCH_WAVG_K", "6"))
EMA_STATE = {}
PACK_MULT = int(os.environ.get("ARCH_PACK_MULT", "16"))
BF16_WEIGHTS = os.environ.get("ARCH_BF16", "1") == "1"
HEAD_BF16 = os.environ.get("ARCH_HEAD_BF16", "1") == "1"
TIMING = os.environ.get("ARCH_TIMING", "0") == "1"
MUON_SHARD = os.environ.get("ARCH_MUON_SHARD", "0") == "1"
MUON_COMPILED_SHARD = os.environ.get("ARCH_MUON_COMPILED_SHARD", "1") == "1"
MUON_SLOW = os.environ.get("ARCH_MUON_SLOW", "0") == "1"
MUON_SLOW_BETA = float(os.environ.get("ARCH_MUON_SLOW_BETA", "0.999"))
MUON_SLOW_ALPHA = float(os.environ.get("ARCH_MUON_SLOW_ALPHA", "2.0"))
MUON_SLOW_WARM = int(os.environ.get("ARCH_MUON_SLOW_WARM", "3000"))
assert not MUON_SLOW or MUON_COMPILED_SHARD, "ARCH_MUON_SLOW exige ARCH_MUON_COMPILED_SHARD=1"
assert 0.0 <= MUON_SLOW_BETA < 1.0 and MUON_SLOW_ALPHA >= 0.0 and MUON_SLOW_WARM >= 0, "ARCH_MUON_SLOW_BETA en [0,1), ARCH_MUON_SLOW_ALPHA >= 0, ARCH_MUON_SLOW_WARM >= 0"
NO_ATTN = os.environ.get("ARCH_NO_ATTN", "0") == "1"
FLASH = int(os.environ.get("ARCH_FLASH", "0"))
MANUAL_ATTN = int(os.environ.get("ARCH_MANUAL_ATTN", "0"))
ATTN_WINDOW = int(os.environ.get("ARCH_ATTN_WINDOW", "0"))
CTE_MM_BF16 = os.environ.get("ARCH_CTE_MM_BF16", "0") == "1"
CTE_SOFTMAX_BF16 = os.environ.get("ARCH_CTE_SOFTMAX_BF16", "0") == "1"
CTE_BWD_MIXED = os.environ.get("ARCH_CTE_BWD_MIXED", "1") == "1"
CTE_TPQ = os.environ.get("ARCH_CTE_TPQ", "0") == "1"
ATTN_WINDOW_LAYERS = os.environ.get("ARCH_ATTN_WINDOW_LAYERS", "")
ATTN_CHUNKS = int(os.environ.get("ARCH_ATTN_CHUNKS", "4"))
MANUAL_ATTN_CKPT = os.environ.get("ARCH_MANUAL_ATTN_CKPT", "0") == "1"
MTP = int(os.environ.get("ARCH_MTP", "2"))
MTP_ALPHA = float(os.environ.get("ARCH_MTP_ALPHA", "1.0")) or (0.5 if MTP == 2 else 0.1)
MTP_PROBE = os.environ.get("ARCH_MTP_PROBE", "0") == "1"
MTP_END = float(os.environ.get("ARCH_MTP_END", "0.4"))
MTP_SHAPE = os.environ.get("ARCH_MTP_SHAPE", "lin")
MTP_ALPHA3 = float(os.environ.get("ARCH_MTP_ALPHA3", "0"))
MTP_H3 = MTP_ALPHA3 > 0
MTP_HEAD = int(os.environ.get("ARCH_MTP_HEAD", "0"))
MTP_HEAD_CHUNKS = int(os.environ.get("ARCH_MTP_HEAD_CHUNKS", "4"))
BOS_ID = 8188
PROF_DIR = os.environ.get("ARCH_PROFILE_DIR", "")
PROF_WAIT = int(os.environ.get("ARCH_PROFILE_WAIT", "30"))
FWD_PROBE = os.environ.get("ARCH_FWD_PROBE", "0") in ("1", "2")
FWD_PROBE_GRAD = os.environ.get("ARCH_FWD_PROBE", "0") == "2"
OPT_COMPILE = os.environ.get("ARCH_OPT_COMPILE", "1") == "1"
TBS_RAMP = float(os.environ.get("ARCH_TBS_RAMP", "0.3"))
TIE_UNTIL = float(os.environ.get("ARCH_TIE_UNTIL", "0"))
PREFETCH = os.environ.get("ARCH_PREFETCH", "1") == "1"
NO_BWD = os.environ.get("ARCH_NO_BWD", "0") == "1"
NO_ROPE = os.environ.get("ARCH_NO_ROPE", "0") == "1"
ROPE_MM = os.environ.get("ARCH_ROPE_MM", "1") == "1"
NO_NORM = os.environ.get("ARCH_NO_NORM", "0") == "1"
NORM_MANUAL = int(os.environ.get("ARCH_NORM_MANUAL", "4"))
QKV_FUSED = os.environ.get("ARCH_QKV_FUSED", "0") == "1"
QKNORM_KERNEL = os.environ.get("ARCH_QKNORM_KERNEL", "0") == "1"
ATTN_FUSED_BWD = os.environ.get("ARCH_ATTN_FUSED_BWD", "1") == "1"
ATTN_FUSED_BWD_GATE = os.environ.get("ARCH_ATTN_FUSED_BWD_GATE", "0") == "1"
ATTFB_LNC = int(os.environ.get("ARCH_ATTN_FUSED_LNC", "2"))
ATTFB_SIM = os.environ.get("ARCH_ATTN_FUSED_SIM", "0") == "1"
ATTN_DW8 = os.environ.get("ARCH_ATTN_DW8", "1") == "1"
ATTN_DW8_K = int(os.environ.get("ARCH_ATTN_DW8_K", "24"))
ATTN_DW8_LNC = int(os.environ.get("ARCH_ATTN_DW8_LNC", "2"))
ATTN_DW8_ACT = os.environ.get("ARCH_ATTN_DW8_ACT", "1") == "1"
assert not ATTN_DW8 or (ATTN_FUSED_BWD and 0 < ATTN_DW8_K < 40), "ARCH_ATTN_DW8=1 exige ARCH_ATTN_FUSED_BWD=1 y 0 < ARCH_ATTN_DW8_K < 40"
MOE_AUX = float(os.environ.get("ARCH_MOE_AUX", "0.01"))
MOE_LOOP = os.environ.get("ARCH_MOE_LOOP", "1") == "1"
MOE_CUMSUM_MM = os.environ.get("ARCH_MOE_CUMSUM_MM", "0") == "1"
MOE_SCATTER = os.environ.get("ARCH_MOE_SCATTER", "0") == "1"
MOE_DISPATCH = os.environ.get("ARCH_MOE_DISPATCH", "mm")
DOC_MASK = os.environ.get("ARCH_DOC_MASK", "0") == "1"
PREFIX_BETA = float(os.environ.get("ARCH_PREFIX_BETA", "0"))
_PREFIX_DEV = None
XSA_FP32_ALPHA = os.environ.get("ARCH_XSA_FP32ALPHA", "0") == "1"
_MEANM = {}
NO_QKNORM = os.environ.get("ARCH_NO_QKNORM", "0") == "1"
_TIE_SPLIT = [False]
_MASK_CACHE = {}
FLAT_SYNC = os.environ.get("ARCH_FLAT_SYNC", "0") == "1"
ACCUM_FOREACH = os.environ.get("ARCH_ACCUM_FOREACH", "1") == "1"
STATIC_CE = os.environ.get("ARCH_STATIC_CE", "1") == "1"
CE_FUSED = os.environ.get("ARCH_CE_FUSED", "1") == "1"
CE_FUSED_LNC = int(os.environ.get("ARCH_CE_FUSED_LNC", "2"))
CE_FUSED_GATE = int(os.environ.get("ARCH_CE_FUSED_GATE", "0"))
ADAMW_FOREACH = os.environ.get("ARCH_ADAMW_FOREACH", "1") == "1"
SYNC_STARTUP = os.environ.get("ARCH_SYNC_STARTUP", "0") == "1"
STALL_TEST = float(os.environ.get("ARCH_STALL_TEST", "0"))
FP8_SIM = int(os.environ.get("ARCH_FP8_SIM", "0"))
MLP_FP8 = os.environ.get("ARCH_MLP_FP8", "1") == "1"
MLP_FP8_LNC = int(os.environ.get("ARCH_MLP_FP8_LNC", "2"))
MLP_FP8_BWDK = os.environ.get("ARCH_MLP_FP8_BWDK", "1") == "1"
MLP_FP8_EAGER = os.environ.get("ARCH_MLP_FP8_EAGER", "0") == "1"
MLP_FP8_DYSCALE = int(os.environ.get("ARCH_MLP_FP8_DYSCALE", "22"))
MLP_DW8 = os.environ.get("ARCH_MLP_DW8", "1") == "1"
MLP_DW8_EVSB = os.environ.get("ARCH_MLP_DW8_EVSB", "0") == "1"
assert not MLP_DW8 or (MLP_FP8 and MLP_FP8_BWDK and MLP_FP8_DYSCALE > 0), "ARCH_MLP_DW8=1 exige ARCH_MLP_FP8=1, ARCH_MLP_FP8_BWDK=1 y ARCH_MLP_FP8_DYSCALE > 0"
MLP_EPIPASS = int(os.environ.get("ARCH_MLP_EPIPASS", "1"))
assert MLP_EPIPASS in (0, 1, 2) and (not MLP_EPIPASS or MLP_DW8), "ARCH_MLP_EPIPASS=1|2 exige ARCH_MLP_DW8=1"
MLP_FWDEPI = int(os.environ.get("ARCH_MLP_FWDEPI", "1"))
assert MLP_FWDEPI in (0, 1, 2) and (not MLP_FWDEPI or (MLP_EPIPASS == 1 and MLP_DW8)), "ARCH_MLP_FWDEPI es 0, 1 o 2, y 1|2 exige ARCH_MLP_EPIPASS=1 y ARCH_MLP_DW8=1"
KA_MLP = int(os.environ.get("ARCH_KA_MLP", "1"))
assert KA_MLP in (0, 1) and (not KA_MLP or MLP_FP8), "ARCH_KA_MLP es 0 o 1, y 1 exige ARCH_MLP_FP8=1"
MLP_HIDDEN = int(os.environ.get("ARCH_MLP_HIDDEN", "5376"))
assert MLP_HIDDEN >= 0 and not (MLP_HIDDEN and MLP_FP8 and MLP_HIDDEN % 256), "ARCH_MLP_HIDDEN con ARCH_MLP_FP8=1: multiplo de 256 (pares double_row de 128 filas)"
LAST_TARGET_GUARD = int(os.environ.get("ARCH_LAST_TARGET_GUARD", "0"))
SCHED_BY_STEP = os.environ.get("ARCH_SCHED_BY_STEP", "0") == "1"
STATIC_INPUTS = os.environ.get("ARCH_STATIC_INPUTS", "0") == "1"
DIAG_BATCH = os.environ.get("ARCH_DIAG_BATCH", "0") == "1"
EMB_CHUNKS = int(os.environ.get("ARCH_EMB_CHUNKS", "0"))
EMB_SCATTER = os.environ.get("ARCH_EMB_SCATTER", "1") == "1"
EMB_SCATTER_GATE = int(os.environ.get("ARCH_EMB_SCATTER_GATE", "0"))
EMB_SCATTER_LNC = int(os.environ.get("ARCH_EMB_SCATTER_LNC", "2"))
RESNORM_FUSED = int(os.environ.get("ARCH_RESNORM_FUSED", "0"))
RESNORM_LNC = int(os.environ.get("ARCH_RESNORM_LNC", "2"))
assert RESNORM_FUSED in (0, 1, 2), "ARCH_RESNORM_FUSED: 0 apagado, 1 todas las fronteras, 2 solo bloques sin atencion"
assert not (EMB_SCATTER and EMB_CHUNKS > 0), "ARCH_EMB_SCATTER y ARCH_EMB_CHUNKS reemplazan los dos el backward de las tablas"
CE_ONEHOT = os.environ.get("ARCH_CE_ONEHOT", "0") == "1"
LP_PAD = int(os.environ.get("ARCH_LP_PAD", "0"))
LOSS_PADROWS = int(os.environ.get("ARCH_LOSS_PADROWS", "0"))
STACKED_PARAMS = os.environ.get("ARCH_STACKED_PARAMS", "0") == "1"
MASTER_FP32 = os.environ.get("ARCH_MASTER_FP32", "1") == "1"
PAIR_PHASE = float(os.environ.get("ARCH_PAIR_PHASE", "0"))
PAIR_MODE = os.environ.get("ARCH_PAIR_MODE", "ordered")
PAIR_WARM = os.environ.get("ARCH_PAIR_WARM", "1") == "1"
TWOSTREAM = os.environ.get("ARCH_TWOSTREAM", "0") == "1"
TWOSTREAM_DIFF = os.environ.get("ARCH_TWOSTREAM_DIFF", "1") == "1"
MUON_VS = os.environ.get("ARCH_MUON_VS", "0") == "1"
HIER_ATTN = os.environ.get("ARCH_HIER_ATTN", "1") == "1"
HIER_ADAMW = os.environ.get("ARCH_HIER_ADAMW", "0") == "1"
HIER_WINDOW = int(os.environ.get("ARCH_HIER_WINDOW", "0"))
REUSE_RANK = int(os.environ.get("ARCH_REUSE_RANK", "0"))
REUSE_BETA0 = os.environ.get("ARCH_REUSE_BETA0", "0") == "1"
REUSE_B0 = float(os.environ.get("ARCH_REUSE_B0", "0.5"))
GROW = float(os.environ.get("ARCH_GROW", "0"))
_GROW_SET = frozenset(int(_v) for _v in os.environ.get("ARCH_GROW_LAYERS", "2,3,6,7").split(",") if _v)
MIMETIC_QK = os.environ.get("ARCH_MIMETIC_QK", "0") == "1"
MIMETIC_QK_ALPHA = float(os.environ.get("ARCH_MIMETIC_QK_ALPHA", "0.7"))
MIMETIC_QK_BETA = float(os.environ.get("ARCH_MIMETIC_QK_BETA", "0.7"))
assert not MIMETIC_QK or (MIMETIC_QK_ALPHA > 0 and MIMETIC_QK_BETA >= 0), "ARCH_MIMETIC_QK: alfa > 0 (con alfa = 0 el subespacio de rango head_dim de beta I es arbitrario) y beta >= 0"
LEXSETS = float(os.environ.get("ARCH_LEXSETS", "0"))
LEXSETS_END = float(os.environ.get("ARCH_LEXSETS_END", str(2.0 / 3.0)))
LEXSETS_RANDOM = int(os.environ.get("ARCH_LEXSETS_RANDOM", "0"))
LEXSETS_GROUP = int(os.environ.get("ARCH_LEXSETS_GROUP", "64"))
_LEX_DEV = None
LEXSETS_MM = os.environ.get("ARCH_LEXSETS_MM", "0") == "1"
NGMEM = os.environ.get("ARCH_NGMEM", "0") == "1"
NGMEM_N = int(os.environ.get("ARCH_NGMEM_N", "8"))
NGMEM_ROWS = int(os.environ.get("ARCH_NGMEM_ROWS", str(1 << 24)))
NGMEM_DIM = int(os.environ.get("ARCH_NGMEM_DIM", "16"))
NGMEM_LR = float(os.environ.get("ARCH_NGMEM_LR", "0"))
NGMEM_RESERVE = float(os.environ.get("ARCH_NGMEM_RESERVE", "10"))
assert not NGMEM or (1 <= NGMEM_N <= 32 and 0 < NGMEM_ROWS <= 1 << 24 and NGMEM_DIM > 0), "ARCH_NGMEM: 1 <= N <= 32, 0 < ROWS <= 2^24 (los indices viajan en fp32 por el all_reduce) y DIM > 0"
if NGMEM:
    CHECKPOINT_RESERVE_SECONDS = CHECKPOINT_RESERVE_SECONDS + NGMEM_RESERVE
_LEX_M = None
_LEX_G = None
NF_ATTN = os.environ.get("ARCH_NF_ATTN", "0") == "1"
NF_FFN = os.environ.get("ARCH_NF_FFN", "0") == "1"
assert not (NF_FFN and MLP_FP8), "ARCH_NF_FFN=1 exige ARCH_MLP_FP8=0: la norma va sobre la oculta h = relu(W1 x)^2 del camino bf16; con el MLP fp8 la oculta vive dentro del kernel NKI y ese kernel no la normaliza"
assert not (NF_FFN and (FP8_SIM or DW_SAMPLE > 0 or REUSE_RANK or MLP_SAVED_RELU == 2)), "ARCH_NF_FFN=1 no admite ARCH_FP8_SIM, ARCH_DW_SAMPLE, ARCH_REUSE_RANK ni ARCH_MLP_SAVED_RELU=2"


def _nf_scale(x):
    return torch.rsqrt((x * x).mean(-1, keepdim=True) + torch.finfo(x.dtype).eps)


def nf_desde_estado(state):
    global NF_ATTN, NF_FFN
    NF_ATTN = NF_ATTN or "nf_attn_on" in state
    NF_FFN = NF_FFN or "nf_ffn_on" in state


def _nll_pick(lp, t):
    """-log p del objetivo t (ya ≥ 0). ARCH_LP_PAD=K: las log-probabilidades se rellenan con K columnas a -1e4 (probabilidad
    nula, misma pérdida) para que el scatter de 8 elementos del backward del gather no se salga del buffer cuando el último
    objetivo del lote está a menos de 8 columnas del final del vocabulario (fallo NRT 1006 con la op MLP fp8, MD17).
    ARCH_CE_ONEHOT=1: producto con one-hot por comparación (no evita el fallo: MD18)."""
    if LP_PAD > 0:
        lp = F.pad(lp, (0, LP_PAD), value=-1e4)
    if CE_ONEHOT:
        oh = (t.unsqueeze(1) == torch.arange(lp.size(-1), device=t.device, dtype=t.dtype).unsqueeze(0)).to(lp.dtype)
        return -(lp * oh).sum(-1).float()
    return -lp.gather(1, t.unsqueeze(1)).squeeze(1).float()


class _EmbChunked(torch.autograd.Function):
    """Embedding con el gradiente de la tabla acumulado por trozos de tokens (index_add_ en fp32): evita que la acumulación
    indirecta del compilador reciba un id con más de ~512 apariciones en una sola op (fallo NRT 1006 con la op MLP fp8)."""
    @staticmethod
    def forward(ctx, idx, weight):
        ctx.save_for_backward(idx)
        ctx.shape = weight.shape
        ctx.dtype = weight.dtype
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        fi = idx.reshape(-1)
        fg = g.reshape(-1, g.shape[-1]).float()
        dw = torch.zeros(ctx.shape, dtype=torch.float32, device=g.device)
        n = fi.shape[0]
        k = max(1, EMB_CHUNKS)
        c = (n + k - 1) // k
        for j in range(k):
            dw.index_add_(0, fi[j * c:(j + 1) * c], fg[j * c:(j + 1) * c])
        return None, dw.to(ctx.dtype)


_EMBK_OP = None
_EMBK_TOL_MAX, _EMBK_TOL_FROB = 1e-3, 1e-4


def _embk_dw(fi, fg, rows):
    """dW fp32 [rows, D] = index_add(0, fi, fg): kernel arch::embk_dw en el dispositivo, index_add_ fp32 en CPU."""
    if fg.device.type == "cpu":
        return torch.zeros((rows, fg.shape[1]), dtype=torch.float32, device=fg.device).index_add_(0, fi.long(), fg.float())
    assert _EMBK_OP is not None, "ARCH_EMB_SCATTER=1 en el dispositivo necesita nki y torch_neuronx"
    assert fi.shape[0] % 128 == 0 and fg.dtype in (torch.bfloat16, torch.float32), (fi.shape, fg.dtype)
    return _EMBK_OP(fg.contiguous(), fi.to(torch.int32).contiguous(), rows)[:rows]


class _EmbScatter(torch.autograd.Function):
    """ARCH_EMB_SCATTER=1: forward F.embedding; backward dW fp32 por dispersión-suma (_embk_dw) con el dtype del parámetro."""
    @staticmethod
    def forward(ctx, idx, weight):
        ctx.save_for_backward(idx)
        ctx.shape = weight.shape
        ctx.dtype = weight.dtype
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        dw = _embk_dw(idx.reshape(-1), g.reshape(-1, g.shape[-1]), ctx.shape[0])
        return None, dw.to(ctx.dtype)


def embk_compara(k, ref):
    """Regla de la puerta EMBK: finito, max|k − ref| / max|ref| ≤ 1e-3 y ‖k − ref‖F / ‖ref‖F ≤ 1e-4."""
    k64, r64 = k.detach().cpu().double(), ref.detach().cpu().double()
    fin = bool(torch.isfinite(k64).all())
    d = k64 - r64
    maxrel = d.abs().max().item() / max(r64.abs().max().item(), 1e-300)
    frob = d.norm().item() / max(r64.norm().item(), 1e-300)
    return {"fin": fin, "maxrel": maxrel, "frob": frob, "ok": fin and maxrel <= _EMBK_TOL_MAX and frob <= _EMBK_TOL_FROB}


def embk_puerta_entradas(ids, d, semilla=0):
    fi = ids.reshape(-1).detach().cpu().long()
    g = torch.Generator().manual_seed(semilla)
    dy = (torch.randn(fi.numel(), d, generator=g) * 1e-3).bfloat16()
    rep = fi.clone()
    rep[:5000] = torch.bincount(fi).argmax()
    return fi, rep, dy


def _embk_puerta_llamada(dy, idx, rows):
    return _embk_dw(idx, dy, rows)


def _embk_ref_llamada(dy, idx, rows):
    return torch.zeros((rows, dy.shape[1]), dtype=torch.float32, device=dy.device).index_add_(0, idx.long(), dy.float())


def embk_puerta_dispositivo(device, ids, rows, d):
    fi, rep, dy = embk_puerta_entradas(ids, d)
    neuron = device.type == "neuron"
    f = torch.compile(_embk_puerta_llamada, backend="neuron", dynamic=False) if neuron else _embk_puerta_llamada
    fr = torch.compile(_embk_ref_llamada, backend="neuron", dynamic=False) if neuron else _embk_ref_llamada
    r = {"maxdup": int(torch.bincount(fi).max().item()), "maxdup_rep": int(torch.bincount(rep).max().item()),
         "dev_ref": "ok" if EMB_SCATTER_GATE >= 2 else "omitida"}
    peor = {"fin": True, "maxrel": 0.0, "frob": 0.0, "ok": True}
    for nombre, ii in (("real", fi), ("rep", rep)):
        with torch.no_grad():
            k = f(dy.to(device), ii.to(torch.int32).to(device), rows).cpu()
        c = embk_compara(k, _embk_ref_llamada(dy, ii, rows))
        r[f"{nombre}_cpu"] = c
        peor = {"fin": peor["fin"] and c["fin"], "maxrel": max(peor["maxrel"], c["maxrel"]), "frob": max(peor["frob"], c["frob"]),
                "ok": peor["ok"] and c["ok"]}
        if EMB_SCATTER_GATE >= 2:
            try:
                with torch.no_grad():
                    r[f"{nombre}_dev"] = embk_compara(k, fr(dy.to(device), ii.to(device), rows).cpu())
            except Exception as e:
                r["dev_ref"] = f"{type(e).__name__}: {str(e)[:120]}"
    r.update(peor)
    return r


def embk_linea(r, sr):
    casos = " ".join(f"{k}={v['maxrel']:.3e}/{v['frob']:.3e}" for k, v in r.items() if isinstance(v, dict))
    return (f"EMBK_GATE ok={r['ok']} maxrel={r['maxrel']:.4e} frob={r['frob']:.4e} maxdup={r['maxdup']} sr={sr} "
            f"maxdup_rep={r['maxdup_rep']} fin={r['fin']} dev_ref={r['dev_ref']!r} {casos}")


def _emb(mod, idx):
    if EMB_SCATTER:
        return _EmbScatter.apply(idx, mod.weight)
    return _EmbChunked.apply(idx, mod.weight) if EMB_CHUNKS > 0 else mod(idx)


def _sched_progress(step, args, budget_started):
    if SCHED_BY_STEP:
        return step / max(1, args.num_steps)
    return (time.monotonic() - budget_started) / max(1.0, args.max_train_seconds)


def _pair_pool(e):
    a, b = e[:, 0::2], e[:, 1::2]
    if PAIR_MODE == "mean":
        return (a + b) / math.sqrt(2.0)
    if PAIR_MODE == "last":
        return b
    assert PAIR_MODE == "ordered", PAIR_MODE
    h = e.size(-1) // 2
    return torch.cat((b[..., :h] - a[..., h:], b[..., h:] + a[..., :h]), dim=-1) / math.sqrt(2.0)


class _TwoStreamRouter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5, 2 * d))
        self.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, math.log(99.0)]))
        self.scale = nn.Parameter(torch.full((3,), 0.01))

    def forward(self, u, v):
        f = norm(torch.cat((u, v), dim=-1))
        dyn = F.linear(f.to(self.weight.dtype), self.weight).float()
        sc = self.scale.float()[torch.tensor([0, 0, 1, 1, 2], device=u.device)]
        raw = dyn * sc + self.bias.float()
        return torch.sigmoid(raw[..., :2]), 2.0 * torch.sigmoid(raw[..., 2:4]), torch.sigmoid(raw[..., 4:])


def _ts_block(block, u, v, x0, cos, sin, ve, kmask, ctx):
    a, b, p = block.ts_router(u, v)
    a, b, p = a.to(u.dtype), b.to(u.dtype), p.to(u.dtype)
    z = a[..., :1] * u + a[..., 1:] * v
    h = block.resid_lambda * z + block.x0_lambda * x0
    delta = h - z
    if block.attn is not None:
        _a = block.attn(norm(h), cos, sin, None if (ve is None or block.ve_lambda is None) else block.ve_lambda.to(ve.dtype) * ve, kmask)
        h = h + _a
        delta = delta + _a
    _m = block.mlp(norm(h))
    delta = delta + _m
    q = 1.0 - p
    return p * u + q * v + b[..., :1] * delta, q * u + p * v + b[..., 1:] * delta, ctx


def _band_mask(t, w, device, dtype):
    key = ("band", t, w, str(device), str(dtype))
    if key not in _MASK_CACHE:
        i = torch.arange(t, device=device)
        keep = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - w)
        _MASK_CACHE[key] = torch.where(keep, torch.zeros((), device=device), torch.full((), -1e4, device=device)).to(dtype)
    return _MASK_CACHE[key]


def _pair_warm_caches(model):
    w = model.transformer.wte.weight
    hd = model.config.n_embd // model.config.n_head
    with torch.no_grad():
        qknorm(torch.zeros(1, 1, 1, hd, device=w.device, dtype=COMPUTE_DTYPE))
        norm(torch.zeros(1, 1, model.config.n_embd, device=w.device, dtype=COMPUTE_DTYPE))
        if MANUAL_ATTN in (1, 2):
            for t in (model.config.sequence_len, model.config.sequence_len // 2):
                z = torch.zeros(1, 1, t, hd, device=w.device, dtype=COMPUTE_DTYPE)
                _manual_attention(z, z, z)
        if HIER_WINDOW and model.config.hier_width:
            _band_mask(model.config.sequence_len, HIER_WINDOW, w.device, COMPUTE_DTYPE)


def _lex_members_table(tokenizer, n):
    if LEXSETS_RANDOM > 0:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(LEXSETS_RANDOM))
    else:
        # get_token_bytes contains byte COUNTS for scoring, not token contents.
        raw = [tokenizer.enc.decode_single_token_bytes(i) for i in range(n)]
        order = torch.tensor(sorted(range(n), key=lambda i: (raw[i], i)), dtype=torch.long)
    assert n % LEXSETS_GROUP == 0
    groups = order.view(n // LEXSETS_GROUP, LEXSETS_GROUP)
    members = torch.empty((n, LEXSETS_GROUP), dtype=torch.long)
    members[order] = groups.repeat_interleave(LEXSETS_GROUP, dim=0)
    return members


def _lex_group_tables(members):
    n = members.size(0)
    _, inv = torch.unique(members.min(dim=1).values, return_inverse=True)
    m = torch.zeros(n, int(inv.max()) + 1, dtype=torch.float32)
    m[torch.arange(n), inv] = 1.0
    return m, inv


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
    torch.manual_seed(int(os.environ.get("ARCH_SEED", "48")))
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
    attn_tau: bool = False
    ve_layers: str = ""
    skip_src: int = -1
    skip_dst: int = -1
    smear: bool = False
    unet: bool = False
    mask_special: bool = False
    vocab_pad: int = 0
    hier_width: int = 0
    attn_layers: str = ""
    mlp_mult: int = 4
    mlp_expand: str = ""
    mlp_budget: int = 0
    mlp_gated: str = ""
    bigram_n: int = 0
    bigram_dim: int = 0
    mtp_head: int = 0
    attn_width: int = 0
    hourglass: str = ""
    logit_asym: bool = False
    partial_rope: bool = False
    rope_learned: bool = False
    doc_mask: bool = False
    key_shift_layers: str = ""
    mudd_tail: bool = False
    xsa: bool = False
    ctx_rank: int = 0
    lga_layers: str = ""
    lga_heads: int = 2
    lga_gate: bool = True
    cfv_layers: str = ""
    tng_rank: int = 0
    tng_order: int = 5
    tng_at: int = 2
    head_gate: bool = False
    fir_layers: str = ""
    ctx_source: str = "attn"
    moe_layers: str = ""
    moe_experts: int = 8
    moe_k: int = 2
    moe_hidden: int = 0
    moe_cap: float = 1.25


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


def _mean_matrix(d, device, dtype, cols=0):
    cols = cols or d
    key = (d, cols, str(device), str(dtype))
    if key not in _MEANM:
        _MEANM[key] = torch.full((d, cols), 1.0 / d, dtype=torch.float32).to(device=device, dtype=dtype)
    return _MEANM[key]


def _rms_manual(x: torch.Tensor, mm: bool = False, slim: bool = False) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    if NORM_MANUAL == 1:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
    if mm and slim:
        return x * torch.rsqrt(torch.matmul(x * x, _mean_matrix(x.shape[-1], x.device, x.dtype, 128))[..., :1] + eps)
    if mm:
        return x * torch.rsqrt(torch.matmul(x * x, _mean_matrix(x.shape[-1], x.device, x.dtype)) + eps)
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)


def norm(x: torch.Tensor) -> torch.Tensor:
    if NO_NORM:
        return x
    if NORM_MANUAL:
        return _rms_manual(x, mm=NORM_MANUAL in (3, 5), slim=NORM_MANUAL == 5)
    return F.rms_norm(x, (x.size(-1),))


def _rms_block(x: torch.Tensor, mblock: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.matmul(x * x, mblock.to(x.dtype)) + torch.finfo(x.dtype).eps)


def qknorm(x: torch.Tensor) -> torch.Tensor:
    if NO_NORM or NO_QKNORM or NORM_MANUAL == 6:
        return x
    if NORM_MANUAL:
        return _rms_manual(x, mm=NORM_MANUAL >= 3)
    return F.rms_norm(x, (x.size(-1),))


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
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) -- eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False


_flash_attention = None
if FLASH == 1:
    from neuronxcc.nki.kernels.attention import FlashConfig as _FlashConfig
    from neuronxcc.nki.kernels.attention import flash_attn_bwd as _nki_flash_bwd
    from neuronxcc.nki.kernels.attention import flash_fwd as _nki_flash_fwd
    from torch_neuronx import nki_op, wrap_nki
    import dataclasses as _dc
    from nki import jit as _nki_jit

    def _as_new_kernel(k):
        return k if _dc.is_dataclass(k) else _nki_jit(k.func)

    _FLASH_FWD = wrap_nki(_as_new_kernel(_nki_flash_fwd))
    _FLASH_BWD = wrap_nki(_as_new_kernel(_nki_flash_bwd))
    _FLASH_CFG = _FlashConfig(seq_tile_size=int(os.environ.get("ARCH_FLASH_TILE", "1024")), attn_core_tile_size=8, training=True, should_transpose_v=False, lse_dtype="float32")
    _FLASH_SCALE = HEAD_DIM ** -0.5

    @nki_op("arch::flash_fwd", mutates_args={})
    def _flash_fwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _FLASH_FWD[q.shape[0], q.shape[1]](q, k, v, seed, softmax_scale=_FLASH_SCALE, use_causal_mask=True, mixed_precision=True, dropout_p=0.0, config=_FLASH_CFG)

    @nki_op("arch::flash_bwd", mutates_args={})
    def _flash_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, dy: torch.Tensor, lse: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _FLASH_BWD[q.shape[0], q.shape[1]](q, k, v, o, dy, lse, seed, softmax_scale=_FLASH_SCALE, use_causal_mask=True, mixed_precision=True, dropout_p=0.0)

    def _flash_setup(ctx, inputs, output):
        q, k, v, seed = inputs
        ctx.save_for_backward(q, k, v, output[0], output[1], seed)
        ctx.mark_non_differentiable(output[1])

    def _flash_backward(ctx, do, dlse):
        q, k, v, o, lse, seed = ctx.saved_tensors
        dq, dk, dv = _flash_bwd_op(q, k, v.transpose(-1, -2).contiguous(), o.transpose(-1, -2).contiguous(), do.transpose(-1, -2).contiguous(), lse, seed)
        return dq, dk, dv.transpose(-1, -2), None

    _flash_fwd_op.register_autograd(_flash_backward, setup_context=_flash_setup)

    def _flash_attention(q, k, v):
        seed = torch.zeros(1, dtype=torch.int32, device=q.device)
        o, _ = _flash_fwd_op(q.transpose(-1, -2).contiguous(), k.transpose(-1, -2).contiguous(), v.contiguous(), seed)
        return o


if FLASH == 2:
    import nki.language as _nl
    from nkilib.core.attention.attention_bwd import attention_bwd as _nkilib_bwd
    from nkilib.core.attention.attention_cte import attention_cte as _nkilib_cte
    from torch_neuronx import nki_op, wrap_nki

    _CTE = wrap_nki(_nkilib_cte)
    _CTE_BWD = wrap_nki(_nkilib_bwd)
    _FLASH_SCALE = HEAD_DIM ** -0.5

    @nki_op("arch::cte_fwd", mutates_args={})
    def _cte_fwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _CTE(q, k, v, scale=(1.0 if w > 0 else _FLASH_SCALE), causal_mask=True, sliding_window=(w if w > 0 else None), tp_q=CTE_TPQ, tp_k=False, tp_out=False, cache_softmax=True, softmax_dtype=(_nl.bfloat16 if CTE_SOFTMAX_BF16 else _nl.float32), mm_out_dtype=(_nl.bfloat16 if CTE_MM_BF16 else _nl.float32))

    @nki_op("arch::cte_bwd", mutates_args={})
    def _cte_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, dy: torch.Tensor, lse: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _CTE_BWD(q, k, v, o, dy, lse, use_causal_mask=True, mixed_precision=CTE_BWD_MIXED, softmax_scale=(1.0 if w > 0 else _FLASH_SCALE), sliding_window=(w if w > 0 else None))

    def _cte_setup(ctx, inputs, output):
        q, k, v, h, w = inputs
        ctx.h = h
        ctx.w = w
        ctx.save_for_backward(q, k, v, output[0], output[1], output[2])
        ctx.mark_non_differentiable(output[1], output[2])

    def _cte_backward(ctx, do, dneg, drecip):
        q, k, v, o, neg_max, recip = ctx.saved_tensors
        if CTE_TPQ:
            q = q.transpose(-1, -2).contiguous()
        bh, d, s = q.shape
        h = ctx.h
        b = bh // h
        lse = (-neg_max.float() - torch.log(recip.float())).reshape(b, h, 128, s // 128).contiguous()
        dq, dk, dv = _cte_bwd_op(q.reshape(b, h, d, s), k.reshape(b, h, d, s), v.transpose(-1, -2).reshape(b, h, d, s).contiguous(), o.transpose(-1, -2).reshape(b, h, d, s).contiguous(), do.transpose(-1, -2).reshape(b, h, d, s).contiguous(), lse, ctx.w)
        _dq = dq.reshape(bh, d, s)
        if CTE_TPQ:
            _dq = _dq.transpose(-1, -2)
        return _dq, dk.reshape(bh, d, s), dv.reshape(bh, d, s).transpose(-1, -2), None, None

    _cte_fwd_op.register_autograd(_cte_backward, setup_context=_cte_setup)

    def _flash_attention(q, k, v, w=0):
        b, h, s, d = q.shape
        if w > 0:
            q = q * _FLASH_SCALE
        _qk = q.reshape(b * h, s, d).contiguous() if CTE_TPQ else q.transpose(-1, -2).reshape(b * h, d, s).contiguous()
        o, _, _ = _cte_fwd_op(_qk, k.transpose(-1, -2).reshape(b * h, d, s).contiguous(), v.reshape(b * h, s, d).contiguous(), h, int(w))
        return o.reshape(b, h, s, d)


def _manual_attention_chunked(q, k, v):
    b, h, t, d = q.shape
    n = ATTN_CHUNKS
    c = t // n
    scale = d ** -0.5
    outs = []
    for i in range(n):
        key = ("chunk", i, c, str(q.device), str(q.dtype))
        if key not in _MASK_CACHE:
            m = torch.zeros((c, (i + 1) * c), device=q.device)
            m[:, i * c:] = torch.full((c, c), -1e4, device=q.device).triu_(1)
            _MASK_CACHE[key] = m.to(q.dtype)
        qs = q[:, :, i * c:(i + 1) * c]
        ks = k[:, :, :(i + 1) * c]
        vs = v[:, :, :(i + 1) * c]
        att = torch.matmul(qs, ks.transpose(-1, -2)) * scale + _MASK_CACHE[key]
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        outs.append(torch.matmul(att, vs))
    return torch.cat(outs, dim=-2)


class _FlashManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, n_chunks):
        b, h, t, d = q.shape
        c = t // n_chunks
        scale = d ** -0.5
        outs, lses = [], []
        for i in range(n_chunks):
            e = (i + 1) * c
            key = ("fm", i, c, str(q.device))
            if key not in _MASK_CACHE:
                m = torch.zeros((c, e), device=q.device)
                m[:, i * c:] = torch.full((c, c), -1e4, device=q.device).triu_(1)
                _MASK_CACHE[key] = m
            sc = torch.matmul(q[:, :, i * c:e], k[:, :, :e].transpose(-1, -2)).float() * scale + _MASK_CACHE[key]
            mx = sc.amax(dim=-1, keepdim=True)
            pe = torch.exp(sc - mx)
            l = pe.sum(dim=-1, keepdim=True)
            outs.append(torch.matmul((pe / l).to(v.dtype), v[:, :, :e]))
            lses.append((mx + torch.log(l)).squeeze(-1))
        o = torch.cat(outs, dim=-2)
        lse = torch.cat(lses, dim=-1)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.n_chunks = n_chunks
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        b, h, t, d = q.shape
        n_chunks = ctx.n_chunks
        c = t // n_chunks
        scale = d ** -0.5
        dq_parts = []
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        delta = (do.float() * o.float()).sum(dim=-1, keepdim=True)
        for i in range(n_chunks):
            e = (i + 1) * c
            key = ("fm", i, c, str(q.device))
            qi = q[:, :, i * c:e]
            sc = torch.matmul(qi, k[:, :, :e].transpose(-1, -2)).float() * scale + _MASK_CACHE[key]
            pr = torch.exp(sc - lse[:, :, i * c:e].unsqueeze(-1))
            doi = do[:, :, i * c:e]
            dv[:, :, :e] += torch.matmul(pr.transpose(-1, -2).to(doi.dtype), doi).float()
            dp = torch.matmul(doi, v[:, :, :e].transpose(-1, -2)).float()
            ds = pr * (dp - delta[:, :, i * c:e])
            dq_parts.append(torch.matmul(ds.to(k.dtype), k[:, :, :e]) * scale)
            dk[:, :, :e] += (torch.matmul(ds.transpose(-1, -2).to(qi.dtype), qi) * scale).float()
        return torch.cat(dq_parts, dim=-2).to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


MUDD_SCALE = 0.2 / (3 ** 0.5)
KEY_SHIFT_SET = set(os.environ.get("ARCH_KEY_SHIFT_LAYERS", "").split(","))


def _key_shift(k, kmask):
    kp = torch.cat((k[:, :1], k[:, :-1]), dim=1)
    ks = torch.where(kmask, k, kp)
    return torch.cat((k[..., :32], ks[..., 32:64], k[..., 64:96], ks[..., 96:128]), dim=-1)


def _hg_pool(h):
    hp = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), dim=1)
    return (hp[:, 0::2] + h[:, 0::2]) * 0.5


def _manual_attention(q, k, v):
    if MANUAL_ATTN == 4:
        return _FlashManual.apply(q, k, v, ATTN_CHUNKS)
    if MANUAL_ATTN == 3:
        return _manual_attention_chunked(q, k, v)
    t = q.shape[-2]
    key = (t, str(q.device), str(q.dtype))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = torch.full((t, t), -1e4, device=q.device).triu_(1).to(q.dtype)
    att = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5) + _MASK_CACHE[key]
    if MANUAL_ATTN == 2:
        att = torch.softmax(att, dim=-1)
    else:
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
    return torch.matmul(att, v)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the example NKI kernel when enabled, else eager."""
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()


_DW_ID = [0]


def _dw_next():
    _DW_ID[0] += 1
    return _DW_ID[0] * 1013


class _MLPReluGuardada(torch.autograd.Function):
    """ReLU^2 con una sola activacion ancha guardada: r = relu(x W1^T). El cuadrado y 2r se reconstruyen
    en el backward a partir de r; la preactivacion no se guarda. Aritmetica identica."""

    @staticmethod
    def forward(ctx, x, w1, w2):
        r = F.relu(F.linear(x, w1.to(dtype=x.dtype), None))
        ctx.save_for_backward(x, w1, w2, r)
        return F.linear(r.square(), w2.to(dtype=x.dtype), None)

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, r = ctx.saved_tensors
        _w1, _w2 = w1.to(dtype=dy.dtype), w2.to(dtype=dy.dtype)
        dz = (dy @ (_w2 + _w2)) * r
        dx = dz @ _w1
        _x2, _dz2 = x.reshape(-1, x.shape[-1]), dz.reshape(-1, dz.shape[-1])
        _dy2, _a2 = dy.reshape(-1, dy.shape[-1]), r.square().reshape(-1, r.shape[-1])
        dw1 = (_dz2.t().to(torch.float32) @ _x2.to(torch.float32)).to(w1.dtype)
        dw2 = (_dy2.t().to(torch.float32) @ _a2.to(torch.float32)).to(w2.dtype)
        return dx, dw1, dw2


class _MLPReluNormGuardada(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        r = F.relu(F.linear(x, w1.to(dtype=x.dtype), None))
        h = r.square()
        ctx.save_for_backward(x, w1, w2, r)
        return F.linear(h * _nf_scale(h), w2.to(dtype=x.dtype), None)

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, r = ctx.saved_tensors
        _w1, _w2 = w1.to(dtype=dy.dtype), w2.to(dtype=dy.dtype)
        h = r.square()
        s = _nf_scale(h)
        n = h * s
        dn = dy @ _w2
        dz = (s * (dn - n * (dn * n).mean(-1, keepdim=True))) * (r + r)
        dx = dz @ _w1
        _x2, _dz2 = x.reshape(-1, x.shape[-1]), dz.reshape(-1, dz.shape[-1])
        _dy2, _n2 = dy.reshape(-1, dy.shape[-1]), n.reshape(-1, n.shape[-1])
        dw1 = (_dz2.t().to(torch.float32) @ _x2.to(torch.float32)).to(w1.dtype)
        dw2 = (_dy2.t().to(torch.float32) @ _n2.to(torch.float32)).to(w2.dtype)
        return dx, dw1, dw2


def _mlp_nf(mlp, x):
    if MLP_SAVED_RELU and mlp.training and not getattr(mlp, "ctx_in", 0):
        return _MLPReluNormGuardada.apply(x, mlp.c_fc.weight, mlp.c_proj.weight)
    h = relu_squared(mlp.c_fc(x))
    return mlp.c_proj(h * _nf_scale(h))


class _MLPReluRecalculada(torch.autograd.Function):
    """ReLU^2 sin ninguna activacion ancha guardada: solo x (b*t x d), W1 y W2. El backward recalcula
    r = relu(x W1^T) con un GEMM mas por bloque y de ahi saca r^2 y 2r. Aritmetica identica a
    _MLPReluGuardada; cambia solo que trafico de HBM por FLOPs (candidato RSI MS3, 15 sep)."""

    @staticmethod
    def forward(ctx, x, w1, w2):
        r = F.relu(F.linear(x, w1.to(dtype=x.dtype), None))
        ctx.save_for_backward(x, w1, w2)
        return F.linear(r.square(), w2.to(dtype=x.dtype), None)

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2 = ctx.saved_tensors
        _w1, _w2 = w1.to(dtype=x.dtype), w2.to(dtype=dy.dtype)
        r = F.relu(F.linear(x, _w1, None))
        dz = (dy @ _w2) * (2 * r)
        dx = dz @ _w1.to(dtype=dy.dtype)
        _x2, _dz2 = x.reshape(-1, x.shape[-1]), dz.reshape(-1, dz.shape[-1])
        _dy2, _a2 = dy.reshape(-1, dy.shape[-1]), r.square().reshape(-1, r.shape[-1])
        dw1 = (_dz2.t().to(torch.float32) @ _x2.to(torch.float32)).to(w1.dtype)
        dw2 = (_dy2.t().to(torch.float32) @ _a2.to(torch.float32)).to(w2.dtype)
        return dx, dw1, dw2


class _DWMuestreado(torch.autograd.Function):
    """Forward y gradiente de activaciones EXACTOS; el gradiente de pesos, sobre una muestra de filas.

    dW = (1/p) X[S]^T G[S] es insesgado condicionado a X y G. Lo que NO es insesgado es lo que Muon hace
    despues con el, porque ortogonaliza y reescala, y eso no se acota leyendo el codigo: lo decide la
    sonda de calidad."""

    @staticmethod
    def forward(ctx, x, w, k, desplazamiento, modo):
        _x2 = x.reshape(-1, x.shape[-1])
        _n = _x2.shape[0]
        if modo == "bloque":
            _ini = (desplazamiento * 128) % max(1, _n - k + 1)
            _xs, _idx = _x2[_ini:_ini + k], torch.empty(0, dtype=torch.long, device=x.device)
        else:
            _gen = torch.Generator().manual_seed(0x5EED ^ desplazamiento)
            _idx = torch.randperm(_n, generator=_gen)[:k].to(x.device)
            _xs, _ini = _x2.index_select(0, _idx), 0
        ctx.save_for_backward(w, _xs, _idx)
        ctx.k, ctx.modo, ctx.ini = k, modo, _ini
        return F.linear(x, w.to(dtype=x.dtype), None)

    @staticmethod
    def backward(ctx, g):
        w, _xs, _idx = ctx.saved_tensors
        dx = g @ w.to(dtype=g.dtype)
        _g2 = g.reshape(-1, g.shape[-1])
        _gs = _g2[ctx.ini:ctx.ini + ctx.k] if ctx.modo == "bloque" else _g2.index_select(0, _idx)
        dw = _gs.t().to(torch.float32) @ _xs.to(torch.float32)
        if DW_SAMPLE_RESCALE:
            # El reescalado 1/p NO hace falta para estas dos matrices: van a Muon, que ortogonaliza,
            # y la ortogonalizacion elimina cualquier constante global del gradiente. Comprobado:
            # cinco pasos con gradiente x1 y x2 dan pesos identicos BIT A BIT. Costaba +42 pasadas
            # elementales, mas que toda la temperatura de atencion. Queda solo para el modo de
            # referencia y para cualquier matriz que algun dia no pase por Muon.
            dw = dw * (_g2.shape[0] / ctx.k)
        return dx, dw.to(w.dtype), None, None, None


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight)."""

    dw_sample = False
    dw_id = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DW_SAMPLE > 0 and self.dw_sample and self.training and x.dim() >= 2:
            _n = x.numel() // x.shape[-1]
            _k = max(128, int(_n * DW_SAMPLE) // 128 * 128)
            if _k < _n:
                # Ventana FIJA por matriz (14 sep): un desplazamiento que cambia por llamada se cuece
                # como constante en el grafo compilado y obliga a recompilar en cada paso; DW0 se paso
                # 48 min compilando sin llegar al paso 0. Con la ventana fija el grafo compila una vez.
                # Sigue siendo una muestra uniforme: las filas de un lote son secuencias aleatorias del
                # corpus, asi que la mitad fija de las secuencias es una mitad aleatoria distinta en
                # cada paso. Las dos matrices del MLP usan ventanas distintas (dw_id).
                return _DWMuestreado.apply(x, self.weight, _k, self.dw_id, DW_SAMPLE_MODE)
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def _rot_matrix(d: int) -> torch.Tensor:
    h = d // 2
    z = torch.zeros(h, h, dtype=torch.float32)
    i = torch.eye(h, dtype=torch.float32)
    return torch.cat((torch.cat((z, -i), dim=1), torch.cat((i, z), dim=1)), dim=0)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor | None = None) -> torch.Tensor:
    if NO_ROPE:
        return x
    if ROPE_MM:
        return x * cos + torch.matmul(x, rot.to(x.dtype)) * sin
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


try:
    import nki as _nki_probe
    _HAS_NKI = True
except ImportError:
    _HAS_NKI = False
if MLP_FP8 and _HAS_NKI:
    import nki as _nki
    import nki.language as nl
    import nki.isa as nisa
    _P8, _NT8, _TB8 = 128, 512, 512
    _FP8 = nl.float8_e4m3
    _FP8G = nl.float8_e5m2
    _SCALE_H = 0.25
    _H_CLAMP = 15.4375
    _DR = nisa.matmul_perf_mode.double_row
    _S8G = float(2 ** MLP_FP8_DYSCALE) if MLP_FP8_DYSCALE > 0 else 0.0
    _EPI = MLP_EPIPASS if MLP_DW8 and _S8G else 0
    _FEP = MLP_FWDEPI if _EPI == 1 else 0
    _KAM = KA_MLP == 1


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




    @_nki.jit
    def mlp_fp8_bwd_act(dy, w2T, w1T, rT, sc):
        """Gradiente de activación del MLP en fp8 double_row: dpre^T [H, M] = (W2 · dy^T) ⊙ 2 r^T y dx [M, D] = dpre · W1^T.
        dy [M, D] bf16; w2T = c_proj.weight [D, H]; w1T = c_fc.weight [H, D]; rT [H, M] = relu(_SCALE_H·pre) del forward.
        Devuelve (dpreT [H, M] bf16, dx [M, D] bf16); dpre^T sirve al compilador para dW1 = (dpre^T · x)^T.
        Con _S8G = S > 0 (ARCH_MLP_FP8_DYSCALE), dy y dpre^T entran en e5m2 multiplicados por S = sc[:, 4] y los epílogos deshacen S
        con sc[:, 2] y sc[:, 3] (bwd_scales): dpreT y dx salen sin escala."""
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
                        nisa.scalar_tensor_tensor(dst=dps[:, c, j, :], data=ps, op0=nl.multiply, operand0=kgs, op1=nl.multiply, operand1=rt)
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


    def _dw8_fase(a, pk, sc, fase):
        """Una fase de ARCH_MLP_DW8=1 (dW del MLP en fp8 double_row, contrayendo los tokens en la partición). pk [H/128, 128, M/256,
        2, 128] (uint8 con bytes fp8) lo escribe mlp_fp8_bwd_act por teselas de 128 filas de H, con el token 256·s + 2·p + r en la
        partición p, el paso s y la fila r. fase 0: a = x [M, D] saturado a ±240 en e4m3, pk = dpk = e5m2(S·dpre), sale dW1o [H, D]
        = dpre^T · x con epílogo sc[:, 1] = 1/S. fase 1: a = dy [M, D] en e5m2 por S = sc[:, 0], pk = hpk = e4m3(rT²) (la h del
        forward: rT ≤ _H_CLAMP, rT² ≤ 238.3), sale dW2o [D, H] = dy^T · (rT/_SCALE_H)² con epílogo sc[:, 2] = 1/(S·_SCALE_H²),
        traspuesto en el PE por teselas de 128 x 128. a queda residente en SBUF (M/256 x 2 x D bytes por partición). Rejilla: cada
        programa hace H/nprg filas de H. Salida bf16 contigua. En mlp_fp8_bwd_act, dps y e4m3(rT²) se trasponen en el PE de dos en
        dos bytes (vista bf16; la trasposición del PE es exacta en NeuronCore-v3) y se desentrelazan al vaciar PSUM (con
        ARCH_MLP_DW8_EVSB=1, pasando antes por SBUF en bf16)."""
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
        """dW1o [H, D] del MLP en fp8 double_row (ARCH_MLP_DW8=1, _dw8_fase 0)."""
        return _dw8_fase(x, dpk, sc, 0)


    @_nki.jit
    def mlp_fp8_dw8_y(dy, hpk, sc):
        """dW2o [D, H] del MLP en fp8 double_row (ARCH_MLP_DW8=1, _dw8_fase 1)."""
        return _dw8_fase(dy, hpk, sc, 1)


    def _pow2_scale(w):
        """Escala potencia de dos que lleva amax(w) a ≤128 (rango seguro de e4m3 sin subnormales), como tensor 0-d fp32."""
        return torch.exp2(torch.floor(7.0 - torch.log2(w.detach().abs().float().amax().clamp_min(1e-12))))


    def fwd_scales(w1, w2):
        """[_P8, 4] fp32 para el kernel forward: (s1, s2, 1/(s2·SCALE_H²), _SCALE_H/s1): los pesos se escalan por s1/s2 antes de
        e4m3; la ReLU deshace s1 (rT = relu(_SCALE_H·pre) queda independiente de las escalas) y la salida deshace s2 y SCALE_H²."""
        s1, s2 = _pow2_scale(w1), _pow2_scale(w2)
        row = torch.stack((s1, s2, 1.0 / (s2 * _SCALE_H * _SCALE_H), _SCALE_H / s1)).view(1, 4)
        return row.expand(_P8, 4).contiguous().to(w1.device), s1, s2


    def bwd_scales(s1, s2):
        """[_P8, 4] fp32 para el kernel backward: (s2, s1, 2/(_SCALE_H·s2), 1/s1). Con ARCH_MLP_FP8_DYSCALE=k > 0, [_P8, 5]:
        (s2, s1, 2/(_SCALE_H·s2·S), 1/(s1·S), S) con S = 2^k, todo potencias de dos (exacto en fp32)."""
        if _S8G:
            row = torch.stack((s2, s1, 2.0 / (_SCALE_H * s2 * _S8G), 1.0 / (s1 * _S8G), torch.full_like(s1, _S8G))).view(1, 5)
        else:
            row = torch.stack((s2, s1, 2.0 / (_SCALE_H * s2), 1.0 / s1)).view(1, 4)
        return row.expand(_P8, row.shape[1]).contiguous()


    def mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1):
        """Pesos en su disposición original: w1o = c_fc.weight [H, D], w2o = c_proj.weight [D, H]. Devuelve dW en esas mismas
        disposiciones y CONTIGUOS (el all_reduce por parámetro de 4 rangos no admite vistas transpuestas: P99b/c/d)."""
        r = rT.t() * (1.0 / s1) if _FEP else rT.t() * (1.0 / _SCALE_H)
        h = r * r
        dpre = (dy @ w2o) * (2.0 * r)
        dx = dpre @ w1o
        dw1o = dpre.t() @ x2
        dw2o = dy.t() @ h
        return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)


    def dw8_scales(device):
        """[_P8, 3] fp32 para mlp_fp8_dw8_x/_y: (S, 1/S, 1/(S·_SCALE_H²)) con S = 2^ARCH_MLP_FP8_DYSCALE, potencias de dos."""
        row = torch.tensor([_S8G, 1.0 / _S8G, 1.0 / (_S8G * _SCALE_H * _SCALE_H)], dtype=torch.float32).view(1, 3)
        return row.expand(_P8, 3).contiguous().to(device)


    def mlp_fp8_backward_kernel(bwd, x2, w1o, w2o, rT, dy, s1, dwk=None):
        """bwd = kernel del gradiente de activación (dpreT, dx), que consume w2T = c_proj.weight [D, H] y w1T = c_fc.weight [H, D]
        tal cual; dW1o [H, D] y dW2o [D, H] en torch (contracción sobre tokens), contiguos. Con dwk (ARCH_MLP_DW8=1), bwd devuelve
        (dpk, hpk, dx) en fp8 por tokens y dwk = (mlp_fp8_dw8_x, mlp_fp8_dw8_y) calcula dW1o y dW2o en fp8 double_row."""
        s2 = _pow2_scale(w2o)
        if dwk is not None:
            dpk, hpk, dx = bwd(dy, w2o, w1o, rT, bwd_scales(s1, s2).to(dy.device))
            sd = dw8_scales(dy.device)
            dw1o, dw2o = dwk[0](x2, dpk, sd), dwk[1](dy, hpk, sd)
            return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)
        dpreT, dx = bwd(dy, w2o, w1o, rT, bwd_scales(s1, s2).to(dy.device))
        rTs = rT * (1.0 / s1) if _FEP else rT * (1.0 / _SCALE_H)
        dw1o = dpreT @ x2
        dw2o = dy.t() @ (rTs * rTs).t()
        return dx.to(x2.dtype), dw1o.to(w1o.dtype), dw2o.to(w2o.dtype)


    def build_op(lnc=1, simulate=False, bwd_kernel=False, eager=False):
        """Devuelve una función y = mlp_fp8(x2, w1o, w2o) diferenciable (pesos en la disposición original de nn.Linear). simulate=True: kernels en el simulador CPU (test).
        bwd_kernel=True: el gradiente de activación también en NKI fp8 (mlp_fp8_bwd_act); si no, backward entero en torch."""
        if simulate:
            fwd = _nki.simulate(mlp_fp8_fwd_r)
            bwd = _nki.simulate(mlp_fp8_bwd_act) if bwd_kernel else None
            dwk = (_nki.simulate(mlp_fp8_dw8_x), _nki.simulate(mlp_fp8_dw8_y)) if bwd_kernel and MLP_DW8 else None

            class _F(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x2, w1o, w2o):
                    sc, s1, s2 = fwd_scales(w1o, w2o)
                    y, rT = fwd(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)
                    ctx.save_for_backward(x2, w1o, w2o, rT, s1)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    if bwd is not None:
                        return mlp_fp8_backward_kernel(bwd, *ctx.saved_tensors[:4], dy, ctx.saved_tensors[4], dwk)
                    return mlp_fp8_backward_math(*ctx.saved_tensors[:4], dy, ctx.saved_tensors[4])

            return _F.apply
        from torch_neuronx import nki_op, wrap_nki
        k = wrap_nki(mlp_fp8_fwd_r)[lnc] if lnc > 1 else wrap_nki(mlp_fp8_fwd_r)
        kb = (wrap_nki(mlp_fp8_bwd_act)[lnc] if lnc > 1 else wrap_nki(mlp_fp8_bwd_act)) if bwd_kernel else None
        kd = tuple(wrap_nki(f)[lnc] if lnc > 1 else wrap_nki(f) for f in (mlp_fp8_dw8_x, mlp_fp8_dw8_y)) if bwd_kernel and MLP_DW8 else None
        if eager:
            class _FE(torch.autograd.Function):
                """Op fuera del grafo compilado (despacho eager de los kernels): evita el HOP dentro de torch.compile."""
                @staticmethod
                def forward(ctx, x2, w1o, w2o):
                    sc, s1, s2 = fwd_scales(w1o, w2o)
                    y, rT = k(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)
                    ctx.save_for_backward(x2, w1o, w2o, rT, s1)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    x2, w1o, w2o, rT, s1 = ctx.saved_tensors
                    if kb is not None:
                        return mlp_fp8_backward_kernel(kb, x2, w1o, w2o, rT, dy, s1, kd)
                    return mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1)

            return _FE.apply

        @nki_op("arch::mlp_fp8_fwd", mutates_args={})
        def _fwd_op(x2: torch.Tensor, w1o: torch.Tensor, w2o: torch.Tensor, sc: torch.Tensor, s1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return k(x2, w1o.t().contiguous(), w2o.t().contiguous(), sc)

        def _setup(ctx, inputs, output):
            x2, w1o, w2o, sc, s1 = inputs
            ctx.save_for_backward(x2, w1o, w2o, output[1], s1)
            ctx.mark_non_differentiable(output[1])

        def _backward(ctx, dy, drT):
            x2, w1o, w2o, rT, s1 = ctx.saved_tensors
            if kb is not None:
                dx, dw1o, dw2o = mlp_fp8_backward_kernel(kb, x2, w1o, w2o, rT, dy, s1, kd)
            else:
                dx, dw1o, dw2o = mlp_fp8_backward_math(x2, w1o, w2o, rT, dy, s1)
            return dx, dw1o, dw2o, None, None

        _fwd_op.register_autograd(_backward, setup_context=_setup)

        def f(x2, w1o, w2o):
            """x2 [M, D]; w1o = c_fc.weight [H, D]; w2o = c_proj.weight [D, H] (disposición original de nn.Linear)."""
            sc, s1, s2 = fwd_scales(w1o, w2o)
            return _fwd_op(x2, w1o, w2o, sc, s1)[0]
        return f

    _MLP_FP8_OP = None


    def _mlp_fp8_impl(x2, w1o, w2o):
        global _MLP_FP8_OP
        if _MLP_FP8_OP is None:
            _MLP_FP8_OP = build_op(lnc=MLP_FP8_LNC, bwd_kernel=MLP_FP8_BWDK, eager=MLP_FP8_EAGER)
        return _MLP_FP8_OP(x2, w1o, w2o)

    _mlp_fp8 = torch._dynamo.disable(_mlp_fp8_impl) if MLP_FP8_EAGER else _mlp_fp8_impl

if QKNORM_KERNEL and _HAS_NKI:
    import nki as _nki
    import nki.language as nl
    import nki.isa as nisa
    from torch_neuronx import nki_op as _nki_op, wrap_nki as _wrap_nki

    P, D, T = 128, 128, 1024
    HALF = D // 2


    def _rope_rows(y, xs, c, s, h):
        t1 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(t1, xs[:, :, :HALF], c[:, :, :HALF], nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, HALF:], s[:, :, :HALF], nl.multiply)
        nisa.tensor_tensor(y[:, :, :HALF], t1, t2, nl.subtract)
        nisa.tensor_tensor(t1, xs[:, :, HALF:], c[:, :, HALF:], nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, :HALF], s[:, :, HALF:], nl.multiply)
        nisa.tensor_tensor(y[:, :, HALF:], t1, t2, nl.add)


    def _rope_t_rows(dx, dy, c, s, h):
        t1 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(t1, dy[:, :, :HALF], c[:, :, :HALF], nl.multiply)
        nisa.tensor_tensor(t2, dy[:, :, HALF:], s[:, :, :HALF], nl.multiply)
        nisa.tensor_tensor(dx[:, :, :HALF], t1, t2, nl.add)
        nisa.tensor_tensor(t1, dy[:, :, HALF:], c[:, :, HALF:], nl.multiply)
        nisa.tensor_tensor(t2, dy[:, :, :HALF], s[:, :, HALF:], nl.multiply)
        nisa.tensor_tensor(dx[:, :, HALF:], t1, t2, nl.subtract)


    @_nki.jit
    def rope_norm_fwd_kernel(x, cos3, sin3, eps):
        N, H, _ = x.shape
        NT = T // P
        B = N // T
        out = nl.ndarray((B, H, T, D), dtype=x.dtype, buffer=nl.shared_hbm)
        rr = nl.ndarray((N, H), dtype=nl.float32, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            j = i % NT
            bb = i // NT
            c = nl.load(cos3[j * P:(j + 1) * P, :, :])
            s = nl.load(sin3[j * P:(j + 1) * P, :, :])
            x_sb = nl.load(x[i * P:(i + 1) * P, :, :])
            y = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            _rope_rows(y, x_sb, c, s, H)
            o_sb = nl.ndarray((P, H, D), dtype=x.dtype, buffer=nl.sbuf)
            r_sb = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
            for h in nl.static_range(H):
                y2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(y2, op=nl.square, data=y[:, h, :], reduce_op=nl.add, reduce_res=ss)
                ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(ms, ss, nl.multiply, 1.0 / D, op1=nl.add, operand1=eps)
                r = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(r, nl.rsqrt, ms)
                nisa.tensor_scalar(o_sb[:, h, :], y[:, h, :], nl.multiply, r)
                nisa.tensor_copy(r_sb[:, h:h + 1], r)
            for h in nl.static_range(H):
                nl.store(out[bb, h, j * P:(j + 1) * P, :], o_sb[:, h, :])
            nl.store(rr[i * P:(i + 1) * P, :], r_sb)
        return out, rr


    @_nki.jit
    def rope_norm_bwd_kernel(x, cos3, sin3, rr, g):
        N, H, _ = x.shape
        NT = T // P
        dx = nl.ndarray((N, H, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            j = i % NT
            bb = i // NT
            c = nl.load(cos3[j * P:(j + 1) * P, :, :])
            s = nl.load(sin3[j * P:(j + 1) * P, :, :])
            x_sb = nl.load(x[i * P:(i + 1) * P, :, :])
            g_sb = nl.ndarray((P, H, D), dtype=g.dtype, buffer=nl.sbuf)
            for h in nl.static_range(H):
                nisa.dma_copy(g_sb[:, h, :], g[bb, h, j * P:(j + 1) * P, :])
            r_sb = nl.load(rr[i * P:(i + 1) * P, :])
            y = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            _rope_rows(y, x_sb, c, s, H)
            gy = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(gy, g_sb, y, nl.multiply)
            dy = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            for h in nl.static_range(H):
                sgy = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(sgy, nl.add, gy[:, h, :], axis=1)
                r = r_sb[:, h:h + 1]
                r2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(r2, r, r, nl.multiply)
                coef = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(coef, sgy, r2, nl.multiply)
                nisa.tensor_scalar(coef, coef, nl.multiply, 1.0 / D)
                yc = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(yc, y[:, h, :], nl.multiply, coef)
                nisa.tensor_tensor(dy[:, h, :], g_sb[:, h, :], yc, nl.subtract)
                nisa.tensor_scalar(dy[:, h, :], dy[:, h, :], nl.multiply, r)
            dx_sb = nl.ndarray((P, H, D), dtype=x.dtype, buffer=nl.sbuf)
            _rope_t_rows(dx_sb, dy, c, s, H)
            nl.store(dx[i * P:(i + 1) * P, :, :], dx_sb)
        return dx

    _RN_FWD = _wrap_nki(rope_norm_fwd_kernel)
    _RN_BWD = _wrap_nki(rope_norm_bwd_kernel)
    _RN_EPS = float(torch.finfo(torch.bfloat16).eps)

    @_nki_op("arch::ropenorm_fwd", mutates_args={})
    def _rn_fwd_op(x: torch.Tensor, cos3: torch.Tensor, sin3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _RN_FWD(x, cos3, sin3, _RN_EPS)

    @_nki_op("arch::ropenorm_bwd", mutates_args={})
    def _rn_bwd_op(x: torch.Tensor, cos3: torch.Tensor, sin3: torch.Tensor, rr: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return _RN_BWD(x, cos3, sin3, rr, g)

    def _rn_setup(ctx, inputs, output):
        x, cos3, sin3 = inputs
        ctx.save_for_backward(x, cos3, sin3, output[1])
        ctx.mark_non_differentiable(output[1])

    def _rn_backward(ctx, g, grr):
        x, cos3, sin3, rr = ctx.saved_tensors
        return _rn_bwd_op(x, cos3, sin3, rr, g.contiguous()), None, None

    _rn_fwd_op.register_autograd(_rn_backward, setup_context=_rn_setup)

    def rope_norm(x3, cos3, sin3):
        out, _ = _rn_fwd_op(x3.contiguous(), cos3, sin3)
        return out


if EMB_SCATTER and _HAS_NKI:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _EK_P = 128


    @nki.jit
    def embk_dw_kernel(dy, idx, rows):
        n, d = dy.shape
        pid = nl.program_id(0)
        dh = d // nl.num_programs(0)
        c0 = pid * dh
        nch = (dh + 511) // 512
        cw = dh // nch
        tot = rows + _EK_P
        dw = nl.ndarray((tot, d), dtype=nl.float32, buffer=nl.shared_hbm)
        zero = nl.ndarray((_EK_P, dh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero, value=0.0)
        nfull = tot // _EK_P
        for r in nl.affine_range(nfull):
            nisa.dma_copy(dst=dw[r * _EK_P:(r + 1) * _EK_P, c0:c0 + dh], src=zero)
        if tot > nfull * _EK_P:
            nisa.dma_copy(dst=dw[nfull * _EK_P:tot, c0:c0 + dh], src=zero[0:tot - nfull * _EK_P, :])
        pos = nl.ndarray((_EK_P, _EK_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(pos, pattern=[[1, _EK_P]], offset=0, channel_multiplier=-1)
        low = nl.ndarray((_EK_P, _EK_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=low, data=pos, op0=nl.less, operand0=0.0)
        slot = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(slot, pattern=[[0, 1]], offset=rows, channel_multiplier=1)
        for t in nl.sequential_range(n // _EK_P):
            idc = nl.ndarray((_EK_P, 1), dtype=idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=idc, src=idx.ap(pattern=[[1, _EK_P], [1, 1]], offset=t * _EK_P))
            idr = nl.ndarray((_EK_P, _EK_P), dtype=idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=idr, src=idx.ap(pattern=[[0, _EK_P], [1, _EK_P]], offset=t * _EK_P))
            idcf = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idcf, src=idc)
            idrf = nl.ndarray((_EK_P, _EK_P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idrf, src=idr)
            eq = nl.ndarray((_EK_P, _EK_P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=idrf, op0=nl.equal, operand0=idcf)
            prev = nl.ndarray((_EK_P, _EK_P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prev, data1=eq, data2=low, op=nl.multiply)
            cnt = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cnt, op=nl.add, data=prev, axis=1)
            fol = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=fol, data=cnt, op0=nl.minimum, operand0=1.0)
            gap = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gap, data1=slot, data2=idcf, op=nl.subtract)
            pen = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pen, data1=fol, data2=gap, op=nl.multiply)
            effF = nl.ndarray((_EK_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=effF, data1=idcf, data2=pen, op=nl.add)
            eff = nl.ndarray((_EK_P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=eff, src=effF)
            eqm = nl.ndarray((_EK_P, _EK_P), dtype=dy.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=eqm, src=eq)
            for c in nl.static_range(nch):
                cc = c0 + c * cw
                dyt = nl.ndarray((_EK_P, cw), dtype=dy.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=dyt, src=dy[t * _EK_P:(t + 1) * _EK_P, cc:cc + cw])
                ps = nl.ndarray((_EK_P, cw), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=eqm, moving=dyt)
                acc = nl.ndarray((_EK_P, cw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=acc, src=ps)
                tgt = dw.ap(pattern=[[d, _EK_P], [1, cw]], offset=cc, vector_offset=eff, indirect_dim=0)
                nisa.dma_compute(dst=tgt, srcs=[tgt, acc], reduce_op=nl.add)
        return dw


    try:
        from torch_neuronx import nki_op as _embk_deco, wrap_nki as _embk_wrap
        _EMBK_RUN = _embk_wrap(embk_dw_kernel)[EMB_SCATTER_LNC] if EMB_SCATTER_LNC > 1 else _embk_wrap(embk_dw_kernel)

        @_embk_deco("arch::embk_dw", mutates_args={})
        def _embk_dw_op(dy: torch.Tensor, idx: torch.Tensor, rows: int) -> torch.Tensor:
            return _EMBK_RUN(dy, idx, rows)

        _EMBK_OP = _embk_dw_op
    except ImportError:
        _EMBK_OP = None


if ATTN_FUSED_BWD and not _HAS_NKI:
    raise RuntimeError("ARCH_ATTN_FUSED_BWD=1 necesita nki (chip o simulador del SDK)")
if ATTN_FUSED_BWD:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    P, D, T, HALF = 128, 128, 1024, 64
    NT, QB = T // P, 512
    VW = D + 32
    SCALE = D ** -0.5
    EPS = 2.0 ** -7
    C_SHIFT = 11.5


    def _rope_norm_t(xT, src, r0, c, s, t, eps, h):
        xs = nl.ndarray((P, h, D), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=src[r0:r0 + P, :])
        y = nl.ndarray((P, h, D), dtype=nl.float32, buffer=nl.sbuf)
        t1 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(t1, xs[:, :, :HALF], c, nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, HALF:], s, nl.multiply)
        nisa.tensor_tensor(y[:, :, :HALF], t1, t2, nl.add)
        nisa.tensor_tensor(t1, xs[:, :, HALF:], c, nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, :HALF], s, nl.multiply)
        nisa.tensor_tensor(y[:, :, HALF:], t1, t2, nl.subtract)
        ss = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.static_range(h):
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(sq, op=nl.square, data=y[:, i, :], reduce_op=nl.add, reduce_res=ss[:, i:i + 1])
        ms = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(ms, ss, nl.multiply, 1.0 / D, op1=nl.add, operand1=eps)
        r = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(r, nl.rsqrt, ms)
        xn = nl.ndarray((P, h, D), dtype=src.dtype, buffer=nl.sbuf)
        for i in nl.static_range(h):
            nisa.tensor_scalar(xn[:, i, :], y[:, i, :], nl.multiply, r[:, i:i + 1])
            pt = nl.ndarray((D, P), dtype=src.dtype, buffer=nl.psum)
            nisa.nc_transpose(pt, xn[:, i, :])
            nisa.tensor_copy(xT[:, i, t * P:(t + 1) * P], pt)


    def _fase_a(qp, kp, v, row, cs, sn, eps, h):
        qT = nl.ndarray((D, h, T), dtype=qp.dtype, buffer=nl.sbuf)
        kT = nl.ndarray((D, h, T), dtype=kp.dtype, buffer=nl.sbuf)
        vx = nl.ndarray((P, NT, h, VW), dtype=v.dtype, buffer=nl.sbuf)
        for t in nl.static_range(NT):
            r0 = row + t * P
            c, s = cs[:, t:t + 1, :].broadcast(1, h), sn[:, t:t + 1, :].broadcast(1, h)
            _rope_norm_t(qT, qp, r0, c, s, t, eps, h)
            _rope_norm_t(kT, kp, r0, c, s, t, eps, h)
            nisa.dma_copy(dst=vx[:, t, :, 0:D], src=v.ap(pattern=[[h * D, P], [D, h], [1, D]], offset=r0 * h * D))
            nisa.tensor_scalar(vx[:, t, :, D:D + 1], cs[:, t:t + 1, 0:1].broadcast(1, h), nl.multiply, 0.0, op1=nl.add, operand1=1.0)
        return qT, kT, vx


    def _attend_head(oa, li, qT, kT, vx, negc, h):
        for qb in nl.static_range(NT // 4):
            acc = []
            for ii in nl.static_range(4):
                acc.append(nl.ndarray((P, D + 1), dtype=nl.float32, buffer=nl.psum))
            for j in nl.static_range(4 * qb + 4):
                c0 = max(qb * QB, j * P)
                w = (qb + 1) * QB - c0
                ps = nl.ndarray((P, QB), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps[:, 0:w], kT[:, h, j * P:(j + 1) * P], qT[:, h, c0:c0 + w], accumulate=False)
                pt = nl.ndarray((P, QB), dtype=vx.dtype, buffer=nl.sbuf)
                nisa.activation(pt[:, 0:w], nl.exp, ps[:, 0:w], bias=negc, scale=SCALE)
                if j >= 4 * qb:
                    nisa.affine_select(pt[:, 0:P], pattern=[[1, P]], offset=0, channel_multiplier=-1, cmp_op=nl.greater_equal,
                                       on_true_tile=pt[:, 0:P], on_false_value=0.0)
                i0 = max(j, 4 * qb)
                for di in nl.static_range(4 * qb + 4 - i0):
                    i = i0 + di
                    nisa.nc_matmul(acc[i - 4 * qb], pt[:, i * P - c0:(i + 1) * P - c0], vx[:, j, h, 0:D + 1], accumulate=(j > 0))
            for ii in nl.static_range(4):
                i = 4 * qb + ii
                nisa.reciprocal(li[:, i, h:h + 1], acc[ii][:, D:D + 1])
                nisa.tensor_scalar(oa[:, i, h * D:(h + 1) * D], acc[ii][:, 0:D], nl.multiply, li[:, i, h:h + 1])


    def _fase_b(o, lse, qT, kT, vx, negc, row, h, C):
        oa = nl.ndarray((P, NT, h * D), dtype=qT.dtype, buffer=nl.sbuf)
        li = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        for hh in nl.static_range(h):
            _attend_head(oa, li, qT, kT, vx, negc, hh)
        lg = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(lg, nl.log, li)
        la = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(la, lg, nl.multiply, -1.0, op1=nl.add, operand1=C)
        for t in nl.static_range(NT):
            nl.store(o[row + t * P:row + (t + 1) * P, :], oa[:, t, :])
            nl.store(lse[row + t * P:row + (t + 1) * P, :], la[:, t, :])


    def _fila(pid, bs, b):
        return (pid * bs + b) * T


    @nki.jit
    def attn_fused_fwd(qp, kp, v, cos_h, sin_h, eps, C):
        N, W = qp.shape
        H = W // D
        B = N // T
        if nl.program_ndim() != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        assert N % T == 0 and W == H * D and H * D == 896 and B % nprg == 0
        assert kp.shape == qp.shape and v.shape == qp.shape and cos_h.shape == (T, HALF) and sin_h.shape == (T, HALF)
        assert qp.dtype == nl.bfloat16 and kp.dtype == nl.bfloat16 and v.dtype == nl.bfloat16
        assert cos_h.dtype == nl.float32 and sin_h.dtype == nl.float32
        BS = B // nprg
        o = nl.ndarray((N, W), dtype=qp.dtype, buffer=nl.shared_hbm)
        lse = nl.ndarray((N, H), dtype=nl.float32, buffer=nl.shared_hbm)
        cs = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        sn = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        for t in nl.static_range(NT):
            nisa.dma_copy(dst=cs[:, t, :], src=cos_h[t * P:(t + 1) * P, :])
            nisa.dma_copy(dst=sn[:, t, :], src=sin_h[t * P:(t + 1) * P, :])
        negc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(negc, cs[:, 0, 0:1], nl.multiply, 0.0, op1=nl.add, operand1=-C)
        qa, ka, va = _fase_a(qp, kp, v, _fila(pid, BS, 0), cs, sn, eps, H)
        for b0 in nl.static_range(BS):
            if b0 + 1 < BS:
                qn, kn, vn = _fase_a(qp, kp, v, _fila(pid, BS, b0 + 1), cs, sn, eps, H)
            _fase_b(o, lse, qa, ka, va, negc, _fila(pid, BS, b0), H, C)
            if b0 + 1 < BS:
                qa, ka, va = qn, kn, vn
        return o, lse


    def rope_halves(cos, sin):
        """Tablas del portador (cos/sin [1, T, 1, 128] bf16 con ROPE_MM, o [1, T, 1, 64]) → cos_h, sin_h [T, 64] fp32 contiguas."""
        return cos.reshape(cos.shape[1], -1)[:, :HALF].float().contiguous(), sin.reshape(sin.shape[1], -1)[:, :HALF].float().contiguous()


    P, D, T, HALF = 128, 128, 1024, 64
    NT, KB = T // P, 512
    SCALE = D ** -0.5


    def _bw_rope_norm(xn, xT, r, src, r0, c, s, t, eps, h):
        xs = nl.ndarray((P, h, D), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=src[r0:r0 + P, :])
        y = nl.ndarray((P, h, D), dtype=nl.float32, buffer=nl.sbuf)
        t1 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, h, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(t1, xs[:, :, :HALF], c, nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, HALF:], s, nl.multiply)
        nisa.tensor_tensor(y[:, :, :HALF], t1, t2, nl.add)
        nisa.tensor_tensor(t1, xs[:, :, HALF:], c, nl.multiply)
        nisa.tensor_tensor(t2, xs[:, :, :HALF], s, nl.multiply)
        nisa.tensor_tensor(y[:, :, HALF:], t1, t2, nl.subtract)
        ss = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.static_range(h):
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(sq, op=nl.square, data=y[:, i, :], reduce_op=nl.add, reduce_res=ss[:, i:i + 1])
        ms = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(ms, ss, nl.multiply, 1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(r[:, t, :], nl.rsqrt, ms)
        for i in nl.static_range(h):
            nisa.tensor_scalar(xn[:, t, i, :], y[:, i, :], nl.multiply, r[:, t, i:i + 1])
        _bw_transpone(xT, xn[:, t, :, :], t, h)


    def _bw_transpone(xT, x, t, h):
        for g in nl.static_range((h + 3) // 4):
            n = min(4, h - 4 * g)
            pt = nl.ndarray((D, 4, P), dtype=x.dtype, buffer=nl.psum)
            for i in nl.static_range(n):
                nisa.nc_transpose(pt[:, i, :], x[:, 4 * g + i, :])
            nisa.tensor_copy(xT[:, 4 * g:4 * g + n, t * P:(t + 1) * P], pt[:, 0:n, :])


    def _bw_fase_a(qp, kp, v, o, do, lse, row, cs, sn, eps, h):
        qn = nl.ndarray((P, NT, h, D), dtype=qp.dtype, buffer=nl.sbuf)
        kn = nl.ndarray((P, NT, h, D), dtype=kp.dtype, buffer=nl.sbuf)
        dn = nl.ndarray((P, NT, h, D), dtype=do.dtype, buffer=nl.sbuf)
        qT = nl.ndarray((D, h, T), dtype=qp.dtype, buffer=nl.sbuf)
        kT = nl.ndarray((D, h, T), dtype=kp.dtype, buffer=nl.sbuf)
        dT = nl.ndarray((D, h, T), dtype=do.dtype, buffer=nl.sbuf)
        vT = nl.ndarray((D, h, T), dtype=v.dtype, buffer=nl.sbuf)
        rq = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        rk = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        dv_ = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        nlse = nl.ndarray((P, NT, h), dtype=nl.float32, buffer=nl.sbuf)
        for t in nl.static_range(NT):
            r0 = row + t * P
            c, s = cs[:, t:t + 1, :].broadcast(1, h), sn[:, t:t + 1, :].broadcast(1, h)
            _bw_rope_norm(qn, qT, rq, qp, r0, c, s, t, eps, h)
            _bw_rope_norm(kn, kT, rk, kp, r0, c, s, t, eps, h)
            nisa.dma_copy(dst=dn[:, t, :, :], src=do[r0:r0 + P, :])
            vs = nl.ndarray((P, h, D), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=vs, src=v[r0:r0 + P, :])
            os_ = nl.ndarray((P, h, D), dtype=o.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=os_, src=o[r0:r0 + P, :])
            pr = nl.ndarray((P, h, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(pr, dn[:, t, :, :], os_, nl.multiply)
            nisa.tensor_reduce(dv_[:, t, :], nl.add, pr, axis=(2,))
            ls = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ls, src=lse[r0:r0 + P, :])
            nisa.tensor_scalar(nlse[:, t, :], ls, nl.multiply, -1.0)
            _bw_transpone(dT, dn[:, t, :, :], t, h)
            _bw_transpone(vT, vs, t, h)
        return qn, kn, dn, qT, kT, dT, vT, rq, rk, dv_, nlse


    def _bw_sdp(qT, kT, dT, vT, dv_, nlse, hh, i, k0, w):
        s_ps = nl.ndarray((P, KB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(s_ps[:, 0:w], qT[:, hh, i * P:(i + 1) * P], kT[:, hh, k0:k0 + w], accumulate=False)
        dp_ps = nl.ndarray((P, KB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dp_ps[:, 0:w], dT[:, hh, i * P:(i + 1) * P], vT[:, hh, k0:k0 + w], accumulate=False)
        pb = nl.ndarray((P, KB), dtype=qT.dtype, buffer=nl.sbuf)
        nisa.activation(pb[:, 0:w], nl.exp, s_ps[:, 0:w], bias=nlse[:, i, hh:hh + 1], scale=SCALE)
        if k0 + w == (i + 1) * P:
            nisa.affine_select(pb[:, w - P:w], pattern=[[-1, P]], offset=0, channel_multiplier=1, cmp_op=nl.greater_equal,
                               on_true_tile=pb[:, w - P:w], on_false_value=0.0)
        ds = nl.ndarray((P, KB), dtype=qT.dtype, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(ds[:, 0:w], data=dp_ps[:, 0:w], op0=nl.subtract, operand0=dv_[:, i, hh:hh + 1],
                                  op1=nl.multiply, operand1=pb[:, 0:w])
        return pb, ds


    def _bw_acum(dq, dva, dka, pb, ds, qn, kn, dn, hh, i, k0, w):
        n = w // P
        j0 = k0 // P
        nisa.nc_matmul(dva[k0 // KB][:, 0:w], dn[:, i, hh, :], pb[:, 0:w], accumulate=True)
        nisa.nc_matmul(dka[k0 // KB][:, 0:w], qn[:, i, hh, :], ds[:, 0:w], accumulate=True)
        pt = nl.ndarray((P, 4, P), dtype=ds.dtype, buffer=nl.psum)
        dsT = nl.ndarray((P, 4, P), dtype=ds.dtype, buffer=nl.sbuf)
        for jj in nl.static_range(n):
            j = j0 + jj
            nisa.nc_transpose(pt[:, jj, :], ds[:, jj * P:(jj + 1) * P])
            if jj > 0:
                nisa.nc_matmul(dq, dsT[:, jj - 1, :], kn[:, j - 1, hh, :], accumulate=(j - 1 > 0))
            nisa.tensor_copy(dsT[:, jj, :], pt[:, jj, :])
        nisa.nc_matmul(dq, dsT[:, n - 1, :], kn[:, j0 + n - 1, hh, :], accumulate=(j0 + n - 1 > 0))


    def _bw_abre():
        dva = []
        dka = []
        for g in nl.static_range(T // KB):
            a = nl.ndarray((D, KB), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(a, 0.0)
            dva.append(a)
            b = nl.ndarray((D, KB), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(b, 0.0)
            dka.append(b)
        g_q = nl.ndarray((P, NT, D), dtype=nl.float32, buffer=nl.sbuf)
        g_k = nl.ndarray((P, NT, D), dtype=nl.float32, buffer=nl.sbuf)
        return dva, dka, g_q, g_k


    def _bw_cierra(dv, row, dva, dka, g_k, hh, h):
        ov = nl.ndarray((P, NT, D), dtype=dv.dtype, buffer=nl.sbuf)
        for g in nl.static_range(T // KB):
            vb = nl.ndarray((D, KB), dtype=dv.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(vb, dva[g])
            kb_ = nl.ndarray((D, KB), dtype=dv.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(kb_, dka[g], nl.multiply, SCALE)
            pv = nl.ndarray((P, 4, D), dtype=dv.dtype, buffer=nl.psum)
            pk = nl.ndarray((P, 4, D), dtype=dv.dtype, buffer=nl.psum)
            for jj in nl.static_range(4):
                nisa.nc_transpose(pv[:, jj, :], vb[:, jj * P:(jj + 1) * P])
                nisa.nc_transpose(pk[:, jj, :], kb_[:, jj * P:(jj + 1) * P])
            nisa.tensor_copy(ov[:, 4 * g:4 * g + 4, :], pv)
            nisa.tensor_copy(g_k[:, 4 * g:4 * g + 4, :], pk)
        _bw_guarda(dv, row, ov, hh, h)


    def _bw_paso(st, pb, ds, ops, hh, i, k0, w, h):
        dva, dka, g_q, g_k, dq, p_q, p_k = st
        qn, kn, dn, rq, rk, cs, sn, dqp, dkp, dv, row = ops
        bi = i + max(0, i - 4) + k0 // KB
        if i == 0 and k0 == 0 and hh > 0:
            p_q, p_k = g_q, g_k
            dva, dka, g_q, g_k = _bw_abre()
        if k0 == 0 and (i > 0 or hh > 0):
            dq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        _bw_acum(dq, dva, dka, pb, ds, qn, kn, dn, hh, i, k0, w)
        if k0 + w == (i + 1) * P:
            nisa.tensor_scalar(g_q[:, i, :], dq, nl.multiply, SCALE)
        if i == NT - 1 and k0 + w == T:
            _bw_cierra(dv, row, dva, dka, g_k, hh, h)
        if hh > 0 and bi == 3:
            _bw_cola(dqp, row, p_q, qn, rq, cs, sn, hh - 1, h)
        if hh > 0 and bi == 7:
            _bw_cola(dkp, row, p_k, kn, rk, cs, sn, hh - 1, h)
        return dva, dka, g_q, g_k, dq, p_q, p_k


    def _bw_cola(dst, row, g, xn, r, cs, sn, hh, h):
        pr = nl.ndarray((P, NT, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(pr, g, xn[:, :, hh, :], nl.multiply)
        sm = nl.ndarray((P, NT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(sm, nl.add, pr, axis=(2,), keepdims=True)
        nisa.tensor_scalar(sm, sm, nl.multiply, 1.0 / D)
        nisa.tensor_tensor(pr, xn[:, :, hh, :], sm.broadcast(2, D), nl.multiply)
        nisa.tensor_tensor(pr, g, pr, nl.subtract)
        nisa.tensor_tensor(pr, pr, r[:, :, hh:hh + 1].broadcast(2, D), nl.multiply)
        a = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        out = nl.ndarray((P, NT, D), dtype=dst.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(a, pr[:, :, :HALF], cs, nl.multiply)
        nisa.tensor_tensor(b, pr[:, :, HALF:], sn, nl.multiply)
        nisa.tensor_tensor(out[:, :, :HALF], a, b, nl.subtract)
        nisa.tensor_tensor(a, pr[:, :, :HALF], sn, nl.multiply)
        nisa.tensor_tensor(b, pr[:, :, HALF:], cs, nl.multiply)
        nisa.tensor_tensor(out[:, :, HALF:], a, b, nl.add)
        _bw_guarda(dst, row, out, hh, h)


    def _bw_guarda(dst, row, out, hh, h):
        nisa.dma_copy(dst=dst.ap(pattern=[[h * D, P], [P * h * D, NT], [1, D]], offset=row * h * D + hh * D), src=out)


    def _bw_secuencia(dqp, dkp, dv, qp, kp, v, o, do, lse, row, cs, sn, eps, h):
        qn, kn, dn, qT, kT, dT, vT, rq, rk, dv_, nlse = _bw_fase_a(qp, kp, v, o, do, lse, row, cs, sn, eps, h)
        ops = (qn, kn, dn, rq, rk, cs, sn, dqp, dkp, dv, row)
        dva, dka, g_q, g_k = _bw_abre()
        dq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        st = (dva, dka, g_q, g_k, dq, g_q, g_k)
        c = 0
        for hh in nl.static_range(h):
            for i in nl.static_range(NT):
                for kb in nl.static_range((i * P) // KB + 1):
                    k0 = kb * KB
                    w = min(KB, (i + 1) * P - k0)
                    pb, ds = _bw_sdp(qT, kT, dT, vT, dv_, nlse, hh, i, k0, w)
                    if c > 0:
                        st = _bw_paso(st, a_pb, a_ds, ops, a_hh, a_i, a_k0, a_w, h)
                    a_pb, a_ds, a_hh, a_i, a_k0, a_w = pb, ds, hh, i, k0, w
                    c = c + 1
        st = _bw_paso(st, a_pb, a_ds, ops, a_hh, a_i, a_k0, a_w, h)
        _bw_cola(dqp, row, st[2], qn, rq, cs, sn, h - 1, h)
        _bw_cola(dkp, row, st[3], kn, rk, cs, sn, h - 1, h)


    def _bw_fila(pid, bs, b):
        return (pid * bs + b) * T


    @nki.jit
    def attn_fused_bwd(qp, kp, v, o, do, lse, cos_h, sin_h, eps):
        N, W = qp.shape
        H = W // D
        B = N // T
        if nl.program_ndim() != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        assert N % T == 0 and W == H * D and H * D == 896 and B % nprg == 0
        assert kp.shape == qp.shape and v.shape == qp.shape and o.shape == qp.shape and do.shape == qp.shape
        assert lse.shape == (N, H) and cos_h.shape == (T, HALF) and sin_h.shape == (T, HALF)
        assert qp.dtype == nl.bfloat16 and kp.dtype == nl.bfloat16 and v.dtype == nl.bfloat16
        assert o.dtype == nl.bfloat16 and do.dtype == nl.bfloat16 and lse.dtype == nl.float32
        assert cos_h.dtype == nl.float32 and sin_h.dtype == nl.float32
        BS = B // nprg
        dqp = nl.ndarray((N, W), dtype=qp.dtype, buffer=nl.shared_hbm)
        dkp = nl.ndarray((N, W), dtype=qp.dtype, buffer=nl.shared_hbm)
        dv = nl.ndarray((N, W), dtype=qp.dtype, buffer=nl.shared_hbm)
        cs = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        sn = nl.ndarray((P, NT, HALF), dtype=nl.float32, buffer=nl.sbuf)
        for t in nl.static_range(NT):
            nisa.dma_copy(dst=cs[:, t, :], src=cos_h[t * P:(t + 1) * P, :])
            nisa.dma_copy(dst=sn[:, t, :], src=sin_h[t * P:(t + 1) * P, :])
        for b0 in nl.static_range(BS):
            _bw_secuencia(dqp, dkp, dv, qp, kp, v, o, do, lse, _bw_fila(pid, BS, b0), cs, sn, eps, H)
        return dqp, dkp, dv


    _ATTFB_FWD_SHA256 = "1e44d9b9920ec1bba2dc32ce6546a8d689210145f4c31102a3f2bd2b891d4e97"
    _ATTFB_BWD_SHA256 = "de7926d252c07d511c558b968f7bc3613fe54fc4ab6a93656ef5612639d6bf86"
    assert not (NO_ROPE or NO_NORM or NO_QKNORM or NO_ATTN or FLASH or MANUAL_ATTN or QKV_FUSED or QKNORM_KERNEL or NORM_MANUAL == 6), \
        "ARCH_ATTN_FUSED_BWD=1 reemplaza RoPE + norma QK + atención: incompatible con esos toggles"
    _ATTFB_EPS = float(torch.finfo(torch.bfloat16).eps)
    _ATTFB_U = 2.0 ** -8
    try:
        from torch_neuronx import nki_op as _attfb_deco, wrap_nki as _attfb_wrap
        _ATTFB_FWD = _attfb_wrap(attn_fused_fwd)[ATTFB_LNC] if ATTFB_LNC > 1 else _attfb_wrap(attn_fused_fwd)
        _ATTFB_BWD = _attfb_wrap(attn_fused_bwd)[ATTFB_LNC] if ATTFB_LNC > 1 else _attfb_wrap(attn_fused_bwd)
        _ATTFB_SIMULADO = False
    except ImportError:
        _attfb_sf = nki.simulate(attn_fused_fwd[ATTFB_LNC] if ATTFB_LNC > 1 else attn_fused_fwd)
        _attfb_sb = nki.simulate(attn_fused_bwd[ATTFB_LNC] if ATTFB_LNC > 1 else attn_fused_bwd)
        _ATTFB_FWD = lambda *a: tuple(torch.as_tensor(r) for r in _attfb_sf(*a))
        _ATTFB_BWD = lambda *a: tuple(torch.as_tensor(r) for r in _attfb_sb(*a))
        _attfb_deco = lambda name, mutates_args: torch.library.custom_op(name, mutates_args=())
        _ATTFB_SIMULADO = True


    @_attfb_deco("arch::attfb_fwd", mutates_args={})
    def _attfb_fwd_op(qp: torch.Tensor, kp: torch.Tensor, v: torch.Tensor, cos_h: torch.Tensor, sin_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _ATTFB_FWD(qp, kp, v, cos_h, sin_h, _ATTFB_EPS, C_SHIFT)


    @_attfb_deco("arch::attfb_bwd", mutates_args={})
    def _attfb_bwd_op(qp: torch.Tensor, kp: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor,
                      cos_h: torch.Tensor, sin_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _ATTFB_BWD(qp, kp, v, o, do, lse, cos_h, sin_h, _ATTFB_EPS)


    if _ATTFB_SIMULADO:
        @_attfb_fwd_op.register_fake
        def _attfb_fake(qp, kp, v, cos_h, sin_h):
            return torch.empty_like(qp), qp.new_empty((qp.shape[0], qp.shape[1] // D), dtype=torch.float32)

        @_attfb_bwd_op.register_fake
        def _attfb_bwd_fake(qp, kp, v, o, do, lse, cos_h, sin_h):
            return torch.empty_like(qp), torch.empty_like(qp), torch.empty_like(qp)


    def _attfb_setup(ctx, inputs, output):
        qp, kp, v, cos_h, sin_h = inputs
        ctx.save_for_backward(qp, kp, v, cos_h, sin_h, output[0], output[1])
        ctx.mark_non_differentiable(output[1])


    def _attfb_backward(ctx, do, dlse):
        qp, kp, v, cos_h, sin_h, o, lse = ctx.saved_tensors
        if _ATTFB_SIMULADO:
            dq, dk, dv = _attfb_bwd_op(qp, kp, v, o, do.contiguous(), lse, cos_h, sin_h)
        else:
            dq, dk, dv = _ATTFB_BWD(qp, kp, v, o, do.contiguous(), lse, cos_h, sin_h, _ATTFB_EPS)
        return dq, dk, dv, None, None


    _attfb_fwd_op.register_autograd(_attfb_backward, setup_context=_attfb_setup)


    def attfb_rope_halves(cos, sin):
        return cos.reshape(cos.shape[1], -1)[:, :HALF].float().contiguous(), sin.reshape(sin.shape[1], -1)[:, :HALF].float().contiguous()


    def attfb_grad64(qp, kp, v, do, cos_h, sin_h):
        c, s = cos_h.double().view(1, T, 1, HALF), sin_h.double().view(1, T, 1, HALF)
        m = torch.ones(T, T, dtype=torch.bool).tril()
        out = {n: [] for n in ("dqp", "dkp", "dv")}
        for i in range(qp.shape[0] // T):
            sl = slice(i * T, (i + 1) * T)
            hojas = [x[sl].detach().double().view(1, T, -1, D).requires_grad_() for x in (qp, kp, v)]
            qk = []
            for x in hojas[:2]:
                y1, y2 = x[..., :HALF], x[..., HALF:]
                y = torch.cat((y1 * c + y2 * s, y2 * c - y1 * s), -1)
                qk.append(y * torch.rsqrt((y * y).mean(-1, keepdim=True) + _ATTFB_EPS))
            sc = (torch.einsum("bthd,bshd->bhts", qk[0], qk[1]) * D ** -0.5).masked_fill(~m, float("-inf"))
            o = torch.einsum("bhts,bshd->bthd", torch.softmax(sc, -1), hojas[2])
            (o.reshape(T, -1) * do[sl].double()).sum().backward()
            for n, x in zip(out, hojas):
                out[n].append(x.grad.reshape(T, -1))
        return {n: torch.cat(x) for n, x in out.items()}


    def attfb_juzga(g, e):
        gd, ed = g.detach().cpu().double(), e.double()
        r = {"fin": bool(torch.isfinite(gd).all()), "frob": ((gd - ed).norm() / ed.norm()).item(),
             "max": ((gd - ed).abs().max() / ed.abs().max()).item(), "escala": ((gd * ed).sum() / (ed * ed).sum()).item() - 1.0}
        r["ok"] = r["fin"] and r["frob"] <= 4 * _ATTFB_U and r["max"] <= 8 * _ATTFB_U and abs(r["escala"]) <= 2.5e-3
        return r


    def _attfb_puerta_llamada(qp, kp, v, cos_h, sin_h):
        return _attfb_fwd_op(qp, kp, v, cos_h, sin_h)[0]


    def attfb_puerta_dispositivo(device, cos, sin, n_seq=16, semilla=0):
        g = torch.Generator().manual_seed(semilla)
        qp, kp, v, do = (torch.randn(n_seq * T, 7 * D, generator=g).bfloat16() for _ in range(4))
        cos_h, sin_h = attfb_rope_halves(cos.detach().cpu(), sin.detach().cpu())
        hojas = [x.clone().to(device).requires_grad_() for x in (qp, kp, v)]
        f = torch.compile(_attfb_puerta_llamada, backend="neuron", dynamic=False) if device.type == "neuron" else _attfb_puerta_llamada
        o = f(*hojas, cos_h.to(device), sin_h.to(device))
        o.backward(do.to(device))
        ex = attfb_grad64(qp, kp, v, do, cos_h, sin_h)
        res = {n: attfb_juzga(x.grad, ex[n]) for n, x in zip(ex, hojas)}
        res["ok"] = all(r["ok"] for r in res.values())
        return res


if ATTN_DW8 and not _HAS_NKI:
    raise RuntimeError("ARCH_ATTN_DW8=1 necesita nki (chip o simulador del SDK)")
if ATTN_DW8:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _AD_P, _AD_NT = 128, 512
    _AD_S = float(2 ** ATTN_DW8_K)
    _AD_ACT = ATTN_DW8_ACT


    @nki.jit
    def attn_dw8_kernel(g, a, sc):
        """dW [DO, DI] = g^T · a de una proyección de atención en fp8 double_row (ARCH_ATTN_DW8=1; generaliza _dw8_fase). g [N, DO] es el
        gradiente de la salida y a [N, DI] la entrada, bf16 en HBM. Cada trozo s de 256 tokens se lee con el token 256·s + 2·p + r en la
        partición p y la fila r de double_row y se convierte en SBUF: g a e5m2 por S = sc[:, 0] (2^ARCH_ATTN_DW8_K; en el motor escalar
        con _AD_ACT), a a e4m3 saturada a ±240. Estacionario = g por teselas de 128 filas de DO, móvil = a: los tokens se contraen en la partición y ningún operando se
        traspone. Cada tesela de filas acumula en su banco de PSUM (fp32, DO/128 <= 8 bancos) a lo largo de los N/256 trozos y el epílogo
        aplica sc[:, 1] = 1/S y guarda bf16 contiguo en la disposición del peso [DO, DI]. Rejilla: cada programa hace DI/nprg columnas
        de a (448 de 896 con LNC=2), en pasadas de <= 512 columnas móviles; sin rejilla, dos pasadas de 448."""
        N, DO = g.shape
        DI = a.shape[1]
        assert a.shape[0] == N and N % (2 * _AD_P) == 0 and DO % _AD_P == 0 and DO // _AD_P <= 8
        out = nl.ndarray((DO, DI), dtype=g.dtype, buffer=nl.shared_hbm)
        scs = nl.load(sc)
        if nl.program_ndim() != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        NS = N // (2 * _AD_P)
        KO = DO // _AD_P
        W = DI // nprg
        NC = (W + _AD_NT - 1) // _AD_NT
        WC = W // NC
        assert W * nprg == DI and WC * NC == W
        for c in nl.static_range(NC):
            c0 = pid * W + c * WC
            ps = nl.ndarray((_AD_P, KO, _AD_NT), dtype=nl.float32, buffer=nl.psum)
            for s in nl.static_range(NS):
                gt = nl.ndarray((_AD_P, 2, DO), dtype=g.dtype, buffer=nl.sbuf)
                nisa.dma_copy(gt, g.ap(pattern=[[2 * DO, _AD_P], [DO, 2], [1, DO]], offset=s * 2 * _AD_P * DO))
                g8 = nl.ndarray((_AD_P, 2, DO), dtype=nl.float8_e5m2, buffer=nl.sbuf)
                if _AD_ACT:
                    nisa.tensor_scalar(g8, gt, nl.multiply, scs[:, 0:1], engine=nisa.engine.scalar)
                else:
                    nisa.tensor_scalar(g8, gt, nl.multiply, scs[:, 0:1])
                at = nl.ndarray((_AD_P, 2, WC), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(at, a.ap(pattern=[[2 * DI, _AD_P], [DI, 2], [1, WC]], offset=s * 2 * _AD_P * DI + c0))
                a8 = nl.ndarray((_AD_P, 2, WC), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                nisa.tensor_scalar(a8, at, nl.minimum, 240.0, op1=nl.maximum, operand1=-240.0)
                for ko in nl.static_range(KO):
                    nisa.nc_matmul(ps[:, ko, 0:WC], g8[:, :, ko * _AD_P:(ko + 1) * _AD_P], a8, accumulate=(s > 0),
                                   perf_mode=nisa.matmul_perf_mode.double_row)
            for ko in nl.static_range(KO):
                o = nl.ndarray((_AD_P, WC), dtype=g.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(o, ps[:, ko, 0:WC], nl.multiply, scs[:, 1:2])
                nl.store(out[ko * _AD_P:(ko + 1) * _AD_P, c0:c0 + WC], o)
        return out


    try:
        from torch_neuronx import nki_op as _adw8_deco, wrap_nki as _adw8_wrap
        _ADW8_RUN = _adw8_wrap(attn_dw8_kernel)[ATTN_DW8_LNC] if ATTN_DW8_LNC > 1 else _adw8_wrap(attn_dw8_kernel)
        _ADW8_SIMULADO = False
    except ImportError:
        _adw8_sim = nki.simulate(attn_dw8_kernel[ATTN_DW8_LNC] if ATTN_DW8_LNC > 1 else attn_dw8_kernel)
        _ADW8_RUN = lambda g, a, sc: torch.as_tensor(_adw8_sim(g.clone(), a.clone(), sc.clone()))
        _adw8_deco = lambda name, mutates_args: torch.library.custom_op(name, mutates_args=())
        _ADW8_SIMULADO = True


    @_adw8_deco("arch::adw8_dw", mutates_args={})
    def _adw8_dw_op(g: torch.Tensor, a: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        return _ADW8_RUN(g, a, sc)


    if _ADW8_SIMULADO:
        @_adw8_dw_op.register_fake
        def _adw8_dw_fake(g, a, sc):
            return g.new_empty((g.shape[1], a.shape[1]))


    def adw8_escalas(device):
        """[_AD_P, 2] fp32 para attn_dw8_kernel: (S, 1/S) con S = 2^ARCH_ATTN_DW8_K, potencias de dos (exactas)."""
        return torch.tensor([_AD_S, 1.0 / _AD_S], dtype=torch.float32).view(1, 2).expand(_AD_P, 2).contiguous().to(device)


    def adw8_dw(g2, a2):
        """dW [DO, DI] bf16 = g2^T · a2 por arch::adw8_dw (g2 [N, DO] y a2 [N, DI] bf16 contiguos en filas de tokens)."""
        return _adw8_dw_op(g2.contiguous(), a2.contiguous(), adw8_escalas(g2.device))


    class _ADW8Lin(torch.autograd.Function):
        """ARCH_ATTN_DW8=1: forward F.linear(x, W) (el GEMM bf16 de Linear.forward) y en el backward dX = g · W (lo que calcula autograd para
        F.linear); solo dW = g^T · x sale de arch::adw8_dw en fp8 double_row, contiguo y con el dtype del peso."""
        @staticmethod
        def forward(ctx, x, w):
            wc = w.to(dtype=x.dtype)
            ctx.save_for_backward(x, wc)
            ctx.wdtype = w.dtype
            return F.linear(x, wc)

        @staticmethod
        def backward(ctx, g):
            x, wc = ctx.saved_tensors
            g2 = g.reshape(-1, g.shape[-1])
            dx = g2.mm(wc).view(x.shape)
            dw = adw8_dw(g2, x.reshape(-1, x.shape[-1]))
            return dx, dw.to(ctx.wdtype)


_CF_P, _CF_TW, _CF_CAP = 128, 2048, LOGIT_SOFTCAP


def _cef_fwd_ref(z):
    """Emulación en torch de arch::cef_fwd: t = bf16(tanh(z/15)), lse = log(sum exp(15 t)) en fp32, forma [N, 1]."""
    t = torch.tanh(z.float() * (1.0 / _CF_CAP)).bfloat16()
    return t, torch.log(torch.exp(_CF_CAP * t.float()).sum(-1, keepdim=True))


def _cef_bwd_ref(t, nlse, g):
    """Emulación en torch de arch::cef_bwd: dz = bf16((g - g t^2) exp(15 t - lse)); nlse = -lse y g en fp32 [N, 1]."""
    tf = t.float()
    return (((tf * -g) * tf + g) * torch.exp(_CF_CAP * tf + nlse)).bfloat16()


class _CEFRef(torch.autograd.Function):
    """ARCH_CE_FUSED=1 fuera del dispositivo: lse [N] fp32 de z [N, V] con los kernels emulados."""
    @staticmethod
    def forward(ctx, z):
        t, lse = _cef_fwd_ref(z)
        ctx.save_for_backward(t, lse)
        ctx.zdtype = z.dtype
        return lse.squeeze(1)

    @staticmethod
    def backward(ctx, g):
        t, lse = ctx.saved_tensors
        return _cef_bwd_ref(t, -lse, g.float().reshape(-1, 1)).to(ctx.zdtype)


_CEF_OP = None
_CEF_BWD_OP = None
if CE_FUSED and _HAS_NKI:
    import nki
    import nki.language as nl
    import nki.isa as nisa


    @nki.jit
    def cef_fwd_kernel(z):
        n, v = z.shape
        t = nl.ndarray((n, v), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        lse = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        grid_ndim = nl.program_ndim()
        if grid_ndim != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        nb = (n // _CF_P) // nprg
        nv = v // _CF_TW
        for b0 in nl.affine_range(nb):
            r = (pid * nb + b0) * _CF_P
            parts = nl.ndarray((_CF_P, nv), dtype=nl.float32, buffer=nl.sbuf)
            for j in nl.static_range(nv):
                zt = nl.load(z[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW])
                tt = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=tt, op=nl.tanh, data=zt, scale=1.0 / _CF_CAP)
                nl.store(t[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW], tt)
                ex = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation_reduce(dst=ex, op=nl.exp, data=tt, reduce_op=nl.add, reduce_res=parts[:, j:j + 1], scale=_CF_CAP)
            s = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s, op=nl.add, data=parts, axis=1)
            ls = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ls, op=nl.log, data=s)
            nl.store(lse[r:r + _CF_P, 0:1], ls)
        return t, lse


    @nki.jit
    def cef_bwd_kernel(t, nlse, g):
        n, v = t.shape
        dz = nl.ndarray((n, v), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        grid_ndim = nl.program_ndim()
        if grid_ndim != 0:
            nprg = nl.num_programs(axes=0)
            pid = nl.program_id(axis=0)
        else:
            nprg = 1
            pid = 0
        nb = (n // _CF_P) // nprg
        nv = v // _CF_TW
        for b0 in nl.affine_range(nb):
            r = (pid * nb + b0) * _CF_P
            bl = nl.load(nlse[r:r + _CF_P, 0:1])
            gg = nl.load(g[r:r + _CF_P, 0:1])
            ng = nl.ndarray((_CF_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ng, data=gg, op0=nl.multiply, operand0=-1.0)
            for j in nl.static_range(nv):
                tt = nl.load(t[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW])
                p = nl.ndarray((_CF_P, _CF_TW), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p, op=nl.exp, data=tt, bias=bl, scale=_CF_CAP)
                w = nl.ndarray((_CF_P, _CF_TW), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=w, data=tt, op0=nl.multiply, operand0=ng, op1=nl.multiply, operand1=tt)
                d = nl.ndarray((_CF_P, _CF_TW), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=d, data=w, op0=nl.add, operand0=gg, op1=nl.multiply, operand1=p)
                nl.store(dz[r:r + _CF_P, j * _CF_TW:(j + 1) * _CF_TW], d)
        return dz


    try:
        from torch_neuronx import nki_op as _cef_deco, wrap_nki as _cef_wrap
        _CEF_FWD = _cef_wrap(cef_fwd_kernel)[CE_FUSED_LNC] if CE_FUSED_LNC > 1 else _cef_wrap(cef_fwd_kernel)
        _CEF_BWD = _cef_wrap(cef_bwd_kernel)[CE_FUSED_LNC] if CE_FUSED_LNC > 1 else _cef_wrap(cef_bwd_kernel)

        @_cef_deco("arch::cef_fwd", mutates_args={})
        def _cef_fwd_op(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return _CEF_FWD(z)

        @_cef_deco("arch::cef_bwd", mutates_args={})
        def _cef_bwd_op(t: torch.Tensor, nlse: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return _CEF_BWD(t, nlse, g)

        def _cef_setup(ctx, inputs, output):
            ctx.save_for_backward(output[0], output[1])
            ctx.mark_non_differentiable(output[0])

        def _cef_backward(ctx, dt, dlse):
            t, lse = ctx.saved_tensors
            return _cef_bwd_op(t, (-lse).contiguous(), dlse.float().contiguous())

        _cef_fwd_op.register_autograd(_cef_backward, setup_context=_cef_setup)
        _CEF_OP = _cef_fwd_op
        _CEF_BWD_OP = _cef_bwd_op
    except ImportError:
        _CEF_OP = None


def _cef_lse(z):
    if z.device.type == "cpu":
        return _CEFRef.apply(z)
    assert _CEF_OP is not None, "ARCH_CE_FUSED=1 en el dispositivo necesita nki y torch_neuronx"
    assert z.shape[0] % (_CF_P * CE_FUSED_LNC) == 0 and z.shape[1] % _CF_TW == 0, z.shape
    return _CEF_OP(z.contiguous())[1].squeeze(1)


def _cef_filas(idx, w):
    """Filas W[idx] [N, D]; con ARCH_EMB_SCATTER=1 el backward es el mismo kernel EMBK (y la misma forma) que el de wte."""
    if EMB_SCATTER:
        return _EmbScatter.apply(idx, w)
    return _EmbChunked.apply(idx, w) if EMB_CHUNKS > 0 else F.embedding(idx, w)


def _cef_loss(model, x, targets, aux_alpha):
    """ARCH_CE_FUSED=1: misma pérdida que la rama STATIC_CE con MTP compartido (aux_alpha) o sin él (aux_alpha None):
    nll_k = lse - 15 tanh(x·W[y_k]/15), con lse de _cef_lse y el término del objetivo por filas reunidas de W."""
    _v = model.config.vocab_size
    _w = model.lm_head.weight
    _xf = x.reshape(-1, x.size(-1))
    _lse = _cef_lse(F.linear(_xf, _w[:_v].to(dtype=_xf.dtype), None))
    _t = targets.reshape(-1)
    _valid = (_t >= 0).float()
    _ids = [_t.clamp_min(0)]
    if aux_alpha is not None:
        _t2 = torch.cat((targets[:, 1:], torch.full_like(targets[:, :1], -1)), dim=1).reshape(-1)
        _m = ((_t2 >= 0) & (_t != BOS_ID) & (_t2 != BOS_ID)).float() * _valid
        _ids.append(_t2.clamp_min(0))
    _zt = torch.stack([(_xf * _cef_filas(_i, _w).to(dtype=_xf.dtype)).float().sum(-1) for _i in _ids], dim=1)
    _nll = _lse.unsqueeze(1) - LOGIT_SOFTCAP * torch.tanh(_zt / LOGIT_SOFTCAP)
    if aux_alpha is None:
        return (_nll[:, 0] * _valid).sum() / _valid.sum()
    _a = aux_alpha.to(_nll.dtype) * _m
    return ((_nll[:, 0] + _a * _nll[:, 1]) / (1.0 + _a) * _valid).sum() / _valid.sum()


def _cef_puerta_llamada(z, g):
    if z.device.type == "cpu":
        t, lse = _cef_fwd_ref(z)
        return t, lse, _cef_bwd_ref(t, -lse, g)
    t, lse = _CEF_OP(z)
    return t, lse, _CEF_BWD_OP(t, (-lse).contiguous(), g)


def cef_puerta_dispositivo(device, n=512):
    """Regla de la puerta CEF: kernels frente a la emulación en CPU (ver ARCH_CE_FUSED_GATE)."""
    gen = torch.Generator().manual_seed(0)
    z = (torch.randn(n, 8192, generator=gen) * 4.0).bfloat16()
    g = torch.rand(n, 1, generator=gen) * 1e-3
    g[3] = 0.0
    te, le = _cef_fwd_ref(z)
    de = _cef_bwd_ref(te, -le, g).double()
    f = torch.compile(_cef_puerta_llamada, backend="neuron", dynamic=False) if device.type == "neuron" else _cef_puerta_llamada
    with torch.no_grad():
        tk, lk, dk = (a.detach().cpu() for a in f(z.to(device), g.to(device)))
    r = {"t_abs": (tk.float() - te.float()).abs().max().item(), "lse_rel": ((lk.double() - le.double()).abs() / le.double().abs()).max().item(),
         "dz_rel": ((dk.double() - de).norm() / de.norm()).item(), "fin": bool(torch.isfinite(dk.float()).all() and torch.isfinite(lk).all()),
         "fila_cero": bool((dk[3] == 0).all())}
    r["ok"] = r["fin"] and r["fila_cero"] and r["t_abs"] <= 2.0 ** -7 and r["lse_rel"] <= 1e-3 and r["dz_rel"] <= 2e-2
    return r


def cef_linea(r):
    return (f"CEF_GATE ok={r['ok']} t_abs={r['t_abs']:.3e} lse_rel={r['lse_rel']:.3e} dz_rel={r['dz_rel']:.3e} fin={r['fin']} "
            f"fila_cero={r['fila_cero']} sr={os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '')}")


_RESNORM_OPS = None
_RESNORM_CPU = False
if RESNORM_FUSED and _HAS_NKI:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _RN_P = 128


    def _rn_rstd(h_t, cf, w, kk, d):
        sq = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
        ss = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.static_range(kk):
            nisa.activation(dst=sq[:, j * d:(j + 1) * d], op=nl.square, data=h_t[:, j * d:(j + 1) * d], reduce_op=nl.add, reduce_res=ss[:, j:j + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rs = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / d, bias=cf[:, 2:3])
        return rs


    def _rn_dh(gh_t, gn_t, h_t, cf, w, kk, d, dt):
        rs = _rn_rstd(h_t, cf, w, kk, d)
        pr = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pr, data1=gn_t, data2=h_t, op=nl.multiply)
        sq = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
        dot = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.static_range(kk):
            nisa.activation(dst=sq[:, j * d:(j + 1) * d], op=nl.copy, data=pr[:, j * d:(j + 1) * d], reduce_op=nl.add, reduce_res=dot[:, j:j + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        r2 = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r2, data1=rs, data2=rs, op=nl.multiply)
        r3 = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r3, data1=r2, data2=rs, op=nl.multiply)
        ng = nl.ndarray((_RN_P, kk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=ng, data=r3, op0=nl.multiply, operand0=-1.0 / d, op1=nl.multiply, operand1=dot)
        t1 = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
        dh = nl.ndarray((_RN_P, w), dtype=dt, buffer=nl.sbuf)
        for j in nl.static_range(kk):
            nisa.scalar_tensor_tensor(dst=t1[:, j * d:(j + 1) * d], data=gn_t[:, j * d:(j + 1) * d], op0=nl.multiply, operand0=rs[:, j:j + 1], op1=nl.add, operand1=gh_t[:, j * d:(j + 1) * d])
            nisa.scalar_tensor_tensor(dst=dh[:, j * d:(j + 1) * d], data=h_t[:, j * d:(j + 1) * d], op0=nl.multiply, operand0=ng[:, j:j + 1], op1=nl.add, operand1=t1[:, j * d:(j + 1) * d])
        return dh


    @nki.jit
    def resnorm_fwd_kernel(x, y, coef, d):
        rows, w = x.shape
        kk = w // d
        pid = nl.program_id(0)
        rp = rows // nl.num_programs(0)
        h = nl.ndarray((rows, w), dtype=x.dtype, buffer=nl.shared_hbm)
        n = nl.ndarray((rows, w), dtype=x.dtype, buffer=nl.shared_hbm)
        cf = nl.ndarray((_RN_P, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cf, src=coef)
        for t in nl.affine_range(rp // _RN_P):
            r0 = pid * rp + t * _RN_P
            xt = nl.ndarray((_RN_P, w), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[r0:r0 + _RN_P, :])
            yt = nl.ndarray((_RN_P, w), dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y[r0:r0 + _RN_P, :])
            ax = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ax, op=nl.copy, data=xt, scale=cf[:, 0:1])
            hb = nl.ndarray((_RN_P, w), dtype=x.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=hb, data=yt, op0=nl.multiply, operand0=cf[:, 1:2], op1=nl.add, operand1=ax)
            rs = _rn_rstd(hb, cf, w, kk, d)
            nb = nl.ndarray((_RN_P, w), dtype=x.dtype, buffer=nl.sbuf)
            for j in nl.static_range(kk):
                nisa.tensor_scalar(dst=nb[:, j * d:(j + 1) * d], data=hb[:, j * d:(j + 1) * d], op0=nl.multiply, operand0=rs[:, j:j + 1])
            nisa.dma_copy(dst=h[r0:r0 + _RN_P, :], src=hb)
            nisa.dma_copy(dst=n[r0:r0 + _RN_P, :], src=nb)
        return h, n


    @nki.jit
    def resnorm_bwd_kernel(gh, gn, h, x, y, coef, d):
        rows, w = h.shape
        kk = w // d
        pid = nl.program_id(0)
        rp = rows // nl.num_programs(0)
        dx = nl.ndarray((rows, w), dtype=h.dtype, buffer=nl.shared_hbm)
        dy = nl.ndarray((rows, w), dtype=h.dtype, buffer=nl.shared_hbm)
        part = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.shared_hbm)
        cf = nl.ndarray((_RN_P, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cf, src=coef)
        for t in nl.affine_range(rp // _RN_P):
            r0 = pid * rp + t * _RN_P
            ght = nl.ndarray((_RN_P, w), dtype=gh.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ght, src=gh[r0:r0 + _RN_P, :])
            gnt = nl.ndarray((_RN_P, w), dtype=gn.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gnt, src=gn[r0:r0 + _RN_P, :])
            ht = nl.ndarray((_RN_P, w), dtype=h.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ht, src=h[r0:r0 + _RN_P, :])
            xt = nl.ndarray((_RN_P, w), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[r0:r0 + _RN_P, :])
            yt = nl.ndarray((_RN_P, w), dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y[r0:r0 + _RN_P, :])
            dh = _rn_dh(ght, gnt, ht, cf, w, kk, d, nl.float32)
            dxt = nl.ndarray((_RN_P, w), dtype=h.dtype, buffer=nl.sbuf)
            nisa.activation(dst=dxt, op=nl.copy, data=dh, scale=cf[:, 0:1])
            dyt = nl.ndarray((_RN_P, w), dtype=h.dtype, buffer=nl.sbuf)
            nisa.activation(dst=dyt, op=nl.copy, data=dh, scale=cf[:, 1:2])
            pp = nl.ndarray((_RN_P, 2), dtype=nl.float32, buffer=nl.sbuf)
            px = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=px, data1=dh, data2=xt, op=nl.multiply)
            nisa.tensor_reduce(dst=pp[:, 0:1], op=nl.add, data=px, axis=1)
            py = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=py, data1=dh, data2=yt, op=nl.multiply)
            sy = nl.ndarray((_RN_P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sy, op=nl.copy, data=py, reduce_op=nl.add, reduce_res=pp[:, 1:2], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.dma_copy(dst=dx[r0:r0 + _RN_P, :], src=dxt)
            nisa.dma_copy(dst=dy[r0:r0 + _RN_P, :], src=dyt)
            nisa.dma_copy(dst=part[r0:r0 + _RN_P, :], src=pp)
        return dx, dy, part


    @nki.jit
    def addnorm_bwd_kernel(gh, gn, h, coef, d):
        rows, w = h.shape
        kk = w // d
        pid = nl.program_id(0)
        rp = rows // nl.num_programs(0)
        dx = nl.ndarray((rows, w), dtype=h.dtype, buffer=nl.shared_hbm)
        cf = nl.ndarray((_RN_P, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cf, src=coef)
        for t in nl.affine_range(rp // _RN_P):
            r0 = pid * rp + t * _RN_P
            ght = nl.ndarray((_RN_P, w), dtype=gh.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ght, src=gh[r0:r0 + _RN_P, :])
            gnt = nl.ndarray((_RN_P, w), dtype=gn.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gnt, src=gn[r0:r0 + _RN_P, :])
            ht = nl.ndarray((_RN_P, w), dtype=h.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ht, src=h[r0:r0 + _RN_P, :])
            dt = _rn_dh(ght, gnt, ht, cf, w, kk, d, h.dtype)
            nisa.dma_copy(dst=dx[r0:r0 + _RN_P, :], src=dt)
        return dx


    def _rn_geom(x, lnc):
        d = x.shape[-1]
        n = x.numel() // d
        assert n % (_RN_P * lnc) == 0, (n, lnc)
        k = 2 if n % (2 * _RN_P * lnc) == 0 else 1
        return n // k, k * d, d


    def _rn_coef(a, b, x):
        e = torch.full((1,), torch.finfo(x.dtype).eps, dtype=torch.float32, device=x.device)
        return torch.cat((a.float().view(1), b.float().view(1), e)).view(1, 3).expand(_RN_P, 3).contiguous()


    def resnorm_build(lnc=2, run=None):
        """Devuelve (resnorm, addnorm). resnorm(x, y, a, b) -> (h, n): h = a·x + b·y (fp32, un solo redondeo al dtype de x) y
        n = h·rsqrt(mean(h²) + eps) con eps = finfo(dtype).eps como _rms_manual; gradientes de x, y, a y b en un kernel.
        addnorm(x, y) -> (h, n) con a = b = 1 constantes (sin gradiente de escalares). run(kernel, lnc) sustituye a wrap_nki
        (simulador en CPU); sin run, nki_op + register_autograd para torch.compile, rejilla lnc."""
        if run is None:
            from torch_neuronx import nki_op, wrap_nki
            kf, kb, ku = ((wrap_nki(_k)[lnc] if lnc > 1 else wrap_nki(_k)) for _k in (resnorm_fwd_kernel, resnorm_bwd_kernel, addnorm_bwd_kernel))
        else:
            kf, kb, ku = (run(_k, lnc) for _k in (resnorm_fwd_kernel, resnorm_bwd_kernel, addnorm_bwd_kernel))

        def _bwd_gen(x, y, coef, h, gh, gn, d):
            gh = torch.zeros_like(h) if gh is None else gh.contiguous()
            gn = torch.zeros_like(h) if gn is None else gn.contiguous()
            dx, dy, part = kb(gh, gn, h, x, y, coef, d)
            return dx, dy, torch.cat((part.view(-1, _RN_P, 2).sum(0), torch.zeros_like(coef[:, 2:3])), dim=1)

        def _bwd_unit(coef, h, gh, gn, d):
            gh = torch.zeros_like(h) if gh is None else gh.contiguous()
            gn = torch.zeros_like(h) if gn is None else gn.contiguous()
            return ku(gh, gn, h, coef, d)

        if run is None:
            @nki_op("arch::resnorm_fwd", mutates_args={})
            def _rn_op(x: torch.Tensor, y: torch.Tensor, coef: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
                return kf(x, y, coef, d)

            @nki_op("arch::addnorm_fwd", mutates_args={})
            def _an_op(x: torch.Tensor, y: torch.Tensor, coef: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
                return kf(x, y, coef, d)

            def _rn_setup(ctx, inputs, output):
                ctx.save_for_backward(inputs[0], inputs[1], inputs[2], output[0])
                ctx.d = inputs[3]

            def _rn_backward(ctx, gh, gn):
                x, y, coef, h = ctx.saved_tensors
                return (*_bwd_gen(x, y, coef, h, gh, gn, ctx.d), None)

            def _an_setup(ctx, inputs, output):
                ctx.save_for_backward(inputs[2], output[0])
                ctx.d = inputs[3]

            def _an_backward(ctx, gh, gn):
                coef, h = ctx.saved_tensors
                dh = _bwd_unit(coef, h, gh, gn, ctx.d)
                return dh, dh, None, None

            _rn_op.register_autograd(_rn_backward, setup_context=_rn_setup)
            _an_op.register_autograd(_an_backward, setup_context=_an_setup)
            fr, fa = _rn_op, _an_op
        else:
            class _RN(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y, coef, d):
                    h, n = kf(x, y, coef, d)
                    ctx.save_for_backward(x, y, coef, h)
                    ctx.d = d
                    return h, n

                @staticmethod
                def backward(ctx, gh, gn):
                    x, y, coef, h = ctx.saved_tensors
                    return (*_bwd_gen(x, y, coef, h, gh, gn, ctx.d), None)

            class _AN(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y, coef, d):
                    h, n = kf(x, y, coef, d)
                    ctx.save_for_backward(coef, h)
                    ctx.d = d
                    return h, n

                @staticmethod
                def backward(ctx, gh, gn):
                    coef, h = ctx.saved_tensors
                    dh = _bwd_unit(coef, h, gh, gn, ctx.d)
                    return dh, dh, None, None

            fr, fa = _RN.apply, _AN.apply

        def resnorm(x, y, a, b):
            rows, w, d = _rn_geom(x, lnc)
            h, n = fr(x.contiguous().view(rows, w), y.to(x.dtype).contiguous().view(rows, w), _rn_coef(a, b, x), d)
            return h.view(x.shape), n.view(x.shape)

        def addnorm(x, y):
            rows, w, d = _rn_geom(x, lnc)
            one = torch.ones(1, dtype=torch.float32, device=x.device)
            h, n = fa(x.contiguous().view(rows, w), y.to(x.dtype).contiguous().view(rows, w), _rn_coef(one, one, x), d)
            return h.view(x.shape), n.view(x.shape)

        return resnorm, addnorm


    try:
        _RESNORM_OPS = resnorm_build(lnc=RESNORM_LNC)
    except ImportError:
        _RESNORM_OPS = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = -1):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attn_width = config.attn_width or config.n_embd
        self.n_head = self.attn_width // self.head_dim
        assert self.head_dim * self.n_head == self.attn_width
        self.c_q = Linear(config.n_embd, self.attn_width, bias=False)
        self.c_k = Linear(config.n_embd, self.attn_width, bias=False)
        self.c_v = Linear(config.n_embd, self.attn_width, bias=False)
        self.c_proj = Linear(self.attn_width, config.n_embd, bias=False)
        if ROPE_MM:
            self.register_buffer("rot", _rot_matrix(self.head_dim).to(COMPUTE_DTYPE), persistent=False)
        else:
            self.rot = None
        self.attn_tau = nn.Parameter(torch.zeros(self.n_head, dtype=torch.float32)) if config.attn_tau else None
        self.xsa_alpha = nn.Parameter(torch.zeros(self.n_head, dtype=torch.float32)) if config.xsa else None
        self.head_gate = None
        if config.head_gate:
            _st = torch.get_rng_state()
            self.head_gate = Linear(config.n_embd, self.n_head, bias=False)
            torch.set_rng_state(_st)
        self.cfv = False
        self.doc_mask = False
        self.lga = str(layer_idx) in [s for s in config.lga_layers.split(",") if s]
        if self.lga:
            _st = torch.get_rng_state()
            _lw = config.lga_heads * self.head_dim
            self.c_q = Linear(config.n_embd, _lw, bias=False)
            self.c_k = Linear(config.n_embd, _lw, bias=False)
            self.c_v = Linear(config.n_embd, _lw, bias=False)
            self.c_proj = Linear(_lw, config.n_embd, bias=False)
            self.lga_gate = Linear(config.n_embd, config.n_embd, bias=False) if config.lga_gate else None
            self.lga_out = Linear(config.n_embd, config.n_embd, bias=False)
            self.attn_width, self.n_head = _lw, config.lga_heads
            torch.set_rng_state(_st)
        if NORM_MANUAL == 6:
            self.register_buffer("mblock", torch.block_diag(*([torch.full((self.head_dim, self.head_dim), 1.0 / self.head_dim)] * self.n_head)).to(COMPUTE_DTYPE), persistent=False)

    def _attfb_activa(self, x, t, kmask):
        capa = str(getattr(self, "layer_idx", -1))
        return (self.training and t == T and x.shape[0] % ATTFB_LNC == 0 and x.dtype == torch.bfloat16 and (x.device.type != "cpu" or ATTFB_SIM) and self.head_dim == D
                and self.attn_tau is None and self.xsa_alpha is None and self.head_gate is None and not self.lga
                and not self.doc_mask and not getattr(self, "hier_window", 0) and not (kmask is not None and capa in KEY_SHIFT_SET)
                and not (ATTN_WINDOW and capa in ATTN_WINDOW_LAYERS.split(",")))

    def _attfb_forward(self, x, cos, sin, ve, x0, b, t):
        n, w = b * t, self.attn_width
        if ATTN_DW8:
            return self._adw8_forward(x, cos, sin, ve, x0, b, t)
        qp = self.c_q(x).view(n, w)
        kp = self.c_k(x).view(n, w)
        v = self.c_v(x0.to(x.dtype) if self.cfv else x)
        if ve is not None:
            v = v + ve.view(b, t, w)
        cos_h, sin_h = attfb_rope_halves(cos, sin)
        o, _ = _attfb_fwd_op(qp, kp, v.view(n, w), cos_h, sin_h)
        return self.c_proj(o.view(b, t, w))

    def _adw8_forward(self, x, cos, sin, ve, x0, b, t):
        """_attfb_forward con ARCH_ATTN_DW8=1: las cuatro proyecciones por _ADW8Lin (dW en fp8 double_row; forward y dX sin cambios)."""
        n, w = b * t, self.attn_width
        qp = _ADW8Lin.apply(x, self.c_q.weight).view(n, w)
        kp = _ADW8Lin.apply(x, self.c_k.weight).view(n, w)
        v = _ADW8Lin.apply(x0.to(x.dtype) if self.cfv else x, self.c_v.weight)
        if ve is not None:
            v = v + ve.view(b, t, w)
        cos_h, sin_h = attfb_rope_halves(cos, sin)
        o, _ = _attfb_fwd_op(qp, kp, v.view(n, w), cos_h, sin_h)
        return _ADW8Lin.apply(o.view(b, t, w), self.c_proj.weight)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve=None, kmask=None, x0=None) -> torch.Tensor:
        b, t, c = x.shape
        if ATTN_FUSED_BWD and self._attfb_activa(x, t, kmask):
            return self._attfb_forward(x, cos, sin, ve, x0, b, t)
        if QKV_FUSED and self.cfv is False:
            _w = torch.cat((self.c_q.weight.t(), self.c_k.weight.t(), self.c_v.weight.t()), dim=1)
            _qkv = torch.matmul(x, _w.to(x.dtype))
            q, k, _vfused = _qkv.split(self.attn_width, dim=-1)
        else:
            _vfused = None
            q = self.c_q(x)
            k = self.c_k(x)
        if NORM_MANUAL == 6:
            q = _rms_block(q, self.mblock)
            k = _rms_block(k, self.mblock)
        _rn = QKNORM_KERNEL and q.device.type != "cpu" and q.dtype == torch.bfloat16 and t == 1024
        if _rn:
            _c2 = cos.reshape(t, -1).float()
            _s2 = sin.reshape(t, -1).float()
            if _c2.shape[-1] != self.head_dim:
                _c2 = torch.cat((_c2, _c2), dim=-1)
                _s2 = torch.cat((_s2, _s2), dim=-1)
            _c3 = _c2[:, None, :].expand(t, self.n_head, self.head_dim).contiguous()
            _s3 = _s2[:, None, :].expand(t, self.n_head, self.head_dim).contiguous()
            q = rope_norm(q.reshape(b * t, self.n_head, self.head_dim), _c3, _s3)
            k = rope_norm(k.reshape(b * t, self.n_head, self.head_dim), _c3, _s3)
        else:
            q = q.view(b, t, self.n_head, self.head_dim)
            k = k.view(b, t, self.n_head, self.head_dim)
        v = (_vfused if _vfused is not None else self.c_v(x0.to(x.dtype) if self.cfv else x)).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            v = v + ve.view(b, t, self.n_head, self.head_dim)
        if not _rn:
            q = qknorm(apply_rotary_emb(q, cos, sin, self.rot))
            k = qknorm(apply_rotary_emb(k, cos, sin, self.rot))
        if kmask is not None and str(getattr(self, "layer_idx", -1)) in KEY_SHIFT_SET:
            k = _key_shift(k, kmask)
        if not _rn:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        if self.attn_tau is not None:
            q = q * self.attn_tau.float().exp().to(q.dtype).view(1, -1, 1, 1)
        v = v.transpose(1, 2)
        if NO_ATTN:
            y = v
        elif FLASH:
            y = _flash_attention(q, k, v, ATTN_WINDOW if (ATTN_WINDOW and str(getattr(self, "layer_idx", -1)) in ATTN_WINDOW_LAYERS.split(",")) else 0)
        elif MANUAL_ATTN:
            y = torch.utils.checkpoint.checkpoint(_manual_attention, q, k, v, use_reentrant=False) if MANUAL_ATTN_CKPT else _manual_attention(q, k, v)
        elif getattr(self, "hier_window", 0):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=_band_mask(t, self.hier_window, q.device, q.dtype))
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask) if self.doc_mask else F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.xsa_alpha is not None:
            _yf, _vf = y.float(), v.float()
            _M = _mean_matrix(self.head_dim, y.device, torch.float32)
            _r = torch.rsqrt(torch.matmul(_vf * _vf, _M) + 1e-6)
            _coef = torch.matmul(_yf * _vf, _M) * _r * _r
            y = (_yf - torch.tanh(self.xsa_alpha.float()).view(1, -1, 1, 1) * _coef * _vf).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(b, t, self.attn_width)
        if self.head_gate is not None:
            _hg = (2.0 * torch.sigmoid(self.head_gate(x).float())).to(y.dtype)
            y = (y.view(b, t, self.n_head, self.head_dim) * _hg.unsqueeze(-1)).view(b, t, self.attn_width)
        if NF_ATTN:
            y = y * _nf_scale(y)
        if self.lga:
            if self.lga_gate is None:
                return self.lga_out(self.c_proj(y))
            _g = (2.0 * torch.sigmoid(self.lga_gate(x).float())).to(y.dtype)
            return self.lga_out(self.c_proj(y) * _g)
        return self.c_proj(y)


def _fq_round(x, mbits):
    drop = 7 - mbits
    i = x.to(torch.bfloat16).view(torch.int16)
    return ((i + (1 << (drop - 1))) & -(1 << drop)).view(torch.bfloat16).float()


def _fq(x, mbits):
    """fp8 emulado por tensor en punto flotante (RNE con log2/exp2/round; sin aritmética de bits, que el backend no compila
    bien): e4m3 legacy (3 bits, min normal 2^-6, subnormales a paso 2^-9, máx 240) o e5m2 (2 bits, 2^-14, 2^-16, 57344);
    escala potencia de dos por amax, rejilla subnormal y saturación."""
    emin, emax, largest = (-6, 7, 240.0) if mbits == 3 else (-14, 15, 57344.0)
    xf = x.float()
    k = torch.floor(emax - torch.log2(xf.detach().abs().amax().clamp_min(1e-30)))
    y = xf * torch.exp2(k)
    e = torch.floor(torch.log2(y.abs().clamp_min(2.0 ** emin)))
    q = (torch.round(y * torch.exp2(mbits - e)) * torch.exp2(e - mbits)).clamp(-largest, largest)
    return (q * torch.exp2(-k)).to(x.dtype)


class _FQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, mode):
        qx, qw = _fq(x, 3), _fq(weight, 3)
        ctx.save_for_backward(qx, qw)
        ctx.mode = mode
        return F.linear(qx, qw)

    @staticmethod
    def backward(ctx, dy):
        qx, qw = ctx.saved_tensors
        dyq = _fq(dy, 2) if ctx.mode >= 2 else dy
        dx = dyq @ qw
        dw = dyq.reshape(-1, dyq.shape[-1]).t() @ qx.reshape(-1, qx.shape[-1])
        return dx.to(dy.dtype), dw.to(qw.dtype), None


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, gated: bool = False, ctx_in: int = 0):
        super().__init__()
        self.gated = gated
        self.ctx_in = ctx_in
        if gated:
            inner = (2 * config.mlp_mult * config.n_embd) // 3
            self.c_fc = Linear(config.n_embd, 2 * inner, bias=False)
            self.c_proj = Linear(inner, config.n_embd, bias=False)
        else:
            self.c_fc = Linear(config.n_embd + ctx_in, config.mlp_mult * config.n_embd, bias=False)
            self.c_proj = Linear(config.mlp_mult * config.n_embd, config.n_embd, bias=False)
            if MLP_HIDDEN:
                assert not (REUSE_RANK or config.mlp_expand), "ARCH_MLP_HIDDEN no se combina con ARCH_REUSE_RANK ni ARCH_MLP_EXPAND"
                self.c_fc = Linear(config.n_embd + ctx_in, MLP_HIDDEN, bias=False)
                self.c_proj = Linear(MLP_HIDDEN, config.n_embd, bias=False)
        if DW_SAMPLE > 0 and not gated:
            self.c_fc.dw_sample = self.c_proj.dw_sample = True
            self.c_fc.dw_id, self.c_proj.dw_id = _dw_next(), _dw_next()

    def reuse_init(self, config, anchor):
        self.reuse_anchor = anchor
        if anchor:
            return
        _h, _d, _r = config.mlp_mult * config.n_embd, config.n_embd, REUSE_RANK
        del self.c_fc, self.c_proj
        _st = torch.get_rng_state()
        self.up_in = Linear(_d, _r, bias=False)
        self.up_out = Linear(_r, _h, bias=False)
        self.down_in = Linear(_h, _r, bias=False)
        self.down_out = Linear(_r, _d, bias=False)
        self.beta_up = nn.Parameter(torch.full((1,), REUSE_B0))
        self.beta_down = nn.Parameter(torch.full((1,), REUSE_B0))
        torch.set_rng_state(_st)

    def reuse_forward(self, x, prev):
        if self.reuse_anchor:
            pre = self.c_fc(x)
            return pre, self.c_proj(relu_squared(pre))
        _bu = torch.zeros_like(self.beta_up) if REUSE_BETA0 else self.beta_up
        _bd = torch.zeros_like(self.beta_down) if REUSE_BETA0 else self.beta_down
        pre = _bu.to(x.dtype) * prev[0] + self.up_out(self.up_in(x))
        out = _bd.to(x.dtype) * prev[1] + self.down_out(self.down_in(relu_squared(pre)))
        return pre, out

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        if self.gated:
            g, u = self.c_fc(x).chunk(2, dim=-1)
            return self.c_proj(F.silu(g) * u)
        if ctx is not None:
            x = torch.cat((x, ctx), dim=-1)
        if NF_FFN:
            return _mlp_nf(self, x)
        if MLP_FP8 and _HAS_NKI and self.training and x.device.type == "neuron":
            return _mlp_fp8(x.reshape(-1, x.shape[-1]).contiguous(), self.c_fc.weight, self.c_proj.weight).view(x.shape)
        if FP8_SIM:
            return _FQLinear.apply(relu_squared(_FQLinear.apply(x, self.c_fc.weight, FP8_SIM)), self.c_proj.weight, FP8_SIM)
        if MLP_SAVED_RELU == 2 and self.training and not getattr(self, "ctx_in", 0):
            return _MLPReluRecalculada.apply(x, self.c_fc.weight, self.c_proj.weight)
        if MLP_SAVED_RELU and self.training and not getattr(self, "ctx_in", 0):
            return _MLPReluGuardada.apply(x, self.c_fc.weight, self.c_proj.weight)
        if DW_SAMPLE > 0 and DW_SAMPLE_MODE == "split" and self.training and not getattr(self, "ctx_in", 0):
            _k = max(1, int(x.shape[0] * DW_SAMPLE))
            if _k < x.shape[0]:
                _w1, _w2 = self.c_fc.weight.to(dtype=x.dtype), self.c_proj.weight.to(dtype=x.dtype)
                _y1 = F.linear(relu_squared(F.linear(x[:_k], _w1, None)), _w2, None)
                _y2 = F.linear(relu_squared(F.linear(x[_k:], _w1.detach(), None)), _w2.detach(), None)
                return torch.cat((_y1, _y2), dim=0)
        return self.c_proj(relu_squared(self.c_fc(x)))


def _cumsum_blocks(m: torch.Tensor) -> torch.Tensor:
    N, E = m.shape
    B = 128
    if N % B:
        return torch.cumsum(m, dim=0)
    tri = torch.ones(B, B, dtype=m.dtype, device=m.device).tril()
    blocks = m.view(N // B, B, E)
    within = torch.matmul(tri, blocks)
    sums = blocks.sum(dim=1)
    nb = N // B
    tri_b = torch.ones(nb, nb, dtype=m.dtype, device=m.device).tril(-1)
    offset = torch.matmul(tri_b, sums)
    return (within + offset.unsqueeze(1)).reshape(N, E)


class MoEMLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.E, self.K, self.cap = config.moe_experts, config.moe_k, config.moe_cap
        hidden = config.moe_hidden or (config.mlp_mult * config.n_embd * config.moe_k // config.moe_experts)
        self.router = Linear(config.n_embd, self.E, bias=False)
        self.c_fc = nn.ModuleList([Linear(config.n_embd, hidden, bias=False) for _ in range(self.E)])
        self.c_proj = nn.ModuleList([Linear(hidden, config.n_embd, bias=False) for _ in range(self.E)])
        self.aux = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        N, E, K = b * t, self.E, self.K
        xf = x.reshape(N, d)
        logits = F.linear(xf.float(), self.router.weight.float())
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(logits, K, dim=-1)
        gates = probs.gather(1, topi)
        if K > 1:
            gates = gates / gates.sum(dim=-1, keepdim=True)
        member = torch.zeros(N, E, dtype=torch.float32, device=x.device)
        gate_ne = torch.zeros(N, E, dtype=torch.float32, device=x.device)
        for j in range(K):
            member = member + F.one_hot(topi[:, j], E).to(torch.float32)
            gate_ne = gate_ne + F.one_hot(topi[:, j], E).to(torch.float32) * gates[:, j:j + 1]
        C = int(math.ceil(t * K / E * self.cap))
        member3 = member.view(b, t, E)
        pos = torch.cumsum(member3, dim=1) * member3
        slot = (pos * (pos <= C).to(pos.dtype)).to(torch.long)
        src = torch.arange(N, device=x.device).view(b, 1, t).expand(b, E, t)
        if MOE_DISPATCH == "mm":
            cols = torch.arange(1, C + 1, device=x.device).view(1, 1, 1, C)
            D = (slot.unsqueeze(-1) == cols).to(x.dtype).reshape(b, t, E * C)
            xin = torch.bmm(D.transpose(1, 2), x).view(b, E, C, d).permute(1, 0, 2, 3).reshape(E, b * C, d)
            if MOE_LOOP:
                out = torch.stack([self.c_proj[e](relu_squared(self.c_fc[e](xin[e]))) for e in range(E)])
            else:
                w1 = torch.stack([m.weight.to(xin.dtype) for m in self.c_fc])
                w2 = torch.stack([m.weight.to(xin.dtype) for m in self.c_proj])
                out = torch.bmm(relu_squared(torch.bmm(xin, w1.transpose(1, 2))), w2.transpose(1, 2))
            G = (D.view(b, t, E, C) * gate_ne.view(b, t, E, 1).to(x.dtype)).reshape(b, t, E * C)
            y = torch.bmm(G, out.view(E, b, C, d).permute(1, 0, 2, 3).reshape(b, E * C, d))
            self.aux = E * (member.mean(dim=0) / K * probs.mean(dim=0)).sum()
            return y.view(b, t, d)
        idx = torch.zeros(b, E, C + 1, dtype=torch.long, device=x.device).scatter(2, slot.permute(0, 2, 1), src)[:, :, 1:]
        cnt = member3.sum(dim=1)
        valid = (torch.arange(C, device=x.device).view(1, 1, C) < cnt.unsqueeze(-1)).to(torch.float32)
        idx_e = idx.permute(1, 0, 2).reshape(E, b * C)
        valid_e = valid.permute(1, 0, 2).reshape(E, b * C)
        g_sel = gate_ne.t().gather(1, idx_e) * valid_e
        xin = xf[idx_e]
        if MOE_LOOP:
            out = torch.stack([self.c_proj[e](relu_squared(self.c_fc[e](xin[e]))) for e in range(E)])
        else:
            w1 = torch.stack([m.weight.to(xin.dtype) for m in self.c_fc])
            w2 = torch.stack([m.weight.to(xin.dtype) for m in self.c_proj])
            out = torch.bmm(relu_squared(torch.bmm(xin, w1.transpose(1, 2))), w2.transpose(1, 2))
        out = out * g_sel.unsqueeze(-1).to(x.dtype)
        if MOE_SCATTER:
            y = torch.zeros(N, d, dtype=out.dtype, device=x.device).index_add(0, idx_e.reshape(-1), out.reshape(E * b * C, d))
        else:
            slot_sel = slot.view(N, E).gather(1, topi)
            row_b = torch.div(torch.arange(N, device=x.device), t, rounding_mode="floor").unsqueeze(1)
            loc = topi * (b * C) + row_b * C + (slot_sel - 1).clamp_min(0)
            keep = (slot_sel > 0).to(out.dtype).unsqueeze(-1)
            y = (out.reshape(E * b * C, d)[loc.reshape(-1)].view(N, K, d) * keep).sum(dim=1)
        self.aux = E * (member.mean(dim=0) / K * probs.mean(dim=0)).sum()
        return y.view(b, t, d)


def _shift_causal(u: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat((torch.zeros_like(u[:, :k]), u[:, :-k]), dim=1)


def _mlp_expand_list(config):
    if not config.mlp_expand:
        return [config.mlp_mult] * config.n_layer
    v = [int(x) for x in config.mlp_expand.split(",") if x != ""]
    presupuesto = config.mlp_budget or config.mlp_mult * config.n_layer
    if len(v) != config.n_layer or any(x < 1 for x in v) or sum(v) != presupuesto:
        raise ValueError(f"ARCH_MLP_EXPAND: hacen falta {config.n_layer} enteros positivos con suma {presupuesto}")
    return v


def _mlp_cfg(config, layer_idx):
    _e = _mlp_expand_list(config)
    return config if _e[layer_idx] == config.mlp_mult else dataclasses.replace(config, mlp_mult=_e[layer_idx])


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.ve_lambda = nn.Parameter(torch.ones(1)) if str(layer_idx) in [s for s in config.ve_layers.split(",") if s] else None
        self.attn = CausalSelfAttention(config, layer_idx) if (not config.attn_layers or str(layer_idx) in config.attn_layers.split(",")) else None
        if self.attn is not None:
            self.attn.layer_idx = layer_idx
        self.ctx_proj = Linear(config.n_embd, config.ctx_rank, bias=False) if (config.ctx_rank and self.attn is not None) else None
        self.ctx_source = config.ctx_source
        self.cfv = str(layer_idx) in [s for s in config.cfv_layers.split(",") if s]
        _fir = str(layer_idx) in [s for s in config.fir_layers.split(",") if s]
        self.fir_a = nn.Parameter(torch.zeros(config.n_embd, dtype=torch.float32)) if _fir else None
        self.fir_b = nn.Parameter(torch.zeros(config.n_embd, dtype=torch.float32)) if _fir else None
        if self.attn is not None:
            self.attn.cfv = self.cfv
            self.attn.doc_mask = config.doc_mask
        _ctx_in = config.ctx_rank if (config.ctx_rank and self.attn is None) else 0
        self.mlp = MoEMLP(config) if str(layer_idx) in [s for s in config.moe_layers.split(",") if s] else MLP(_mlp_cfg(config, layer_idx), gated=str(layer_idx) in [s for s in config.mlp_gated.split(",") if s], ctx_in=_ctx_in)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.ts_router = _TwoStreamRouter(config.n_embd) if TWOSTREAM else None
        if REUSE_RANK:
            self.mlp.reuse_init(config, layer_idx == 0)

    def _forward_resnorm(self, x, x0, cos, sin, ve, kmask):
        h, u = _RESNORM_OPS[0](x, x0, self.resid_lambda, self.x0_lambda)
        if self.attn is not None:
            _a = self.attn(u, cos, sin, None if (ve is None or self.ve_lambda is None) else self.ve_lambda.to(ve.dtype) * ve, kmask, x0)
            h, u = _RESNORM_OPS[1](h, _a)
        return h + self.mlp(u)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve=None, kmask=None, ctx=None, hist=None):
        if RESNORM_FUSED and self.training and _RESNORM_OPS is not None and (RESNORM_FUSED == 1 or self.attn is None) and (x.device.type == "neuron" or _RESNORM_CPU) and not REUSE_RANK and not TWOSTREAM and not NO_NORM and self.fir_a is None and self.ctx_proj is None and not getattr(self.mlp, "ctx_in", 0):
            return self._forward_resnorm(x, x0, cos, sin, ve, kmask), ctx
        x = self.resid_lambda * x + self.x0_lambda * x0
        if self.attn is not None:
            _a = self.attn(norm(x), cos, sin, None if (ve is None or self.ve_lambda is None) else self.ve_lambda.to(ve.dtype) * ve, kmask, x0)
            x = x + _a
            if self.ctx_proj is not None:
                ctx = norm(self.ctx_proj(x if self.ctx_source == "resid" else _a))
        if REUSE_RANK:
            hist = self.mlp.reuse_forward(norm(x), hist)
            return x + hist[1], ctx, hist
        if self.fir_a is not None:
            _u = norm(x)
            _fa = (0.25 * torch.tanh(self.fir_a)).to(_u.dtype)
            _fb = (0.25 * torch.tanh(self.fir_b)).to(_u.dtype)
            _u = _u + _fa * _shift_causal(_u, 1) + _fb * _shift_causal(_u, 2)
            x = x + self.mlp(_u)
            return x, ctx
        if getattr(self.mlp, "ctx_in", 0):
            x = x + self.mlp(norm(x), ctx)
            ctx = None
        else:
            x = x + self.mlp(norm(x))
        return x, ctx


def _tng_past(x, lag):
    if lag == 0:
        return x
    if lag >= x.shape[1]:
        return torch.zeros_like(x)
    return torch.cat((torch.zeros_like(x[:, :lag]), x[:, :-lag]), dim=1)


def _tng_rms(x):
    z = x.float()
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class CompositionalMemory(nn.Module):
    def __init__(self, vocab, width, rank, order):
        super().__init__()
        self.vocab, self.width, self.rank, self.order = vocab, width, rank, order
        _st = torch.get_rng_state()
        torch.manual_seed(71)
        self.factors = nn.Embedding(vocab, order * rank)
        self.absorb = nn.Parameter(torch.ones(order - 2, rank, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(order - 1, dtype=torch.float32))
        self.kv = Linear((order - 1) * rank, 2 * width, bias=False)
        self.conv = nn.Parameter(torch.zeros(3, width, dtype=torch.float32))
        with torch.no_grad():
            self.factors.weight.bernoulli_(0.5).mul_(2).sub_(1)
            self.kv.weight[:width].normal_(0, ((order - 1) * rank) ** -0.5)
            self.kv.weight[width:].zero_()
        torch.set_rng_state(_st)

    def forward(self, tokens, h):
        d = h.dtype
        b, t = tokens.shape
        n, r = self.order, self.rank
        a = F.embedding(tokens, self.factors.weight.to(d)).view(b, t, n, r)
        shifted = [_tng_past(a[:, :, i], n - 1 - i) for i in range(n)]
        suffix = shifted[-1]
        parts = []
        for size in range(2, n + 1):
            suffix = shifted[n - size] * suffix
            z = suffix
            for i in range(n - size):
                z = z * self.absorb[i].to(d)
            parts.append(_tng_rms(z) * self.log_scale[size - 2].exp().to(d))
        e = torch.cat(parts, dim=-1)
        kv = F.linear(e, self.kv.weight.to(d))
        k, v = kv.split(self.width, dim=-1)
        score = (_tng_rms(h).float() * _tng_rms(k).float()).sum(-1, keepdim=True) / math.sqrt(self.width)
        u = torch.sigmoid(score).to(d) * v
        z = _tng_rms(u)
        c = sum(_tng_past(z, i * self.order) * self.conv[i].to(d) for i in range(3))
        return h + (u + F.silu(c)).to(h.dtype)


def ngmem_indices(idx, n=NGMEM_N, filas=NGMEM_ROWS):
    x = idx.to(torch.int64) + 1
    h = torch.zeros_like(x)
    for k in range(n):
        s = torch.cat((torch.zeros_like(x[:, :k]), x[:, :x.shape[1] - k]), dim=1)
        h = torch.remainder(h * 8209 + s, 2147483647)
    return torch.remainder(h * 1103515245 + 12345, 2147483647) % filas


def ngmem_filas(model, idx):
    if model.training and tuple(model.ngmem_rows.shape[:2]) == tuple(idx.shape):
        return model.ngmem_rows
    h = ngmem_indices(idx.detach().cpu(), NGMEM_N, model.ngmem_table.shape[0])
    return F.embedding(h.to(idx.device), model.ngmem_table).to(COMPUTE_DTYPE)


class _NGMemHost:
    def __init__(self, model, device, rank, world, lr):
        import queue as _q
        import threading as _t
        self.filas, self.d = int(model.ngmem_table.shape[0]), int(model.ngmem_table.shape[1])
        self.tab = torch.zeros((self.filas, self.d), dtype=torch.float32)
        self.acc = torch.zeros(self.filas, dtype=torch.float32)
        self.device, self.rank, self.world, self.lr = device, rank, world, float(lr)
        self.pend = []
        self.err = None
        self.pasos = 0
        self.seg = 0.0
        self.tocadas = 0
        self.q = _q.Queue(maxsize=4)
        self.th = _t.Thread(target=self._run, daemon=True)
        self.th.start()

    def lote(self, bx, by, bs, device):
        h = ngmem_indices(bx, NGMEM_N, self.filas)
        e = F.embedding(h, self.tab).to(COMPUTE_DTYPE)
        return bx.to(device, copy=True), by.to(device, copy=True), dict(bs, ng_h=h.to(torch.float32).to(device), ng_e=e.to(device))

    def carga(self, model, st):
        with torch.no_grad():
            model.ngmem_rows.copy_(st["ng_e"])

    def guarda(self, model, st):
        g = model.ngmem_rows.grad
        model.ngmem_rows.grad = None
        if g is not None:
            self.pend.append((st["ng_h"], g))

    def _todos(self, t):
        if self.world == 1:
            return t.unsqueeze(0)
        z = torch.zeros_like(t)
        full = torch.stack([t if r == self.rank else z for r in range(self.world)])
        dist.all_reduce(full, op=dist.ReduceOp.SUM)
        return full

    def flush(self, lrm, ga):
        if self.err is not None:
            raise RuntimeError(f"ARCH_NGMEM: el hilo de la tabla murio: {self.err!r}")
        if not self.pend:
            return
        hs = self._todos(torch.stack([h for h, _ in self.pend]))
        gs = self._todos(torch.stack([g for _, g in self.pend]))
        self.pend = []
        self.q.put((hs, gs, self.lr * float(lrm), float(ga * DEVICE_BATCH_SIZE * SEQ_LEN * PACK_MULT)))

    def aplica(self, h, g, lr):
        u, inv = torch.unique(h, return_inverse=True)
        gu = torch.zeros((u.numel(), self.d), dtype=torch.float32).index_add_(0, inv, g)
        a = self.acc[u] + gu.square().mean(dim=1)
        self.acc[u] = a
        self.tab[u] = self.tab[u] - lr * gu / (a.sqrt() + 1e-12).unsqueeze(1)
        return u.numel()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                t0 = time.monotonic()
                hs, gs, lr, esc = item[0].cpu(), item[1].cpu(), item[2], item[3]
                self.tocadas += self.aplica(hs.reshape(-1).to(torch.int64), gs.reshape(-1, self.d).to(torch.float32) * esc, lr)
                self.pasos += 1
                self.seg += time.monotonic() - t0
        except Exception as ex:
            self.err = ex

    def finish(self, model):
        self.q.put(None)
        self.th.join()
        if self.err is not None:
            raise RuntimeError(f"ARCH_NGMEM: el hilo de la tabla murio: {self.err!r}")
        with torch.no_grad():
            model.ngmem_table.copy_(self.tab.to(model.ngmem_table.dtype))


def _mimetic_qk_(attn, gen):
    d, hd = attn.c_q.weight.shape[1], attn.head_dim
    eye = torch.eye(d, dtype=torch.float64)
    wq = torch.empty(attn.n_head * hd, d, dtype=torch.float64)
    wk = torch.empty_like(wq)
    for h in range(attn.n_head):
        z = torch.randn(d, d, generator=gen, dtype=torch.float64) / math.sqrt(d)
        u, s, vh = torch.linalg.svd(MIMETIC_QK_ALPHA * z + MIMETIC_QK_BETA * eye)
        u, vh = u[:, :hd], vh[:hd]
        r = s[:hd].sqrt() * torch.sign(u.gather(0, u.abs().argmax(0, keepdim=True))[0])
        wq[h * hd:(h + 1) * hd] = r[:, None] * u.T
        wk[h * hd:(h + 1) * hd] = r[:, None] * vh
    for w, m in ((attn.c_q.weight, wq), (attn.c_k.weight, wk)):
        w.copy_(m * (w.double().square().mean().sqrt() / m.square().mean().sqrt()))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64 + config.vocab_pad
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.register_buffer("pad_bias", torch.cat((torch.zeros(config.vocab_size), torch.full((padded_vocab_size - config.vocab_size,), -1e4))), persistent=False)
        self.value_embed = nn.Embedding(padded_vocab_size, config.attn_width or config.n_embd) if config.ve_layers else None
        self.tng = CompositionalMemory(padded_vocab_size, config.n_embd, config.tng_rank, config.tng_order) if config.tng_rank else None
        if config.hier_width:
            _hc = dataclasses.replace(config, n_embd=config.hier_width, n_head=config.hier_width // (config.n_embd // config.n_head), attn_layers="0,1" if HIER_ATTN else "99", ve_layers="", attn_width=0, mlp_gated="", moe_layers="", ctx_rank=0, key_shift_layers="", xsa=False, skip_src=-1, skip_dst=-1)
            self.hier_to_local = Linear(config.n_embd, config.hier_width, bias=False)
            self.hier_encoder = Block(_hc, 0)
            self.hier_pack = Linear(2 * config.hier_width, config.n_embd, bias=False)
            self.hier_condition = Linear(config.n_embd, config.hier_width, bias=False)
            self.hier_reader = Block(_hc, 1)
            for _blk in (self.hier_encoder, self.hier_reader):
                if _blk.attn is not None and HIER_WINDOW:
                    _blk.attn.hier_window = HIER_WINDOW
            self.hier_to_wide = Linear(config.hier_width, config.n_embd, bias=False)
        else:
            self.hier_to_local = None
        self.skip_lambda = nn.Parameter(torch.zeros(1)) if config.skip_src >= 0 else None
        self.ts_center = nn.Parameter(torch.ones(config.n_embd)) if TWOSTREAM else None
        self.ts_offset = nn.Parameter(torch.zeros(config.n_embd)) if TWOSTREAM else None
        self.smear_lambda = nn.Parameter(torch.zeros(1)) if config.smear else None
        self.unet_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2)) if config.unet else None
        self.bigram_embed = nn.Embedding(config.bigram_n, config.bigram_dim) if config.bigram_n else None
        self.bigram_proj = Linear(config.bigram_dim, config.n_embd, bias=False) if config.bigram_n else None
        self.mtp_head = nn.ModuleList([Linear(config.n_embd, config.mtp_head, bias=False), Linear(config.mtp_head, padded_vocab_size, bias=False)]) if config.mtp_head else None
        self.mudd_r1 = nn.ModuleList([Linear(config.n_embd, 64, bias=False), Linear(64, 6, bias=False)]) if config.mudd_tail else None
        self.mudd_r2 = nn.ModuleList([Linear(config.n_embd, 64, bias=False), Linear(64, 3, bias=False)]) if config.mudd_tail else None
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.rope_log_scale = nn.Parameter(torch.zeros((config.n_embd // config.n_head) // 2, dtype=torch.float32)) if config.rope_learned else None
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        if NF_ATTN:
            self.register_buffer("nf_attn_on", torch.ones(1, dtype=torch.int32))
        if NF_FFN:
            self.register_buffer("nf_ffn_on", torch.ones(1, dtype=torch.int32))
        self.ngmem_proj = None
        if NGMEM:
            with torch.random.fork_rng(devices=[]):
                self.ngmem_proj = Linear(NGMEM_DIM, config.n_embd, bias=False)
            _ngb = math.sqrt(3.0 / NGMEM_DIM)
            with torch.no_grad():
                self.ngmem_proj.weight.uniform_(-_ngb, _ngb, generator=torch.Generator().manual_seed(torch.initial_seed() + 11))
            self.register_buffer("ngmem_table", torch.zeros(NGMEM_ROWS, NGMEM_DIM))
            self.ngmem_rows = nn.Parameter(torch.zeros(DEVICE_BATCH_SIZE * PACK_MULT, config.sequence_len, NGMEM_DIM))

    @torch.no_grad()
    def init_weights(self) -> None:
        _seed0 = torch.initial_seed()
        _mexp = _mlp_expand_list(self.config)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        if TIE_UNTIL > 0:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.005)
            self.transformer.wte.weight.copy_(self.lm_head.weight)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for _gi, block in enumerate(self.transformer.h):
            if block.attn is not None:
                if block.attn.lga:
                    for _w in (block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight):
                        _full = torch.empty(self.config.n_embd, self.config.n_embd)
                        torch.nn.init.uniform_(_full, -s, s)
                        _w.copy_(_full[:_w.shape[0]])
                    _lg = torch.Generator().manual_seed(torch.initial_seed() + 1)
                    _bl = math.sqrt(3.0 / block.attn.c_proj.weight.shape[1])
                    torch.nn.init.uniform_(block.attn.c_proj.weight, -_bl, _bl, generator=_lg)
                    if block.attn.lga_gate is not None:
                        torch.nn.init.zeros_(block.attn.lga_gate.weight)
                    torch.nn.init.zeros_(block.attn.lga_out.weight)
                else:
                    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                    torch.nn.init.zeros_(block.attn.c_proj.weight)
                if block.attn.head_gate is not None:
                    torch.nn.init.zeros_(block.attn.head_gate.weight)
            if REUSE_RANK and not block.mlp.reuse_anchor:
                torch.nn.init.uniform_(torch.empty(block.mlp.up_out.weight.shape[0], block.mlp.up_in.weight.shape[1]), -0.4 * s, 0.4 * s)
            else:
                for _fc in (block.mlp.c_fc if isinstance(block.mlp.c_fc, nn.ModuleList) else [block.mlp.c_fc]):
                    if _mexp[_gi] != self.config.mlp_mult and _fc.weight.shape[0] == _mexp[_gi] * self.config.n_embd:
                        _uni = torch.empty(self.config.mlp_mult * self.config.n_embd, _fc.weight.shape[1], device=_fc.weight.device, dtype=_fc.weight.dtype)
                        torch.nn.init.uniform_(_uni, -0.4 * s, 0.4 * s)
                        _st = torch.get_rng_state()
                        torch.manual_seed(_seed0 + 10000 + _gi)
                        torch.nn.init.uniform_(_fc.weight, -0.4 * s, 0.4 * s)
                        torch.set_rng_state(_st)
                        _sh = min(_fc.weight.shape[0], _uni.shape[0])
                        _fc.weight[:_sh].copy_(_uni[:_sh])
                    else:
                        torch.nn.init.uniform_(_fc.weight, -0.4 * s, 0.4 * s)
                for _pj in (block.mlp.c_proj if isinstance(block.mlp.c_proj, nn.ModuleList) else [block.mlp.c_proj]):
                    torch.nn.init.zeros_(_pj.weight)
            if getattr(block.mlp, "ctx_in", 0):
                block.mlp.c_fc.weight[:, -block.mlp.ctx_in:].zero_()
            if block.ctx_proj is not None:
                torch.nn.init.uniform_(block.ctx_proj.weight, -s, s)
            if isinstance(block.mlp, MoEMLP):
                torch.nn.init.normal_(block.mlp.router.weight, mean=0.0, std=0.02)
            block.resid_lambda.fill_(1.0)
            block.x0_lambda.fill_(0.05)
            if GROW > 0 and _gi in _GROW_SET:
                block.x0_lambda.zero_()
        if self.value_embed is not None:
            torch.nn.init.normal_(self.value_embed.weight, mean=0.0, std=self.config.n_embd**-0.5)
            for block in self.transformer.h:
                if block.ve_lambda is not None:
                    block.ve_lambda.fill_(1.0)
        if self.skip_lambda is not None:
            self.skip_lambda.zero_()
        if self.smear_lambda is not None:
            self.smear_lambda.zero_()
        if self.unet_lambdas is not None:
            self.unet_lambdas.zero_()
        if self.bigram_embed is not None:
            torch.nn.init.normal_(self.bigram_embed.weight, mean=0.0, std=0.8)
            torch.nn.init.zeros_(self.bigram_proj.weight)
        if self.mtp_head is not None:
            torch.nn.init.normal_(self.mtp_head[0].weight, mean=0.0, std=self.config.n_embd**-0.5)
            torch.nn.init.normal_(self.mtp_head[1].weight, mean=0.0, std=0.001)
        if self.mudd_r1 is not None:
            assert self.config.n_layer == 8, "mudd_tail asume 8 capas (h3, h5, capa 6)"
            for _r in (self.mudd_r1, self.mudd_r2):
                torch.nn.init.kaiming_normal_(_r[0].weight)
                torch.nn.init.zeros_(_r[1].weight)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        if self.ts_center is not None:
            self.ts_center.fill_(1.0)
            self.ts_offset.copy_(0.01 * (2.0 * torch.rand(self.config.n_embd, generator=torch.Generator().manual_seed(1234)) - 1.0) if TWOSTREAM_DIFF else torch.zeros(self.config.n_embd))
            _ts_rng = torch.Generator().manual_seed(torch.initial_seed())
            for _blk in self.transformer.h:
                torch.nn.init.normal_(_blk.ts_router.weight, std=(2 * self.config.n_embd) ** -0.5, generator=_ts_rng)
                _blk.ts_router.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, math.log(99.0)]))
                _blk.ts_router.scale.fill_(0.01)
        if REUSE_RANK:
            _ru_rng = torch.Generator().manual_seed(torch.initial_seed())
            for _blk in self.transformer.h:
                if _blk.mlp.reuse_anchor:
                    continue
                for _l in (_blk.mlp.up_in, _blk.mlp.up_out, _blk.mlp.down_in):
                    _b = math.sqrt(3.0 / _l.weight.shape[1])
                    torch.nn.init.uniform_(_l.weight, -_b, _b, generator=_ru_rng)
                _blk.mlp.up_out.weight.mul_(0.4 * math.sqrt(1.0 - REUSE_B0 * REUSE_B0))
                torch.nn.init.zeros_(_blk.mlp.down_out.weight)
                _blk.mlp.beta_up.fill_(REUSE_B0)
                _blk.mlp.beta_down.fill_(REUSE_B0)
        if self.hier_to_local is not None:
            _sh = math.sqrt(3.0) * self.config.n_embd**-0.5
            _sl = math.sqrt(3.0) * self.config.hier_width**-0.5
            torch.nn.init.uniform_(self.hier_to_local.weight, -_sh, _sh)
            torch.nn.init.uniform_(self.hier_pack.weight, -_sl / math.sqrt(2.0), _sl / math.sqrt(2.0))
            torch.nn.init.uniform_(self.hier_condition.weight, -_sh, _sh)
            torch.nn.init.zeros_(self.hier_to_wide.weight)
            for _blk in (self.hier_encoder, self.hier_reader):
                if _blk.attn is not None:
                    for _w in (_blk.attn.c_q.weight, _blk.attn.c_k.weight, _blk.attn.c_v.weight):
                        torch.nn.init.uniform_(_w, -_sl, _sl)
                    torch.nn.init.zeros_(_blk.attn.c_proj.weight)
                torch.nn.init.uniform_(_blk.mlp.c_fc.weight, -0.4 * _sl, 0.4 * _sl)
                torch.nn.init.zeros_(_blk.mlp.c_proj.weight)
                _blk.resid_lambda.fill_(1.0)
                _blk.x0_lambda.zero_()
        if MIMETIC_QK:
            _mg = torch.Generator().manual_seed(_seed0 + 104729)
            for _blk in self.transformer.h:
                if _blk.attn is not None and not _blk.attn.lga:
                    _mimetic_qk_(_blk.attn, _mg)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 100000):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        if self.config.partial_rope:
            inv_freq = torch.cat((inv_freq[:32], torch.zeros_like(inv_freq[32:])))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        if ROPE_MM:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer  # per-block scalars
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T y AV: 2*h*d*T por token cada uno, x3 por fwd+bwd, y la mitad por el enmascarado
        # causal; solo cuentan los bloques que tienen atencion, con sus cabezas reales.
        attn_flops = sum(6 * blk.attn.n_head * blk.attn.head_dim * t for blk in self.transformer.h if blk.attn is not None)
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        grow_scalar, grow_matrix, reuse_params, hgate_params, fir_params = [], [], [], [], []
        for _bi, block in enumerate(self.transformer.h):
            if GROW > 0 and _bi in _GROW_SET:
                assert block.ve_lambda is None and block.ctx_proj is None
                grow_scalar.extend([block.resid_lambda, block.x0_lambda])
                if block.attn is not None:
                    grow_matrix.extend(p for n, p in block.attn.named_parameters() if n not in ("xsa_alpha", "attn_tau"))
                grow_matrix.extend(block.mlp.parameters())
                continue
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.fir_a is not None:
                fir_params.extend([block.fir_a, block.fir_b])
            if block.ve_lambda is not None:
                scalar_params.append(block.ve_lambda)
            if block.attn is not None:
                matrix_params.extend(p for n, p in block.attn.named_parameters() if n not in ("xsa_alpha", "attn_tau", "head_gate.weight"))
                if block.attn.head_gate is not None:
                    hgate_params.append(block.attn.head_gate.weight)
                if block.attn.xsa_alpha is not None:
                    scalar_params.append(block.attn.xsa_alpha)
                if block.attn.attn_tau is not None:
                    scalar_params.append(block.attn.attn_tau)
            if REUSE_RANK and not block.mlp.reuse_anchor:
                reuse_params.extend([block.mlp.up_in.weight, block.mlp.up_out.weight, block.mlp.down_in.weight, block.mlp.down_out.weight])
                scalar_params.extend([block.mlp.beta_up, block.mlp.beta_down])
            else:
                matrix_params.extend(block.mlp.parameters())
        tng_a, tng_kv, tng_s = [], [], []
        if self.tng is not None:
            tng_a = [self.tng.factors.weight]
            tng_kv = [self.tng.kv.weight]
            tng_s = [self.tng.absorb, self.tng.log_scale, self.tng.conv]
        ts_params = []
        if TWOSTREAM:
            ts_params = [self.ts_center, self.ts_offset] + [q for block in self.transformer.h for q in block.ts_router.parameters()]
        embedding_params = [] if TIE_UNTIL > 0 else list(self.transformer.wte.parameters())
        if self.value_embed is not None:
            embedding_params += list(self.value_embed.parameters())
        if self.bigram_embed is not None:
            embedding_params += list(self.bigram_embed.parameters())
            matrix_params += list(self.bigram_proj.parameters())
        if self.skip_lambda is not None:
            scalar_params.append(self.skip_lambda)
        if self.smear_lambda is not None:
            scalar_params.append(self.smear_lambda)
        if self.unet_lambdas is not None:
            scalar_params.append(self.unet_lambdas)
        hier_adamw = []
        if self.hier_to_local is not None:
            _hm = hier_adamw if HIER_ADAMW else matrix_params
            for _blk in (self.hier_encoder, self.hier_reader):
                scalar_params.extend([_blk.resid_lambda, _blk.x0_lambda])
                if _blk.attn is not None:
                    _hm.extend(p for n, p in _blk.attn.named_parameters() if n != "xsa_alpha")
                _hm.extend(_blk.mlp.parameters())
            _hm.extend([self.hier_to_local.weight, self.hier_pack.weight, self.hier_condition.weight, self.hier_to_wide.weight])
        lm_head_params = list(self.lm_head.parameters()) + (list(self.mtp_head.parameters()) if self.mtp_head is not None else [])
        if self.ngmem_proj is not None:
            lm_head_params += list(self.ngmem_proj.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale * EMB_LR_MULT, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            *([dict(kind="adamw", params=list(self.transformer.wte.parameters()), lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01)] if TIE_UNTIL > 0 else []),
            *([dict(kind="adamw", params=list(self.mudd_r1.parameters()) + list(self.mudd_r2.parameters()), lr=0.25 * UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.01)] if self.mudd_r1 is not None else []),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
            *([dict(kind="adamw", params=ts_params, lr=0.25 * UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0)] if ts_params else []),
            *([dict(kind="adamw", params=hier_adamw, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=WEIGHT_DECAY)] if hier_adamw else []),
            *([dict(kind="adamw", params=reuse_params, lr=0.001, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)] if reuse_params else []),
            *([dict(kind="adamw", params=hgate_params, lr=0.25 * UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0)] if hgate_params else []),
            *([dict(kind="adamw", params=fir_params, lr=0.01, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0)] if fir_params else []),
            *([dict(kind="adamw", params=[self.rope_log_scale], lr=0.001, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)] if self.rope_log_scale is not None else []),
            *([dict(kind="adamw", params=tng_a, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)] if tng_a else []),
            *([dict(kind="adamw", params=tng_kv, lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)] if tng_kv else []),
            *([dict(kind="adamw", params=tng_s, lr=0.005, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)] if tng_s else []),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            _lr_fix = ((shape[-1] / self.config.n_embd) ** 0.5) if (self.config.ctx_rank and shape[-1] == self.config.n_embd + self.config.ctx_rank) else 1.0
            param_groups.append(dict(kind="muon", params=params, lr=MATRIX_LR * _lr_fix, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))
        if GROW > 0:
            param_groups.append(dict(kind="adamw", params=grow_scalar, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, grow=True))
            for shape in sorted({p.shape for p in grow_matrix}):
                param_groups.append(dict(kind="muon", params=[p for p in grow_matrix if p.shape == shape], lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY, grow=True))
        opt = MuonAdamW(param_groups)
        opt.grow_frozen = GROW > 0
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"] * LR_MULT
            if group.get("grow") and group["kind"] == "muon" and group["params"]:
                _p0 = group["params"][0]; _st = opt.state[_p0]; _sh = _p0.shape; _n = len(group["params"])
                _st["momentum_buffer"] = torch.zeros(_n, *_sh, dtype=_p0.dtype, device=_p0.device)
                _st["second_momentum_buffer"] = torch.zeros((_n, _sh[-2], 1) if _sh[-2] >= _sh[-1] else (_n, 1, _sh[-1]), dtype=torch.float32, device=_p0.device)
                _st["stacked_grads"] = torch.empty(_n, *_sh, dtype=_p0.dtype, device=_p0.device)
                _st["stacked_params"] = torch.empty(_n, *_sh, dtype=_p0.dtype, device=_p0.device)
            if STACKED_PARAMS and group["kind"] == "muon" and group["params"]:
                with torch.no_grad():
                    _stack = torch.stack([q.data for q in group["params"]]).contiguous()
                    for _i, _q in enumerate(group["params"]):
                        _q.data = _stack[_i]
                group["stack"] = _stack
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", *args, **kwargs):
        out = self._forward_core(idx, targets, loss_reduction, *args, **kwargs)
        if not self.config.moe_layers or targets is None or loss_reduction != "mean" or MOE_AUX <= 0:
            return out
        aux = 0.0
        for blk in self.transformer.h:
            if isinstance(blk.mlp, MoEMLP):
                aux = aux + blk.mlp.aux
        return out + MOE_AUX * aux

    def _forward_core(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", aux_alpha=None, aux_alpha3=None, targets_prefix=None, pair=False, grow=False, targets_members=None, lex_beta=None):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        if self.rope_log_scale is not None:
            _hd = self.config.n_embd // self.config.n_head
            _idx = torch.arange(0, _hd, 2, dtype=torch.float32, device=idx.device)
            _fr = (1.0 / (100000.0 ** (_idx / _hd))) * self.rope_log_scale.float().exp()
            _pos = torch.arange(t, dtype=torch.float32, device=idx.device)
            _ang = torch.outer(_pos, _fr)
            _c = _ang.cos().to(COMPUTE_DTYPE)[None, :, None, :]
            _s = _ang.sin().to(COMPUTE_DTYPE)[None, :, None, :]
            cos, sin = torch.cat((_c, _c), dim=-1), torch.cat((_s, _s), dim=-1)
        else:
            cos, sin = self.cos[:, :t], self.sin[:, :t]
        _t2p = None
        if pair:
            assert t % 2 == 0 and self.bigram_embed is None and self.smear_lambda is None and not self.config.hourglass and not DOC_MASK and not self.config.key_shift_layers and self.mtp_head is None and self.mudd_r1 is None and not MTP_H3 and PREFIX_BETA == 0
            cos, sin = self.cos[:, 1:t:2], self.sin[:, 1:t:2]
            if targets is not None:
                targets, _t2p = targets[:, 1::2], torch.cat((targets[:, 2::2], torch.full_like(targets[:, :1], -1)), dim=1)
            x = _pair_pool(_emb(self.transformer.wte, idx).to(COMPUTE_DTYPE))
        else:
            x = _emb(self.transformer.wte, idx).to(COMPUTE_DTYPE)
        if self.bigram_embed is not None:
            _prev = torch.cat((torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]), dim=1)
            _h = torch.remainder(_prev.to(torch.int64) * 8192 + idx.to(torch.int64), self.config.bigram_n)
            x = x + self.bigram_proj(self.bigram_embed(_h).to(COMPUTE_DTYPE))
        if self.smear_lambda is not None:
            x = x + self.smear_lambda.to(x.dtype) * torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        x = norm(x)
        x0 = x
        ve = _emb(self.value_embed, idx).to(COMPUTE_DTYPE) if self.value_embed is not None else None
        if pair and ve is not None:
            ve = _pair_pool(ve)
        _ts = self.ts_center is not None
        if _ts:
            assert not pair and not grow and not self.config.hourglass and self.mudd_r1 is None and self.unet_lambdas is None and self.hier_to_local is None
            _tu = x * (self.ts_center + self.ts_offset).to(x.dtype)
            _tv = x * (self.ts_center - self.ts_offset).to(x.dtype)
            _saved_v = None
        _hier = self.hier_to_local is not None
        if _hier:
            assert not pair and not grow and t % 2 == 0 and _t2p is None and not DOC_MASK and self.mtp_head is None
            _hraw = x
            _hin = self.hier_to_local(x)
            _hloc, _ = self.hier_encoder(_hin, _hin, cos, sin)
            x = norm(self.hier_pack(_hloc.reshape(_b, t // 2, 2 * self.config.hier_width)))
            x0 = x
            cos, sin = self.cos[:, 1:t:2], self.sin[:, 1:t:2]
            if ve is not None:
                ve = (ve[:, 0::2] + ve[:, 1::2]) / math.sqrt(2.0)
        _kmask = torch.cat((torch.ones_like(idx[:, :1], dtype=torch.bool), idx[:, 1:] == BOS_ID), dim=1)[:, :, None, None] if self.config.key_shift_layers else None
        if self.config.doc_mask:
            _doc = torch.cumsum((idx == BOS_ID).to(torch.int32), dim=1)
            _kmask = ((_doc[:, :, None] == _doc[:, None, :]) & torch.ones(t, t, dtype=torch.bool, device=idx.device).tril())[:, None, :, :]
        saved = None
        skips = []
        n_enc = self.config.n_layer // 2
        _hg = [int(_v) for _v in self.config.hourglass.split(",") if _v]
        _ctx = None
        _hist = None
        for i, block in enumerate(self.transformer.h):
            if self.tng is not None and i == self.config.tng_at:
                x = self.tng(idx, x)
            if grow and i in _GROW_SET:
                continue
            if _ts:
                if self.skip_lambda is not None and i == self.config.skip_dst and saved is not None:
                    _tu = _tu + self.skip_lambda.to(x.dtype) * saved
                    _tv = _tv + self.skip_lambda.to(x.dtype) * _saved_v
                _tu, _tv, _ctx = _ts_block(block, _tu, _tv, x0, cos, sin, ve, _kmask, _ctx)
                if self.skip_lambda is not None and i == self.config.skip_src:
                    saved, _saved_v = _tu, _tv
                continue
            if self.skip_lambda is not None and i == self.config.skip_dst and saved is not None:
                x = x + self.skip_lambda.to(x.dtype) * saved
            if self.unet_lambdas is not None and i >= n_enc and skips:
                x = x + self.unet_lambdas[i - n_enc].to(x.dtype) * skips.pop()
            if _hg and i == _hg[0]:
                _h_full, _x0_full, _cos_full, _sin_full, _ve_full = x, x0, cos, sin, ve
                _s = _hg_pool(x)
                x, x0 = _s, _hg_pool(x0)
                cos, sin = cos[:, :t // 2], sin[:, :t // 2]
                ve = _hg_pool(ve) if ve is not None else None
            _ve_i = ve
            if self.mudd_r1 is not None and i == 6:
                _ve_i = _vadd if ve is None else ve + _vadd
            if REUSE_RANK:
                x, _ctx, _hist = block(x, x0, cos, sin, _ve_i, _kmask, _ctx, _hist)
            else:
                x, _ctx = block(x, x0, cos, sin, _ve_i, _kmask, _ctx)
            if self.mudd_r1 is not None:
                if i == 3:
                    _h3 = x
                if i == 5:
                    _h5 = x
                    _c = (self.mudd_r1[1](F.gelu(self.mudd_r1[0](norm(x)))) * MUDD_SCALE).to(x.dtype)
                    x = x + _c[..., 0:1] * x0 + _c[..., 1:2] * _h3 + _c[..., 2:3] * _h5
                    _vadd = _c[..., 3:4] * x0 + _c[..., 4:5] * _h3 + _c[..., 5:6] * _h5
            if _hg and i == _hg[-1]:
                x = _h_full + (x - _s).repeat_interleave(2, dim=1)
                x0, cos, sin, ve = _x0_full, _cos_full, _sin_full, _ve_full
            if self.skip_lambda is not None and i == self.config.skip_src:
                saved = x
            if self.unet_lambdas is not None and i < n_enc:
                skips.append(x)
        if _ts:
            x = 0.5 * (_tu + _tv)
        if self.mudd_r1 is not None:
            _c2 = (self.mudd_r2[1](F.gelu(self.mudd_r2[0](norm(x)))) * MUDD_SCALE).to(x.dtype)
            x = x + _c2[..., 0:1] * x0 + _c2[..., 1:2] * _h3 + _c2[..., 2:3] * _h5
        x = norm(x)
        if _hier:
            _hctx = torch.cat((torch.zeros_like(x[:, :1]), torch.stack((x, x), dim=2).reshape(_b, t, x.size(-1))), dim=1)[:, :t]
            _hcnd = self.hier_condition(x)
            _hcnd = torch.cat((torch.zeros_like(_hcnd[:, :1]), torch.stack((_hcnd, _hcnd), dim=2).reshape(_b, t, _hcnd.size(-1))), dim=1)[:, :t]
            _hrin = _hloc + _hcnd
            _hdec, _ = self.hier_reader(_hrin, _hrin, self.cos[:, :t], self.sin[:, :t])
            x = norm(_hraw + _hctx + self.hier_to_wide(_hdec))
        if self.ngmem_proj is not None:
            x = x + self.ngmem_proj(ngmem_filas(self, idx).to(x.dtype))
        _aux_x = x if (self.mtp_head is not None and targets is not None) else None
        if (CE_FUSED and self.training and targets is not None and loss_reduction == "mean" and STATIC_CE and _aux_x is None
                and not MTP_H3 and PREFIX_BETA == 0 and targets_members is None and _t2p is None and LOSS_PADROWS == 0
                and not self.config.mask_special and not self.config.logit_asym):
            return _cef_loss(self, x, targets, aux_alpha if MTP else None)
        logits = self.lm_head(x) if self.config.vocab_pad else self.lm_head(x)[..., :self.config.vocab_size]
        if not HEAD_BF16:
            logits = logits.float()
        logits = (11.5 * torch.tanh((logits + 5.0) / 15.0)) if self.config.logit_asym else (LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP))
        if self.config.mask_special:
            _vs = self.config.vocab_size
            logits = torch.cat((logits[..., :_vs - 4], torch.full_like(logits[..., _vs - 4:_vs], -10000.0), logits[..., _vs:]), dim=-1)
            if targets is not None:
                targets = targets.masked_fill(targets >= self.config.vocab_size - 4, -1)
        if targets is None:
            return logits[..., :self.config.vocab_size] if self.config.vocab_pad else logits
        if self.config.vocab_pad:
            logits = logits + self.pad_bias.to(logits.dtype)
        if STATIC_CE and loss_reduction == "mean":
            _t = targets.reshape(-1)
            _lp = torch.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
            if LOSS_PADROWS > 0:
                _lp = torch.cat((_lp, _lp[-LOSS_PADROWS:]), dim=0)
                _t = torch.cat((_t, torch.full_like(_t[:LOSS_PADROWS], -1)), dim=0)
            _valid = (_t >= 0).float()
            _nll = _nll_pick(_lp, _t.clamp_min(0))
            if MTP and aux_alpha is not None:
                _t2 = (torch.cat((targets[:, 1:], torch.full_like(targets[:, :1], -1)), dim=1) if _t2p is None else _t2p).reshape(-1)
                if LOSS_PADROWS > 0:
                    _t2 = torch.cat((_t2, torch.full_like(_t2[:LOSS_PADROWS], -1)), dim=0)
                _m = ((_t2 >= 0) & (_t != BOS_ID) & (_t2 != BOS_ID)).float() * _valid
                if _aux_x is not None:
                    _h0, _h1 = self.mtp_head[0], self.mtp_head[1]
                    def _aux_chunk(_xc, _tc):
                        _lg = _h1(_h0(_xc))
                        _lg = (_lg + self.pad_bias.to(_lg.dtype)) if self.config.vocab_pad else _lg[..., :self.config.vocab_size]
                        _lpc = torch.log_softmax(_lg.reshape(-1, _lg.size(-1)), dim=-1)
                        return _nll_pick(_lpc, _tc.clamp_min(0))
                    _ax = _aux_x.reshape(-1, _aux_x.size(-1))
                    if LOSS_PADROWS > 0:
                        _ax = torch.cat((_ax, _ax[-LOSS_PADROWS:]), dim=0)
                    _xs = _ax.chunk(MTP_HEAD_CHUNKS, dim=0)
                    _ts = _t2.chunk(MTP_HEAD_CHUNKS, dim=0)
                    _nll2 = torch.cat([torch.utils.checkpoint.checkpoint(_aux_chunk, _xc, _tc, use_reentrant=False) for _xc, _tc in zip(_xs, _ts)], dim=0)
                    _w = aux_alpha.to(_nll.dtype) * _m
                    return ((_nll + _w * _nll2) / (1.0 + _w) * _valid).sum() / _valid.sum()
                if MTP_H3 and aux_alpha3 is not None:
                    _t3 = torch.cat((targets[:, 2:], torch.full_like(targets[:, :2], -1)), dim=1).reshape(-1)
                    _m3 = ((_t3 >= 0) & (_t3 != BOS_ID)).float() * _m
                    _g3 = _lp.gather(1, torch.stack((_t.clamp_min(0), _t2.clamp_min(0), _t3.clamp_min(0)), dim=1)).float()
                    _nll, _nll2, _nll3 = -_g3[:, 0], -_g3[:, 1], -_g3[:, 2]
                    _w = aux_alpha.to(_nll.dtype) * _m
                    _w3 = aux_alpha3.to(_nll.dtype) * _m3
                    return ((_nll + _w * _nll2 + _w3 * _nll3) / (1.0 + _w + _w3) * _valid).sum() / _valid.sum()
                if PREFIX_BETA > 0 and (targets_prefix is not None or getattr(self, "prefix_table", None) is not None):
                    _tp = targets_prefix.reshape(-1) if targets_prefix is not None else self.prefix_table[_t.clamp_min(0)]
                    _g3p = _lp.gather(1, torch.stack((_t.clamp_min(0), _t2.clamp_min(0), _tp.clamp_min(0)), dim=1)).float()
                    _nll, _nll2, _nllp = -_g3p[:, 0], -_g3p[:, 1], -_g3p[:, 2]
                    _w = aux_alpha.to(_nll.dtype) * _m
                    _wp = PREFIX_BETA * (_tp >= 0).float() * _valid
                    return ((_nll + _w * _nll2 + _wp * _nllp) / (1.0 + _w + _wp) * _valid).sum() / _valid.sum()
                if targets_members is not None:
                    assert _t2p is None
                    _ymf = targets_members.reshape(-1, 1) if LEXSETS_MM else targets_members.reshape(-1, targets_members.size(-1))
                    if LOSS_PADROWS > 0:
                        _ymf = torch.cat((_ymf, _ymf[-LOSS_PADROWS:]), dim=0)
                    if LEXSETS_MM:
                        _g2m = _lp.gather(1, torch.stack((_t.clamp_min(0), _t2.clamp_min(0)), dim=1)).float()
                        _nll, _nll2 = -_g2m[:, 0], -_g2m[:, 1]
                        _pg = torch.matmul(torch.exp(_lp), _LEX_M.to(_lp.dtype))
                        _oh = F.one_hot(_ymf.reshape(-1), _pg.size(1)).to(_pg.dtype)
                        _nllg = -torch.log((_pg * _oh).sum(dim=1).float().clamp_min(1e-30))
                    else:
                        _gm = _lp.gather(1, torch.cat((_t.clamp_min(0).unsqueeze(1), _t2.clamp_min(0).unsqueeze(1), _ymf), dim=1)).float()
                        _nll, _nll2 = -_gm[:, 0], -_gm[:, 1]
                        _nllg = -torch.logsumexp(_gm[:, 2:], dim=1)
                    _w = aux_alpha.to(_nll.dtype) * _m
                    _bt = lex_beta.to(_nll.dtype) * _valid
                    return (((_nll + _w * _nll2) / (1.0 + _w) + _bt * _nllg) / (1.0 + _bt) * _valid).sum() / _valid.sum()
                _g2 = _lp.gather(1, torch.stack((_t.clamp_min(0), _t2.clamp_min(0)), dim=1)).float()
                _nll, _nll2 = -_g2[:, 0], -_g2[:, 1]
                _w = aux_alpha.to(_nll.dtype) * _m
                return ((_nll + _w * _nll2) / (1.0 + _w) * _valid).sum() / _valid.sum()
            if PREFIX_BETA > 0 and aux_alpha is not None and (targets_prefix is not None or getattr(self, "prefix_table", None) is not None):
                _tp = targets_prefix.reshape(-1) if targets_prefix is not None else self.prefix_table[_t.clamp_min(0)]
                _wp = PREFIX_BETA * (_tp >= 0).float() * _valid
                _nllp = -_lp.gather(1, _tp.clamp_min(0).unsqueeze(1)).squeeze(1).float()
                return ((_nll + _wp * _nllp) / (1.0 + _wp) * _valid).sum() / _valid.sum()
            return (_nll * _valid).sum() / _valid.sum()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config(depth: int = DEPTH) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    _width = int(os.environ.get("ARCH_WIDTH", "896"))
    if _width > 0:
        n_embd = _width
    n_head = n_embd // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        ve_layers=os.environ.get("ARCH_VE", "0,1,10,11"),
        skip_src=int(os.environ.get("ARCH_SKIP_SRC", "1")),
        skip_dst=int(os.environ.get("ARCH_SKIP_DST", "10")),
        attn_layers=os.environ.get("ARCH_ATTN_LAYERS", "0,2,4,6,8,10"),
        mlp_mult=int(os.environ.get("ARCH_MLP_MULT", "4")),
        mlp_expand=os.environ.get("ARCH_MLP_EXPAND", ""),
        mlp_budget=int(os.environ.get("ARCH_MLP_BUDGET", "0")),
        mlp_gated=os.environ.get("ARCH_MLP_GATED", ""),
        bigram_n=int(os.environ.get("ARCH_BIGRAM_N", "0")),
        bigram_dim=int(os.environ.get("ARCH_BIGRAM_DIM", "256")),
        mtp_head=int(os.environ.get("ARCH_MTP_HEAD", "0")),
        attn_width=int(os.environ.get("ARCH_ATTN_WIDTH", "0")),
        hourglass=os.environ.get("ARCH_HOURGLASS", ""),
        logit_asym=os.environ.get("ARCH_LOGIT_ASYM", "0") == "1",
        partial_rope=os.environ.get("ARCH_PARTIAL_ROPE", "0") == "1",
        rope_learned=os.environ.get("ARCH_ROPE_LEARNED", "0") == "1",
        doc_mask=os.environ.get("ARCH_DOC_MASK", "0") == "1",
        key_shift_layers=os.environ.get("ARCH_KEY_SHIFT_LAYERS", ""),
        mudd_tail=os.environ.get("ARCH_MUDD_TAIL", "0") == "1",
        xsa=os.environ.get("ARCH_XSA", "0") == "1",
        ctx_rank=int(os.environ.get("ARCH_CTX_RANK", "0")),
        lga_layers=os.environ.get("ARCH_LGA_LAYERS", ""),
        lga_heads=int(os.environ.get("ARCH_LGA_HEADS", "2")),
        lga_gate=os.environ.get("ARCH_LGA_GATE", "1") == "1",
        cfv_layers=os.environ.get("ARCH_CFV_LAYERS", ""),
        tng_rank=int(os.environ.get("ARCH_TNG_RANK", "0")),
        tng_order=int(os.environ.get("ARCH_TNG_ORDER", "5")),
        tng_at=int(os.environ.get("ARCH_TNG_AT", "2")),
        head_gate=os.environ.get("ARCH_HEAD_GATE", "0") == "1",
        fir_layers=os.environ.get("ARCH_FIR_LAYERS", ""),
        ctx_source=os.environ.get("ARCH_CTX_SOURCE", "attn"),
        moe_layers=os.environ.get("ARCH_MOE_LAYERS", ""),
        moe_experts=int(os.environ.get("ARCH_MOE_E", "8")),
        moe_k=int(os.environ.get("ARCH_MOE_K", "2")),
        moe_hidden=int(os.environ.get("ARCH_MOE_HIDDEN", "0")),
        moe_cap=float(os.environ.get("ARCH_MOE_CAP", "1.25")),
        smear=os.environ.get("ARCH_SMEAR", "0") == "1",
        unet=os.environ.get("ARCH_UNET", "0") == "1",
        attn_tau=ATTN_TAU,
        mask_special=os.environ.get("ARCH_MASK_SPECIAL", "0") == "1",
        vocab_pad=int(os.environ.get("ARCH_VOCAB_PAD", "64")),
        hier_width=int(os.environ.get("ARCH_HIER_WIDTH", "0")),
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


_MUON_STEP_CACHE = {}


def _muon_core(g, w, mb, second, lr_t, mom_t, beta2, wd, ns_steps, tall, vv=None, vm=None):
    if vv is not None:
        vv = vv * mom_t + (mb - g).square() * (mom_t * (1.0 - mom_t))
        vm = vm * mom_t + (1.0 - mom_t)
    mb = torch.lerp(mb, g, 1.0 - mom_t)
    g = torch.lerp(g, mb, mom_t)
    if vv is not None:
        g = g / ((vv / vm).clamp_min(0.0).sqrt() + 1e-8).to(g.dtype)
    x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if tall:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    red_dim = -1 if tall else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second = torch.lerp(second, v_mean, 1.0 - beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = ((g * w) >= 0).to(w.dtype)
    lr = lr_t.to(w.dtype)
    w = w - (lr * g + lr * wd * w * mask)
    if vv is not None:
        return w, mb, second, vv, vm
    return w, mb, second


def _muon_step_fn(dev_type):
    if dev_type not in _MUON_STEP_CACHE:
        _MUON_STEP_CACHE[dev_type] = torch.compile(_muon_core, backend="neuron", dynamic=False) if dev_type == "neuron" else _muon_core
    return _MUON_STEP_CACHE[dev_type]


def _muon_compiled_sharded_step(self, group):
    """Same Muon core, disjoint matrices; gather FP32 before each local BF16 cast."""
    assert MASTER_FP32 and OPT_COMPILE and not (STACKED_PARAMS or MUON_SHARD or MUON_VS)
    params = group["params"]
    if not params:
        return
    rank, world = dist.get_rank(), dist.get_world_size()
    count, shape = len(params), params[0].shape
    assert len(shape) == 2 and all(p.shape == shape for p in params)
    slots = (count + world - 1) // world
    start = rank * slots
    valid = max(0, min(slots, count - start))
    local_params = params[start:start + valid]
    state = self.state[params[0]]
    if "mcs_layout" not in state:
        state["mcs_layout"] = (rank, world, count, slots, tuple(shape))
        kw = dict(device=params[0].device)
        state["master"] = torch.zeros(slots, *shape, dtype=torch.float32, **kw)
        state["momentum_buffer"] = torch.zeros(slots, *shape, dtype=params[0].dtype, **kw)
        second_shape = (shape[-2], 1) if shape[-2] >= shape[-1] else (1, shape[-1])
        state["second_momentum_buffer"] = torch.zeros(slots, *second_shape, dtype=torch.float32, **kw)
        state["stacked_grads"] = torch.zeros_like(state["momentum_buffer"])
        state["gathered_master"] = torch.empty(slots * world, *shape, dtype=torch.float32, **kw)
        if valid:
            torch._foreach_copy_(list(state["master"][:valid].unbind(0)), local_params)
    assert state["mcs_layout"] == (rank, world, count, slots, tuple(shape))
    grads = state["stacked_grads"]
    for j, p in enumerate(local_params):
        if p.grad is None:
            grads[j].zero_()
        else:
            grads[j].copy_(p.grad)
    w = state["master"]
    mb, sec = state["momentum_buffer"], state["second_momentum_buffer"]
    lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
    lr_t = torch.tensor(lr, dtype=torch.float32, device=w.device)
    mom_t = torch.tensor(group["momentum"], dtype=torch.float32, device=w.device)
    wn, mn, sn = _muon_step_fn(w.device.type)(
        grads, w, mb, sec, lr_t, mom_t, group["beta2"], group["weight_decay"],
        group["ns_steps"], shape[-2] >= shape[-1])
    w.copy_(wn)
    mb.copy_(mn)
    sec.copy_(sn)
    # Do not gather BF16: each rank must cast with its own stochastic rounding.
    gathered = state["gathered_master"]
    dist.all_gather_into_tensor(gathered, w)
    torch._foreach_copy_(params, list(gathered[:count].unbind(0)))


_MUON_SLOW_STEP_CACHE = {}


def _muon_core_slow(g, w, mb, second, ms, lr_t, mom_t, alpha_t, beta3, beta2, wd, ns_steps, tall):
    ms = torch.lerp(ms, g.float(), 1.0 - beta3)
    mb = torch.lerp(mb, g, 1.0 - mom_t)
    g = torch.lerp(g, mb, mom_t)
    g = (g.float() + alpha_t * ms).to(g.dtype)
    x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if tall:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    red_dim = -1 if tall else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second = torch.lerp(second, v_mean, 1.0 - beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = ((g * w) >= 0).to(w.dtype)
    lr = lr_t.to(w.dtype)
    w = w - (lr * g + lr * wd * w * mask)
    return w, mb, second, ms


def _muon_slow_step_fn(dev_type):
    if dev_type not in _MUON_SLOW_STEP_CACHE:
        _MUON_SLOW_STEP_CACHE[dev_type] = torch.compile(_muon_core_slow, backend="neuron", dynamic=False) if dev_type == "neuron" else _muon_core_slow
    return _MUON_SLOW_STEP_CACHE[dev_type]


def _muon_compiled_sharded_slow_step(self, group):
    assert MASTER_FP32 and OPT_COMPILE and not (STACKED_PARAMS or MUON_SHARD or MUON_VS)
    params = group["params"]
    if not params:
        return
    rank, world = dist.get_rank(), dist.get_world_size()
    count, shape = len(params), params[0].shape
    assert len(shape) == 2 and all(p.shape == shape for p in params)
    slots = (count + world - 1) // world
    start = rank * slots
    valid = max(0, min(slots, count - start))
    local_params = params[start:start + valid]
    state = self.state[params[0]]
    if "mcs_layout" not in state:
        state["mcs_layout"] = (rank, world, count, slots, tuple(shape))
        kw = dict(device=params[0].device)
        state["master"] = torch.zeros(slots, *shape, dtype=torch.float32, **kw)
        state["momentum_buffer"] = torch.zeros(slots, *shape, dtype=params[0].dtype, **kw)
        second_shape = (shape[-2], 1) if shape[-2] >= shape[-1] else (1, shape[-1])
        state["second_momentum_buffer"] = torch.zeros(slots, *second_shape, dtype=torch.float32, **kw)
        state["slow_momentum_buffer"] = torch.zeros(slots, *shape, dtype=torch.float32, **kw)
        state["stacked_grads"] = torch.zeros_like(state["momentum_buffer"])
        state["gathered_master"] = torch.empty(slots * world, *shape, dtype=torch.float32, **kw)
        if valid:
            torch._foreach_copy_(list(state["master"][:valid].unbind(0)), local_params)
    assert state["mcs_layout"] == (rank, world, count, slots, tuple(shape))
    grads = state["stacked_grads"]
    for j, p in enumerate(local_params):
        if p.grad is None:
            grads[j].zero_()
        else:
            grads[j].copy_(p.grad)
    w = state["master"]
    mb, sec, ms = state["momentum_buffer"], state["second_momentum_buffer"], state["slow_momentum_buffer"]
    lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
    lr_t = torch.tensor(lr, dtype=torch.float32, device=w.device)
    mom_t = torch.tensor(group["momentum"], dtype=torch.float32, device=w.device)
    alpha_t = torch.tensor(group.get("slow_alpha", 0.0), dtype=torch.float32, device=w.device)
    wn, mn, sn, msn = _muon_slow_step_fn(w.device.type)(
        grads, w, mb, sec, ms, lr_t, mom_t, alpha_t, MUON_SLOW_BETA, group["beta2"], group["weight_decay"],
        group["ns_steps"], shape[-2] >= shape[-1])
    w.copy_(wn)
    mb.copy_(mn)
    sec.copy_(sn)
    ms.copy_(msn)
    gathered = state["gathered_master"]
    dist.all_gather_into_tensor(gathered, w)
    torch._foreach_copy_(params, list(gathered[:count].unbind(0)))



class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def _adamw_step(self, group):
        if ADAMW_FOREACH:
            return self._adamw_step_foreach(group)
        assert not MASTER_FP32, "ARCH_MASTER_FP32 exige ARCH_ADAMW_FOREACH=1"
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            p.mul_(1 - group["lr"] * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.add_(update, alpha=-group["lr"])

    @torch.no_grad()
    def _adamw_step_foreach(self, group):
        beta1, beta2 = group["betas"]
        ps = [p for p in group["params"] if p.grad is not None]
        if not ps:
            return
        for p in ps:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
        grads = [p.grad for p in ps]
        m = [self.state[p]["exp_avg"] for p in ps]
        v = [self.state[p]["exp_avg_sq"] for p in ps]
        b1 = [1 - beta1 ** self.state[p]["step"] for p in ps]
        b2 = [1 - beta2 ** self.state[p]["step"] for p in ps]
        _tgt = ps
        if MASTER_FP32:
            for p in ps:
                if "master" not in self.state[p]:
                    self.state[p]["master"] = p.detach().float().clone()
            _tgt = [self.state[p]["master"] for p in ps]
        torch._foreach_mul_(_tgt, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_lerp_(m, grads, 1 - beta1)
        torch._foreach_lerp_(v, torch._foreach_mul(grads, grads), 1 - beta2)
        den = torch._foreach_div(v, b2)
        torch._foreach_sqrt_(den)
        torch._foreach_add_(den, group["eps"])
        upd = torch._foreach_div(torch._foreach_div(m, b1), den)
        torch._foreach_add_(_tgt, upd, alpha=-group["lr"])
        if MASTER_FP32:
            torch._foreach_copy_(ps, _tgt)

    @torch.no_grad()
    def _muon_step(self, group):
        if MUON_SLOW:
            assert dist.is_available() and dist.is_initialized(), "ARCH_MUON_SLOW exige torch.distributed inicializado (torchrun)"
            return _muon_compiled_sharded_slow_step(self, group)
        if MUON_COMPILED_SHARD and dist.is_available() and dist.is_initialized():
            return _muon_compiled_sharded_step(self, group)
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
        if MASTER_FP32 and "master" not in state:
            state["master"] = torch.stack([p.detach().float() for p in params])

        _stk = STACKED_PARAMS and "stack" in group
        assert not (MASTER_FP32 and (_stk or MUON_SHARD)), "ARCH_MASTER_FP32 no admite ARCH_STACKED_PARAMS ni ARCH_MUON_SHARD"
        if _stk:
            w = group["stack"]
            g = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        else:
            g = state["stacked_grads"]
            w = state["stacked_params"]
            for i, p in enumerate(params):
                if p.grad is None:
                    g[i].zero_()
                else:
                    g[i].copy_(p.grad)
                if not MASTER_FP32:
                    w[i].copy_(p)
            if MASTER_FP32:
                w = state["master"]
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]
        if MUON_VS:
            assert not MUON_SHARD
            if "vs_var" not in state:
                state["vs_var"] = torch.zeros(len(params), *shape, dtype=torch.float32, device=p0.device)
                state["vs_mass"] = torch.zeros((), dtype=torch.float32, device=p0.device)
            _vb = group["momentum"]
            _vv, _vm = state["vs_var"], state["vs_mass"]
        if MUON_VS and not OPT_COMPILE:
            if "vs_tmp" not in state:
                state["vs_tmp"] = torch.empty_like(_vv)
            _vt = state["vs_tmp"]
            torch.sub(momentum_buffer, g, out=_vt)
            _vt.mul_(_vt)
            _vv.mul_(_vb).add_(_vt, alpha=_vb * (1.0 - _vb))
            _vm.mul_(_vb).add_(1.0 - _vb)
        _sl = None
        if MUON_SHARD and dist.is_available() and dist.is_initialized():
            _ws, _rk, _n = dist.get_world_size(), dist.get_rank(), len(params)
            _b, _e = divmod(_n, _ws)
            _st = _rk * _b + min(_rk, _e)
            _sl = slice(_st, _st + _b + (1 if _rk < _e else 0))
            if _sl.start >= _sl.stop:
                _sl = slice(0, 1)
            _wfull = w
            g, w, momentum_buffer, second = g[_sl], w[_sl], momentum_buffer[_sl], second[_sl]

        if OPT_COMPILE:
            assert _sl is None, "ARCH_OPT_COMPILE no admite ARCH_MUON_SHARD"
            _lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
            _lrt = torch.tensor(_lr, dtype=torch.float32, device=w.device)
            _mt = torch.tensor(group["momentum"], dtype=torch.float32, device=w.device)
            if MUON_VS:
                _w_new, _mb_new, _sec_new, _vv_new, _vm_new = _muon_step_fn(w.device.type)(g, w, momentum_buffer, second, _lrt, _mt, group["beta2"], group["weight_decay"], group["ns_steps"], shape[-2] >= shape[-1], state["vs_var"], state["vs_mass"])
                state["vs_var"].copy_(_vv_new)
                state["vs_mass"].copy_(_vm_new)
            else:
                _w_new, _mb_new, _sec_new = _muon_step_fn(w.device.type)(g, w, momentum_buffer, second, _lrt, _mt, group["beta2"], group["weight_decay"], group["ns_steps"], shape[-2] >= shape[-1])
            momentum_buffer.copy_(_mb_new)
            second.copy_(_sec_new)
            w.copy_(_w_new)
        else:
            momentum_buffer.lerp_(g, 1 - group["momentum"])
            g = g.lerp_(momentum_buffer, group["momentum"])
            if MUON_VS and not OPT_COMPILE:
                _vt = state["vs_tmp"]
                torch.div(state["vs_var"], state["vs_mass"], out=_vt)
                _vt.clamp_min_(0.0).sqrt_().add_(1e-8)
                g = g / _vt.to(g.dtype)
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
            second.lerp_(v_mean, 1 - group["beta2"])
            step_size = second.clamp_min(1e-10).rsqrt()
            scaled = (v_mean * red_dim_size) * step_size.square()
            v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
            g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

            lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
            wd = group["weight_decay"]
            mask = (g * w) >= 0
            w.sub_(lr * g + lr * wd * w * mask)
        if _sl is not None:
            _ws, _n = dist.get_world_size(), len(params)
            _b, _e = divmod(_n, _ws)
            for _i in range(_n):
                _src = _i // (_b + 1) if _i < _e * (_b + 1) else _e + (_i - _e * (_b + 1)) // max(_b, 1)
                dist.broadcast(_wfull[_i], src=_src)
            w = _wfull
        if not _stk:
            torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self):
        if NO_BWD:
            return
        for group in self.param_groups:
            if group.get("grow") and getattr(self, "grow_frozen", False):
                continue
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


def _accum_grads(params, acc):
    gs = [p.grad for p in params]
    for p in params:
        p.grad = None
    if acc is None:
        return gs
    dst, src = [], []
    for i, g in enumerate(gs):
        if g is None:
            continue
        if acc[i] is None:
            acc[i] = g
        else:
            dst.append(acc[i])
            src.append(g)
    if dst:
        torch._foreach_add_(dst, src)
    return acc


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if FLAT_SYNC:
        by_dtype = {}
        for p in model.parameters():
            if p.grad is not None:
                by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
        for gs in by_dtype.values():
            flat = torch.cat([g.reshape(-1) for g in gs])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            torch._foreach_copy_(gs, [c.view_as(g) for g, c in zip(gs, flat.split([g.numel() for g in gs]))])
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": EMA_STATE.get("override") or {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    nf_desde_estado(payload["model_state"])
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Training

def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    return 1.0


def muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * MUON_MOM_FINAL


def muon_slow_alpha(step: int) -> float:
    return MUON_SLOW_ALPHA * (min(step / MUON_SLOW_WARM, 1.0) if MUON_SLOW_WARM > 0 else 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron Competition R1 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "1") != "0",
                        help="enable torch.compile on the full model")
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic optimizer step to warm the compile cache, then exit")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    args = parser.parse_args()

    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2

    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")

    tokenizer = ensure_tokenizer(build_if_missing=False)
    if MTP:
        assert tokenizer.get_bos_token_id() == BOS_ID, tokenizer.get_bos_token_id()
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config), indent=2)}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    if MIMETIC_QK:
        print0(f"mimetic_qk: alfa={MIMETIC_QK_ALPHA} beta={MIMETIC_QK_BETA} semilla_z={torch.initial_seed() + 104729} capas={[b.attn.layer_idx for b in model.transformer.h if b.attn is not None and not b.attn.lga]}")
    if BF16_WEIGHTS:
        model.to(torch.bfloat16)
        for _n, _p in model.named_parameters():
            if _n.endswith("xsa_alpha") and XSA_FP32_ALPHA:
                _p.data = _p.data.float()
            if _n.endswith("attn_tau"):
                _p.data = _p.data.float()
            if TWOSTREAM and (_n.endswith("ts_router.bias") or _n.endswith("ts_router.scale") or _n in ("ts_center", "ts_offset")):
                _p.data = _p.data.float()
            if _n.startswith("tng.") and _n != "tng.factors.weight":
                _p.data = _p.data.float()
            if _n == "rope_log_scale":
                _p.data = _p.data.float()
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = model.estimate_flops()
    print0(f"num_params={num_params:,} flops_per_token={flops_per_token:e}")
    if NF_ATTN or NF_FFN:
        print0(f"nf: attn={NF_ATTN} ffn={NF_FFN} mlp_saved_relu={MLP_SAVED_RELU} mlp_fp8={MLP_FP8}")
    if MLP_DW8:
        print0(f"dw8: ARCH_MLP_DW8=1 dW del MLP en fp8 double_row (mlp_fp8_bwd_act -> dpk/hpk + mlp_fp8_dw8_x/_y), S=2^{MLP_FP8_DYSCALE}")
    if MLP_EPIPASS:
        print0(f"epipass: ARCH_MLP_EPIPASS={MLP_EPIPASS} epilogo de dpre^T en mlp_fp8_bwd_act por {'scalar_tensor_tensor directa a e5m2' if MLP_EPIPASS == 1 else 'tensor_tensor + tensor_scalar por kgs'} (sin gb bf16)")
    if MLP_FWDEPI:
        print0(f"fwdepi: ARCH_MLP_FWDEPI={MLP_FWDEPI} epilogo de mlp_fp8_fwd_r en dos pasadas (rT' = min(max(ps, 0), 61.75*s1) de PSUM{' en ACT' if MLP_FWDEPI == 2 else ''} y hs = e4m3(rT'*q*rT'), q = (_SCALE_H/s1)^2); rT sin escala, kgs y h8 pliegan _SCALE_H/s1")
    if KA_MLP:
        print0("ka_mlp: ARCH_KA_MLP=1 LT512 (load_transpose2d de 512 tokens en mlp_fp8_fwd_r y mlp_fp8_bwd_act) + PK1 (dpk/hpk por una pk [128, 4, 128] bf16 de PSUM)")
    if ATTN_DW8:
        print0(f"adw8: ARCH_ATTN_DW8=1 dW de c_q/c_k/c_v/c_proj en fp8 double_row (arch::adw8_dw), S=2^{ATTN_DW8_K} lnc={ATTN_DW8_LNC} act={ATTN_DW8_ACT}")

    if LEXSETS > 0:
        assert MTP and not MTP_H3 and PREFIX_BETA == 0 and PAIR_PHASE == 0, "ARCH_LEXSETS requiere MTP compartido y excluye H3, prefijo y parejas"
        global _LEX_DEV, _LEX_M, _LEX_G
        _ltab = _lex_members_table(tokenizer, TOKENIZER_VOCAB_SIZE)
        _LEX_DEV = _ltab.to(device)
        if LEXSETS_MM:
            _lm, _lg = _lex_group_tables(_ltab)
            _LEX_M, _LEX_G = _lm.to(device), _lg.to(device)
        print0(f"lexsets: {TOKENIZER_VOCAB_SIZE // LEXSETS_GROUP} grupos de {LEXSETS_GROUP} ids ({'permutacion aleatoria semilla ' + str(LEXSETS_RANDOM) if LEXSETS_RANDOM else 'orden (bytes, id)'}), beta0={LEXSETS}, fin={LEXSETS_END:.4f}")
    if GROW > 0:
        _c = model.config
        assert all(0 <= _i < _c.n_layer for _i in _GROW_SET) and _c.skip_src not in _GROW_SET and _c.skip_dst not in _GROW_SET and not any(int(_v) in _GROW_SET for _v in _c.ve_layers.split(",") if _v) and not _c.hourglass and not _c.unet and not _c.mudd_tail and not _c.ctx_rank and PAIR_PHASE == 0
        print0(f"grow: bloques {sorted(_GROW_SET)} inactivos hasta la fraccion {GROW} del reloj cobrado; x0_lambda=0 y proyecciones de salida a cero")
    if PREFIX_BETA > 0:
        _enc = tokenizer.enc
        _ranks = _enc._mergeable_ranks
        _tab = torch.full((model.lm_head.weight.shape[0],), -1, dtype=torch.long)
        for _i in range(tokenizer.get_vocab_size()):
            try:
                _b = _enc.decode_single_token_bytes(_i)
            except Exception:
                continue
            for _k in range(len(_b) - 1, 0, -1):
                _j = _ranks.get(_b[:_k])
                if _j is not None:
                    _tab[_i] = _j
                    break
        model.register_buffer("prefix_table", _tab.to(device), persistent=False)
        global _PREFIX_DEV
        _PREFIX_DEV = _tab.to(device)
    orig_model = model
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    world_tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN * PACK_MULT * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")

    if args.compile_only:
        synthetic_x = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y, aux_alpha=torch.tensor(MTP_ALPHA, dtype=torch.float32, device=device), aux_alpha3=(torch.tensor(MTP_ALPHA3, dtype=torch.float32, device=device) if MTP_H3 else None)) if MTP else model(synthetic_x, synthetic_y)
            (loss / grad_accum_steps).backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        sync_gradients(model)
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN * PACK_MULT, "train", device)
    _ng_host = None
    if NGMEM:
        assert PREFETCH and not STATIC_INPUTS and not NO_BWD and PAIR_PHASE == 0 and GROW == 0, "ARCH_NGMEM exige ARCH_PREFETCH=1, sin ARCH_STATIC_INPUTS, ARCH_NO_BWD, ARCH_PAIR_PHASE ni ARCH_GROW"
        _ng_host = _NGMemHost(orig_model, device, rank, world_size, NGMEM_LR or EMBEDDING_LR * (orig_model.config.n_embd / 768) ** -0.5 * EMB_LR_MULT)
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN * PACK_MULT, "train", "cpu")
        print0(f"ngmem: ARCH_NGMEM=1 n={NGMEM_N} tabla {_ng_host.filas}x{_ng_host.d} fp32 en el anfitrion, Adagrad por fila lr={_ng_host.lr:.4f} x lrm, reserva de checkpoint {CHECKPOINT_RESERVE_SECONDS:.0f} s")

    def _next_batch(loader):
        bx, by, bs = next(loader)
        if PACK_MULT > 1:
            bx = bx.reshape(-1, SEQ_LEN)
            by = by.reshape(-1, SEQ_LEN)
        if LAST_TARGET_GUARD:
            by[-1, -1:] = by[-1, -1:].masked_fill(by[-1, -1:] >= 8185, -1)
        return bx, by, bs

    if PREFETCH:
        import queue as _queue
        import threading as _threading
        _pf_q = _queue.Queue(maxsize=2)
        _pf_src = train_loader
        _pf_orig = _next_batch
        _pf_stop = [False]
        def _pf_worker():
            while not _pf_stop[0]:
                _bx, _by, _bs = _pf_orig(_pf_src)
                if _ng_host is not None:
                    _pf_q.put(_ng_host.lote(_bx, _by, _bs, device))
                    continue
                _pf_q.put((_bx.clone(), _by.clone(), _bs))
        _pf_t = _threading.Thread(target=_pf_worker, daemon=True)
        _pf_t.start()
        def _next_batch(loader):
            while True:
                try:
                    return _pf_q.get(timeout=120)
                except _queue.Empty:
                    if not _pf_t.is_alive():
                        raise RuntimeError("el hilo de precarga murió (ver traza anterior)")
    x, y, loader_state = _next_batch(train_loader)
    if STATIC_INPUTS:
        _xb, _yb = x.clone(), y.clone()
        x, y = _xb, _yb
    _yp = _PREFIX_DEV[y.long()] if (PREFIX_BETA > 0 and _PREFIX_DEV is not None) else None
    _ym = (_LEX_G if LEXSETS_MM else _LEX_DEV)[y.long().clamp_min(0)] if _LEX_DEV is not None else None
    optimizer = model.setup_optimizer()
    if MUON_SLOW:
        print0(f"muon_slow: beta3={MUON_SLOW_BETA} alpha={MUON_SLOW_ALPHA} warm={MUON_SLOW_WARM} buffer=fp32 shard=mcs")
    ema_params = None
    wavg_acc, wavg_n = None, 0
    if WAVG_FRAC > 0:
        assert EMA_DECAY == 0, "ARCH_WAVG y ARCH_EMA escriben los dos en final_raw.pt y en EMA_STATE: son excluyentes"
        assert RANK_SOUP == 0, "ARCH_WAVG toma instantaneas antes del promedio entre rangos: no se combinan sin medirlos por separado"
        assert 0 < WAVG_FRAC <= 1 and WAVG_K >= 1, "ARCH_WAVG en (0,1] y ARCH_WAVG_K >= 1"
        wavg_acc = {} if rank == 0 else None
    if EMA_DECAY > 0:
        ema_params = {n: p.detach().to(dtype=torch.float32, copy=True) for n, p in orig_model.named_parameters()}
    if EMB_SCATTER:
        print0(f"embk: ARCH_EMB_SCATTER=1 op={'arch::embk_dw' if _EMBK_OP is not None else 'index_add_ fp32 (sin nki)'} lnc={EMB_SCATTER_LNC}")
    if EMB_SCATTER and EMB_SCATTER_GATE and rank == 0:
        _ew = orig_model.transformer.wte.weight
        _gr = embk_puerta_dispositivo(device, x, _ew.shape[0], _ew.shape[1])
        print(embk_linea(_gr, os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN", "")), flush=True)
        if not _gr["ok"]:
            import sys
            sys.exit(3)
    if ATTN_FUSED_BWD and ATTN_FUSED_BWD_GATE and rank == 0:
        _gb = attfb_puerta_dispositivo(device, orig_model.cos, orig_model.sin)
        print("ATTFB_GATE ok=" + str(_gb["ok"]) + " " + " ".join(f"{k}:frob={_gb[k]['frob']:.3e},max={_gb[k]['max']:.3e},escala={_gb[k]['escala']:+.2e}"
              for k in ("dqp", "dkp", "dv")) + f" sr={os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '')}", flush=True)
        if not _gb["ok"]:
            raise SystemExit(3)
    if CE_FUSED:
        print0(f"cef: ARCH_CE_FUSED=1 op={'arch::cef_fwd + arch::cef_bwd' if _CEF_OP is not None else 'emulacion torch (sin nki)'} lnc={CE_FUSED_LNC} tw={_CF_TW}")
    if CE_FUSED and CE_FUSED_GATE and rank == 0:
        _cg = cef_puerta_dispositivo(device)
        print(cef_linea(_cg), flush=True)
        if not _cg["ok"]:
            import sys
            sys.exit(3)
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    _pair_on = PAIR_PHASE > 0
    _grow_on = GROW > 0
    _acc_ps = list(orig_model.parameters()) if ACCUM_FOREACH else None
    if ACCUM_FOREACH:
        print0(f"acum: ARCH_ACCUM_FOREACH=1 p.grad al acumulador tras cada backward y un torch._foreach_add_ por micro-paso >= 2 ({len(_acc_ps)} parametros)")

    _prof = None
    if PROF_DIR and rank == 0 and str(device).startswith("neuron"):
        import torch.profiler as _tp
        from torch_neuronx.profiling import NeuronConfig as _NC, ProfileMode as _PM, NeuronProfiler as _NP
        _ncfg = _NC(modes=[_PM.DEVICE, _PM.RUNTIME], profile_output_dir=PROF_DIR, capture_enabled_for_nc="0")
        _nprof = _NP(_ncfg)
        _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU], schedule=_tp.schedule(wait=PROF_WAIT, warmup=1, active=1, repeat=1), experimental_config=_ncfg, on_trace_ready=lambda p: _nprof.export_trace(p))
        _prof.__enter__()
        print0(f"profile: dir={PROF_DIR} wait={PROF_WAIT}")
    while step < args.num_steps:
        synchronize(device)
        t0 = time.monotonic()
        _tdata = 0.0
        optimizer.zero_grad(set_to_none=True)
        _acc = None
        last_loss = None
        _ga = grad_accum_steps
        if TBS_RAMP > 0 and _sched_progress(step, args, budget_started) < TBS_RAMP:
            _ga = max(1, grad_accum_steps // 2)
        if _pair_on and step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            _pf = torch.tensor([1.0 if _sched_progress(step, args, budget_started) < PAIR_PHASE else 0.0], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(_pf, 0)
            if float(_pf.item()) < 0.5:
                _pair_on = False
                print0(f"pair: cambio a resolución normal en el paso {step} ({time.monotonic() - budget_started:.0f} s cobrados)")
        if _grow_on and step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            _gf = torch.tensor([1.0 if _sched_progress(step, args, budget_started) < GROW else 0.0], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(_gf, 0)
            if float(_gf.item()) < 0.5:
                _grow_on = False
                optimizer.grow_frozen = False
                print0(f"grow: bloques {sorted(_GROW_SET)} activados en el paso {step} ({time.monotonic() - budget_started:.0f} s cobrados)")
        if (_pair_on or _grow_on) and PAIR_WARM and step == 0:
            _pair_warm_caches(orig_model)
            _tw = time.monotonic()
            (model(x, y, aux_alpha=torch.tensor(0.0, dtype=torch.float32, device=device), aux_alpha3=None, **({"targets_members": _ym, "lex_beta": torch.tensor(0.0, dtype=torch.float32, device=device)} if _ym is not None else {})) if MTP else model(x, y)).backward()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            print0(f"pair: grafo normal compilado y descartado en el paso 0 ({time.monotonic() - _tw:.1f} s)")
        for _ in range(_ga):
            if _ng_host is not None:
                _ng_host.carga(orig_model, loader_state)
            if FWD_PROBE and step == 8:
                def _timeit(fn, n=3):
                    synchronize(device); fn(); synchronize(device)
                    _t = time.monotonic()
                    for _ in range(n):
                        fn()
                    synchronize(device)
                    return (time.monotonic() - _t) / n
                _a0 = torch.tensor(0.0, dtype=torch.float32, device=device)
                with torch.no_grad():
                    _ta = _timeit(lambda: model(x))
                    _tb = _timeit(lambda: (model(x, y, aux_alpha=_a0) if MTP else model(x, y)))
                if FWD_PROBE_GRAD:
                    def _fwd_con_grad():
                        _o = (model(x, y, aux_alpha=_a0) if MTP else model(x, y))
                        del _o
                    _tg = _timeit(_fwd_con_grad)
                    print0(f"fwdprobe2: fwd_con_gradiente_vivo {_tg*1000:.1f} ms | micro_tokens {x.numel()}")
                def _fb():
                    optimizer.zero_grad(set_to_none=True)
                    (model(x, y, aux_alpha=_a0) if MTP else model(x, y)).backward()
                _tc = _timeit(_fb)
                print0(f"fwdprobe: fwd_sin_perdida {_ta*1000:.1f} ms | fwd_con_perdida {_tb*1000:.1f} ms | fwd+bwd {_tc*1000:.1f} ms | micro_tokens {x.numel()}")
            if MTP:
                _u = _sched_progress(step, args, budget_started)
                if MTP_PROBE:
                    _f = max(0.0, 1.0 - step / 60.0)
                elif MTP == 1:
                    _f = 1.0 if _u < 0.5 else 0.0
                elif MTP_SHAPE == "const":
                    _f = 1.0 if _u < MTP_END else 0.0
                else:
                    _f = max(0.0, 1.0 - _u / MTP_END)
                _a = MTP_ALPHA * _f
                _beta = LEXSETS * max(0.0, 1.0 - _u / LEXSETS_END)
                _a3 = torch.tensor(MTP_ALPHA3 * _f, dtype=torch.float32, device=device) if MTP_H3 else None
                loss = model(x, y, aux_alpha=torch.tensor(_a, dtype=torch.float32, device=device), aux_alpha3=_a3, **({"targets_prefix": _yp} if _yp is not None else {}), **({"pair": True} if _pair_on else {}), **({"grow": True} if _grow_on else {}), **({"targets_members": _ym, "lex_beta": torch.tensor(_beta, dtype=torch.float32, device=device)} if _ym is not None else {}))
            else:
                loss = model(x, y, **({"pair": True} if _pair_on else {}), **({"grow": True} if _grow_on else {}))
            last_loss = loss.detach()
            if not NO_BWD:
                (loss / _ga).backward()
            if _ng_host is not None:
                _ng_host.guarda(orig_model, loader_state)
            if ACCUM_FOREACH:
                _acc = _accum_grads(_acc_ps, _acc)
            if TIMING:
                synchronize(device)
                _tb = time.monotonic()
            x, y, loader_state = _next_batch(train_loader)
            if STATIC_INPUTS:
                _xb.copy_(x)
                _yb.copy_(y)
                x, y = _xb, _yb
            if DIAG_BATCH:
                _xc, _yc = x.cpu(), y.cpu()
                print(f"diagbatch rank {rank} step {step} x[{int(_xc.min())},{int(_xc.max())}] y[{int(_yc.min())},{int(_yc.max())}] neg1={int((_yc < 0).sum())} filas_todo_neg1={int((_yc < 0).all(dim=1).sum())} shape={tuple(_xc.shape)}", flush=True)
                Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                torch.save((_xc, _yc), Path(args.out_dir) / f"lastbatch_rank{rank}.pt")
            _yp = _PREFIX_DEV[y.long()] if (PREFIX_BETA > 0 and _PREFIX_DEV is not None) else None
            _ym = (_LEX_G if LEXSETS_MM else _LEX_DEV)[y.long().clamp_min(0)] if _LEX_DEV is not None else None
            if TIMING:
                _tdata += time.monotonic() - _tb

        if ACCUM_FOREACH and _acc is not None:
            for _p, _g in zip(_acc_ps, _acc):
                _p.grad = _g
            _acc = None
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if TIMING:
            synchronize(device)
            _t1 = time.monotonic()
        sync_gradients(model)
        if TIMING:
            synchronize(device)
            _t2 = time.monotonic()

        lrm = lr_multiplier(step)
        if WARMDOWN_FRAC > 0:
            _wf = min(1.0, max(0.0, (1.0 - _sched_progress(step, args, budget_started)) / WARMDOWN_FRAC))
            lrm *= (1.0 - math.sqrt(1.0 - _wf)) if WARMDOWN_SHAPE == "sqrt" else _wf
        if LR_LEVELS > 0:
            # El backend nativo compila un kernel por cada valor escalar distinto del LR
            # (medido 4 sep: 0.49 s/paso con lrm fijo, ~3 s/paso cuando cambia cada paso).
            # Escalera de LR_LEVELS niveles: como mucho LR_LEVELS compilaciones.
            if LR_TAIL:
                lrm = min([_v for _v in _LR_LADDER if _v >= lrm - 1e-9] or [0.0])
            else:
                lrm = math.ceil(lrm * LR_LEVELS) / LR_LEVELS
            if dist.is_available() and dist.is_initialized():
                _lt = torch.tensor([lrm], dtype=torch.float32, device=device)
                dist.broadcast(_lt, 0)
                lrm = float(_lt.item())
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY
                if MUON_SLOW:
                    group["slow_alpha"] = muon_slow_alpha(step)
        _tied_now = False
        if TIE_UNTIL > 0:
            _E, _H = orig_model.transformer.wte.weight, orig_model.lm_head.weight
            _tied_now = _sched_progress(step, args, budget_started) < TIE_UNTIL
            if _tied_now:
                if _E.grad is not None:
                    _H.grad.add_(_E.grad)
                    _E.grad = None
            elif not _TIE_SPLIT[0]:
                _sh = optimizer.state.get(_H)
                if _sh:
                    optimizer.state[_E] = {"step": _sh["step"], "exp_avg": _sh["exp_avg"].clone(), "exp_avg_sq": _sh["exp_avg_sq"].clone()}
                _TIE_SPLIT[0] = True
                print0(f"tie: separacion en el paso {step}")
        optimizer.step()
        if _ng_host is not None:
            _ng_host.flush(lrm, _ga)
        if _tied_now:
            _E.data.copy_(_H.data)

        synchronize(device)
        if TIMING:
            print0(f"timing step {step} fwdbwd {_t1 - t0 - _tdata:.3f} data {_tdata:.3f} sync {_t2 - _t1:.3f} opt {time.monotonic() - _t2:.3f}")
        if SYNC_STARTUP and step < 5:
            os.sync()
        record_step_boundary(step)
        if STALL_TEST and 5 <= step < 20:
            time.sleep(STALL_TEST)
        if _prof is not None:
            _prof.step()
        if ema_params is not None:
            for _n, _p in orig_model.named_parameters():
                ema_params[_n].mul_(EMA_DECAY).add_(_p.detach().float(), alpha=1.0 - EMA_DECAY)
        if WAVG_FRAC > 0 and wavg_n < WAVG_K and step % 10 == 0 and step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            _u = _sched_progress(step, args, budget_started)
            _wf = torch.tensor([1.0 if _u >= 1.0 - WAVG_FRAC * (1.0 - wavg_n / WAVG_K) else 0.0], dtype=torch.float32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(_wf, 0)
            if float(_wf.item()) >= 0.5:
                if wavg_acc is not None:
                    for _n, _p in orig_model.named_parameters():
                        _v = _p.detach().float().cpu()
                        wavg_acc[_n] = _v if _n not in wavg_acc else wavg_acc[_n].add_(_v)
                wavg_n += 1
                print0(f"wavg: instantanea {wavg_n}/{WAVG_K} en el paso {step} (progreso {_u:.3f})")
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
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
        elapsed_after_step = time.monotonic() - budget_started
        # Fold the deadline decision into the existing health collective. A
        # local wall-clock check can otherwise make one rank begin another
        # optimizer step while its peers head for the checkpoint barrier.
        # The budget clock is only meaningful once it has been reset at the
        # excluded-startup boundary; before that, elapsed_after_step is measured
        # from loop start and a slow first-step compile could trip the predictive
        # stop and break before five steps complete -- leaving the run unscorable.
        # Gate the time-based stop behind reaching the excluded-step count so the
        # run is always scorable; an explicit num_steps budget still stops early.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
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
        control = torch.tensor(
            [
                1 if (math.isnan(train_loss) or train_loss > 100) else 0,
                1 if stop_after_step else 0,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")

        total_train_tokens += world_tokens_per_micro * _ga
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = args.num_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | "
                f"remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}",
                flush=True,
            )

        if step == 0:
            # After first step all persistent objects are allocated; freeze to avoid GC pauses
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
        if int(control[1].item()) != 0:
            break

    if RANK_SOUP and dist.is_available() and dist.is_initialized():
        _ts = time.monotonic()
        _mx = torch.zeros((), dtype=torch.float32, device=device)
        _rl = torch.zeros((), dtype=torch.float32, device=device)
        # Un solo all_reduce sobre todo el modelo aplanado en fp32: por parametro eran ~200
        # colectivas y 12.47 s (SP1, 14 sep), dentro de una reserva de checkpoint de 5 s.
        _ps = [_p for _, _p in orig_model.named_parameters()]
        _loc = torch.cat([_p.detach().float().reshape(-1) for _p in _ps])
        _av = _loc.clone()
        dist.all_reduce(_av, op=dist.ReduceOp.AVG)
        _d = (_loc - _av).abs()
        _mx = _d.amax()
        _rl = _mx / _av.abs().amax().clamp_min(1e-12)
        if RANK_SOUP == 2:
            _o = 0
            for _p in _ps:
                _p.data.copy_(_av[_o:_o + _p.numel()].view_as(_p).to(_p.dtype))
                _o += _p.numel()
        del _loc, _d
        print0(f"soup: divergencia entre rangos max {float(_mx):.3e} relativa {float(_rl):.3e} en {time.monotonic() - _ts:.2f} s; promedio {'aplicado' if RANK_SOUP == 2 else 'no aplicado'}")
    if wavg_acc is not None and wavg_n > 0 and rank == 0:
        save_checkpoint(Path(args.out_dir) / "final_raw.pt", orig_model, {"wavg": "raw"})
        _st = {k: v.detach().cpu() for k, v in orig_model.state_dict().items()}
        for _n, _p in orig_model.named_parameters():
            wavg_acc[_n] = wavg_acc[_n].add_(_p.detach().float().cpu())
        wavg_n += 1
        for _n, _a in wavg_acc.items():
            _st[_n] = (_a / wavg_n).to(_st[_n].dtype)
        EMA_STATE["override"] = _st
        print0(f"wavg: promedio de {wavg_n} instantaneas (la ultima son los pesos finales) en final.pt; crudos en final_raw.pt")
    if _prof is not None:
        _prof.__exit__(None, None, None)
        print0("profile: cerrado")
    if ema_params is not None:
        if rank == 0:
            save_checkpoint(Path(args.out_dir) / "final_raw.pt", orig_model, {"ema": "raw"})
        EMA_STATE["override"] = {k: v.detach().cpu() for k, v in orig_model.state_dict().items()}
        EMA_STATE["override"].update({n: e.detach().cpu() for n, e in ema_params.items()})
    if _ng_host is not None:
        _ng_host.finish(orig_model)
        print0(f"ngmem: {_ng_host.pasos} actualizaciones de la tabla {_ng_host.filas}x{_ng_host.d} en el anfitrion, {1000.0 * _ng_host.seg / max(1, _ng_host.pasos):.1f} ms por actualizacion, {_ng_host.tocadas / max(1, _ng_host.pasos):.0f} filas tocadas por actualizacion, {int((_ng_host.acc > 0).sum())} filas escritas (rango 0)")
    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        })
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            DEVICE_BATCH_SIZE,
            SEQ_LEN,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        # The budget clock restarts at the fifth completed step to mirror the
        # runner's charged window, so post-restart elapsed time IS the charged
        # estimate (conservative: the runner also excludes pre-loop startup).
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"target_steps:      {args.num_steps}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f"seq_len:           {SEQ_LEN}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")

    if PREFETCH:
        _pf_stop[0] = True
        for _ in range(4):
            try:
                _pf_q.get_nowait()
            except _queue.Empty:
                break
        _pf_t.join(timeout=60)
    cleanup_runtime()


if __name__ == "__main__":
    main()

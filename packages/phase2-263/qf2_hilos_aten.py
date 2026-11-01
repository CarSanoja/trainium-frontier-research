"""QF2_HILOS_ATEN and QF2_HILOS_HILO: fix the ATen intra-op thread count inside train.py instead of trusting the launcher.

The block queues export OMP_NUM_THREADS=4 to the 64 ranks, while torchrun sets OMP_NUM_THREADS=1 when the variable is
absent, so the official evaluator may train with another thread count than our local runs [SIN VERIFICAR]. The MEMHOST
table thread does its gathers with index_select, which ATen parallelises over that count; 64 ranks x 4 threads exceed the
192 vCPU of the host. ATEN > 0 calls torch.set_num_threads(ATEN) in the main thread at the start of main(); HILO > 0 calls
it again from the table thread when it starts. With OpenMP the count of the calling thread changes and threads created
later inherit the last value, while threads already running keep theirs. 0 (default) leaves everything as the launcher
set it. train.py sets ATEN and HILO from the environment, so the values are baked with the other flags.
"""
from __future__ import annotations

import os

import torch

ATEN = 0
HILO = 0


def _fijar(n: int, nombre: str) -> int:
    if n <= 0:
        return 0
    torch.set_num_threads(n)
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_hilos_aten: {nombre} hilos={torch.get_num_threads()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '-')}", flush=True)
    return n


def fijar_proceso() -> int:
    return _fijar(ATEN, "proceso")


def fijar_hilo(nombre: str) -> int:
    return _fijar(HILO, nombre)

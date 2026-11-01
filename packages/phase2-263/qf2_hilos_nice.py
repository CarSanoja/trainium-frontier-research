"""QF2_HILOS_NICE: lower the CPU priority of a background thread (the MEMHOST table thread and the loader prefetch).

With 64 ranks the table threads spend ~0.9-1.2 s of CPU per step in the first ~1.4 s of it, and the 20p6 profile of the
122 shows the device idle +285 ms per step against the 106 in exactly that window (the last micro-step, after the table
work, matches the 106). Linux keeps the nice value per thread, so os.setpriority on the native thread id only affects the
calling thread; raising it needs no privilege. No tensor, order or value changes: only the scheduler weight. 0 (default)
leaves the thread alone. train.py sets NICE from QF2_HILOS_NICE (so the value is baked with the other flags).
"""
from __future__ import annotations

import os
import threading

NICE = 0


def ceder(nombre: str) -> int:
    n = NICE
    if n <= 0:
        return 0
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), n)
    except (OSError, AttributeError) as ex:
        print(f"qf2_hilos_nice: {nombre} sin nice ({ex})", flush=True)
        return 0
    if os.environ.get("RANK", "0") == "0":
        print(f"qf2_hilos_nice: {nombre} nice={os.getpriority(os.PRIO_PROCESS, threading.get_native_id())}", flush=True)
    return n

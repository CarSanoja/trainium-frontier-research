from __future__ import annotations

SEQ_MAX = 0
CARGADOR_LARGO = False
COLA_FRAC = 0.0
CALIENTA_LARGO = False
PASO_CALIENTA = 1


def configurar(seq_max: str, cargador_largo: str, cola_frac: str, calienta_largo: str) -> None:
    global SEQ_MAX, CARGADOR_LARGO, COLA_FRAC, CALIENTA_LARGO
    SEQ_MAX = int(seq_max.strip() or 0)
    CARGADOR_LARGO = cargador_largo.strip() not in ("", "0")
    COLA_FRAC = float(cola_frac.strip() or 0.0)
    CALIENTA_LARGO = COLA_FRAC > 0 or calienta_largo.strip() not in ("", "0")


def maximo(seq_len: int) -> int:
    return SEQ_MAX or seq_len


def descripcion() -> str:
    return (f"qf2_cola: seq_max={SEQ_MAX or '-'} cargador_largo={int(CARGADOR_LARGO)} "
            f"cola_frac={COLA_FRAC} calienta_largo={int(CALIENTA_LARGO)} paso_calienta={PASO_CALIENTA}")


class Cola:
    def __init__(self, seq_len: int, device_batch_size: int, total_batch_size: int, world_size: int,
                 sequence_max: int, avisar=print):
        if COLA_FRAC and not 0.0 < COLA_FRAC < 1.0:
            raise ValueError(f"QF2_COLA_FRAC must be in (0, 1), got {COLA_FRAC}")
        if CALIENTA_LARGO and not CARGADOR_LARGO:
            raise ValueError("QF2_COLA_FRAC and QF2_CALIENTA_LARGO require QF2_CARGADOR_LARGO=1")
        if sequence_max < seq_len:
            raise ValueError(f"QF2_SEQ_MAX={sequence_max} is below --seq-len={seq_len}")
        if CALIENTA_LARGO and sequence_max < 2 * seq_len:
            raise ValueError(f"long rows of {2 * seq_len} need QF2_SEQ_MAX >= {2 * seq_len}, got {sequence_max}")
        self.filas_partidas = device_batch_size
        self.filas_largas_micro = device_batch_size // 2 if device_batch_size % 2 == 0 else device_batch_size
        tokens_largos = self.filas_largas_micro * 2 * seq_len * world_size
        if CARGADOR_LARGO and total_batch_size % tokens_largos:
            raise ValueError(f"total_batch_size={total_batch_size} must divide long micro tokens={tokens_largos}")
        self.seq_len = seq_len
        self.filas_largas = total_batch_size // (2 * seq_len * world_size) if CARGADOR_LARGO else 0
        self.micro_partido = 2 * self.filas_largas // device_batch_size
        self.micro_largo = self.filas_largas // self.filas_largas_micro
        self.forma = None
        self.avisar = avisar

    def cargador(self, filas: int, seq_len: int) -> tuple[int, int]:
        return (self.filas_largas, 2 * self.seq_len) if CARGADOR_LARGO else (filas, seq_len)

    def es_larga(self, step: int, progreso: float) -> bool:
        calienta = CALIENTA_LARGO and step == PASO_CALIENTA
        return calienta or (COLA_FRAC > 0 and progreso >= 1.0 - COLA_FRAC)

    def pasos_compilacion(self) -> tuple[int, ...]:
        return (0, PASO_CALIENTA) if CALIENTA_LARGO else (0,)

    def lote(self, x, y, step: int, progreso: float, micro_kit: int):
        if not CARGADOR_LARGO:
            return x, y, micro_kit, self.filas_partidas
        larga = self.es_larga(step, progreso)
        micro = self.micro_largo if larga else self.micro_partido
        filas = self.filas_largas_micro if larga else self.filas_partidas
        if larga != self.forma:
            self.forma = larga
            self.avisar(f"qf2_cola: step {step} progress {progreso:.6f} form {'long' if larga else 'split'} "
                        f"rows {x.shape[0] * (1 if larga else 2)}x{x.shape[1] // (1 if larga else 2)} "
                        f"micro_rows={filas} grad_accum_steps={micro}")
        if larga:
            return x, y, micro, filas
        return x.reshape(-1, self.seq_len), y.reshape(-1, self.seq_len), micro, filas

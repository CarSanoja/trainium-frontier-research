"""Propiedad de rowgroups de TRAIN compartida por los dos brazos desde cero."""
from __future__ import annotations

CONFIG: tuple[int, int] | None = None


def leer(texto: str) -> tuple[int, int] | None:
    if not texto:
        return None
    partes = texto.split(":")
    if len(partes) != 3 or partes[0] != "grupos" or not all(p.isdigit() for p in partes[1:]):
        raise ValueError("QF2_AGRUPA_FUENTE espera grupos:inicio:fin o vacío")
    inicio, fin = map(int, partes[1:])
    if not 0 <= inicio < fin <= 6500:
        raise ValueError("QF2_AGRUPA_FUENTE sólo admite TRAIN00000-06499")
    return inicio, fin


def shards(respaldo=None):
    return respaldo if CONFIG is None else list(range(*CONFIG))


def grupos_de_rango(rank: int, world: int):
    import prepare
    if CONFIG is None or not 0 <= rank < world:
        raise ValueError("Fuente sin configurar o rango inválido")
    grupos = prepare._row_groups_for_split("train", rank, world, shard_ids=shards())
    if not grupos:
        raise ValueError(f"El rango {rank}/{world} no tiene rowgroups propios")
    return grupos


def documentos_de_rango(rank: int, world: int, registro=None):
    import pyarrow.parquet as pq
    grupos = grupos_de_rango(rank, world)
    epoca = 0
    while True:
        epoca += 1
        if rank == 0:
            print(f"qf2_agrupa_fuente: rango={rank}/{world} grupos={len(grupos)} epoca={epoca} shards={CONFIG}", flush=True)
        for ruta, grupo in grupos:
            documentos = pq.ParquetFile(ruta).read_row_group(grupo, columns=["text"]).column("text").to_pylist()
            for fila, documento in enumerate(documentos):
                if registro is not None:
                    registro.append((epoca, ruta.name, grupo, fila))
                yield documento

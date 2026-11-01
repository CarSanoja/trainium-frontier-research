"""QF2_RECOLOCADO_FORMATO=pares: cada documento del flujo de recolocado se escribe entero y en orden como pares etiquetados.

El documento se parte en tramos consecutivos c1 c2 c3 ... de k palabras (k uniforme en [minimo, maximo] por tramo, con
el rng del hilo de recolocado; el último tramo se queda con las palabras que sobren). Solo se corta en el blanco que hay
entre dos palabras: cada tramo es el texto del documento byte a byte, con sus espacios y saltos internos, y solo pierde
el blanco de los bordes de corte. Se escribe 'Input: c1\\nOutput: c2\\n\\nInput: c3\\nOutput: c4...' (o 'Q: c1\\nA: c2...');
con un número impar de tramos el último par lleva solo la primera etiqueta. QF2_PARES_ETIQUETAS='ambas' alterna
Input/Output y Q/A por documento escrito, empezando por Input/Output. Un documento sin palabras no da nada.

Etiquetas genéricas fijas, admitidas por la organización el 23 oct (docs/bitacora/20261026T032641Z y 20261026T032911Z):
la primera parte es texto del corpus y la segunda su continuación literal. Nada se calcula y nada viene del bundle de CORE.
"""
from __future__ import annotations

import re

ETIQUETAS = {"Input/Output": ("Input: ", "\nOutput: "), "Q/A": ("Q: ", "\nA: ")}
AMBAS = "ambas"
SEPARADOR_PARES = "\n\n"
FORMATOS = ("ejemplos", "pares")
RE_PALABRA = re.compile(r"\S+")


def leer_tramo(texto: str) -> tuple[int, int]:
    partes = (texto or "").strip().split(":")
    if len(partes) != 2 or not all(p.isdigit() for p in partes) or not 1 <= int(partes[0]) <= int(partes[1]):
        raise ValueError(f"QF2_PARES_TRAMO={texto!r}: se espera 'min:max' en palabras con 1 <= min <= max")
    return int(partes[0]), int(partes[1])


def leer_etiquetas(texto: str) -> tuple[str, ...]:
    texto = (texto or "").strip()
    if texto == AMBAS:
        return tuple(ETIQUETAS)
    if texto not in ETIQUETAS:
        raise ValueError(f"QF2_PARES_ETIQUETAS={texto!r} fuera de {(*ETIQUETAS, AMBAS)}")
    return (texto,)


def configurar(formato: str, ce: str, rafaga, tramo: str, etiquetas: str) -> tuple | None:
    if formato not in FORMATOS:
        raise ValueError(f"QF2_RECOLOCADO_FORMATO={formato!r} fuera de {FORMATOS}")
    if formato != "pares":
        return None
    if ce != "todo" or rafaga is not None:
        raise ValueError(f"QF2_RECOLOCADO_FORMATO=pares necesita QF2_RECOLOCADO_CE=todo y QF2_RECOLOCADO_RAFAGA vacía (ce={ce!r}, rafaga={rafaga!r})")
    return leer_tramo(tramo), leer_etiquetas(etiquetas)


def tramos(doc: str, rng, minimo: int, maximo: int) -> list[str]:
    palabras = [(m.start(), m.end()) for m in RE_PALABRA.finditer(doc)]
    salida: list[str] = []
    i = 0
    while i < len(palabras):
        j = min(len(palabras), i + rng.randint(minimo, maximo))
        salida.append(doc[palabras[i][0]:palabras[j - 1][1]])
        i = j
    return salida


def texto(partes: list[str], etiquetas: str) -> str:
    izq, der = ETIQUETAS[etiquetas]
    pares = [izq + partes[n] + (der + partes[n + 1] if n + 1 < len(partes) else "") for n in range(0, len(partes), 2)]
    return SEPARADOR_PARES.join(pares)


def flujo(documentos, rng, tramo: tuple[int, int], etiquetas: tuple[str, ...]):
    escritos = 0
    for doc in documentos:
        partes = tramos(doc, rng, *tramo)
        if not partes:
            continue
        nombre = etiquetas[escritos % len(etiquetas)]
        escritos += 1
        yield {"tipo": f"pares:{nombre}", "texto": texto(partes, nombre)}

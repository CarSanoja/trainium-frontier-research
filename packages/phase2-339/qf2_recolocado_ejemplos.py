"""QF2_RECOLOCADO: ejemplos de texto recolocado, construidos solo con texto del propio documento de entreno.

Cada ejemplo tiene dos partes unidas por etiquetas fijas y genericas. La primera parte es texto del documento (o,
para Brackets, sus caracteres de corchete en el orden original) y la segunda es la continuacion literal del
documento justo despues. Nada se calcula: no se invierte, ordena ni cuenta. Regla de la organizacion (guangtai,
26 oct 2026): pares de sustantivos genericos admitidos cuando la primera parte es texto del corpus y la segunda su
continuacion. No se usa ninguna etiqueta de plantilla de tarea.

QF2_RECOLOCADO_RAFAGA (documentos_de con rafaga=(minimo, maximo)): por cada documento fuente y cada tipo se sortea k
uniforme en [minimo, maximo] con el mismo rng y se buscan k ejemplos de ese tipo sobre tramos [i, k) que no se solapen,
como mucho INTENTOS_POR_EJEMPLO * k llamadas al constructor. Los hallados se ordenan por su posicion en el documento y se
unen con SEPARADOR en un solo documento recolocado; con uno solo queda el ejemplo suelto de siempre y con ninguno, nada.
fronteras() da los prefijos de texto que delimitan, en cada ejemplo, el contexto (hasta la segunda etiqueta) y el final
de su continuacion: el empaquetador los convierte en tramos de tokens con objetivo -1.
"""
from __future__ import annotations

import re

ETIQUETAS = {
    "corchetes": ("Brackets: ", "\nClosers: "),
    "serie": ("Series: ", "\nRemainder:"),
    "tramo": ("String: ", "\nTail:"),
}
TIPOS = tuple(ETIQUETAS)
CORCHETES = "()[]{}<>"
RE_CORCHETE = re.compile(r"[()\[\]{}<>]")
_ELEM = r"[\w$%.\-']{1,24}"
RE_SERIE = re.compile(rf"(?<![\w$%.\-']){_ELEM}(?:, {_ELEM}){{5,59}}(?![\w$%.\-'])")
RE_INICIO_PALABRA = re.compile(r"(?<=\s)\S")
LARGO_MIN_DOC = 600
SEPARADOR = "\n"
INTENTOS_POR_EJEMPLO = 3


def texto(tipo: str, a: str, b: str) -> str:
    izq, der = ETIQUETAS[tipo]
    return izq + a + der + b


def hasta_segunda_etiqueta(e: dict) -> str:
    return texto(e["tipo"], e["a"], "")


def corchetes(doc: str, rng) -> dict | None:
    pos = [m.start() for m in RE_CORCHETE.finditer(doc)]
    n1, n2 = rng.randint(5, 32), rng.randint(4, 16)
    if len(pos) < n1 + n2:
        return None
    for _ in range(8):
        t = rng.randrange(0, len(pos) - n1 - n2 + 1)
        if pos[t + n1 + n2 - 1] - pos[t + n1 - 1] <= 1500:
            i, j = pos[t], pos[t + n1 - 1] + 1
            a = "".join(c for c in doc[i:j] if c in CORCHETES)
            b = "".join(doc[p] for p in pos[t + n1:t + n1 + n2])
            return {"tipo": "corchetes", "i": i, "j": j, "k": pos[t + n1 + n2 - 1] + 1, "a": a, "b": b}
    return None


def serie(doc: str, rng) -> dict | None:
    corridas = [m for m in RE_SERIE.finditer(doc) if m.end() - m.start() <= 1200]
    if not corridas:
        return None
    m = rng.choice(corridas)
    elementos = m.group(0).split(", ")
    t = rng.randint(3, len(elementos) - 3)
    p = m.start() + sum(len(e) + 2 for e in elementos[:t]) - 1
    return {"tipo": "serie", "i": m.start(), "j": p, "k": m.end(), "a": doc[m.start():p], "b": doc[p:m.end()]}


def tramo(doc: str, rng) -> dict | None:
    l1, l2 = rng.randint(150, 500), rng.randint(100, 400)
    if len(doc) < max(LARGO_MIN_DOC, l1 + l2 + 200):
        return None
    m = RE_INICIO_PALABRA.search(doc, rng.randrange(0, len(doc) - l1 - l2 - 100))
    if m is None:
        return None
    i = m.start()
    j = doc.find(" ", i + l1)
    if j < 0 or j - i > l1 + 60:
        return None
    fin = re.compile(r"\s").search(doc, j + l2)
    k = fin.start() if fin else len(doc)
    if k - j > l2 + 60:
        return None
    return {"tipo": "tramo", "i": i, "j": j, "k": k, "a": doc[i:j], "b": doc[j:k]}


CONSTRUCTORES = {"corchetes": corchetes, "serie": serie, "tramo": tramo}


def ejemplos_de(doc: str, rng) -> list[dict]:
    salida = []
    for tipo in TIPOS:
        e = CONSTRUCTORES[tipo](doc, rng)
        if e is not None:
            e["texto"] = texto(tipo, e["a"], e["b"])
            salida.append(e)
    return salida


def rafaga_de(doc: str, tipo: str, rng, minimo: int, maximo: int) -> dict | None:
    k = rng.randint(minimo, maximo)
    elegidos: list[dict] = []
    for _ in range(INTENTOS_POR_EJEMPLO * k):
        if len(elegidos) == k:
            break
        e = CONSTRUCTORES[tipo](doc, rng)
        if e is not None and all(e["k"] <= o["i"] or o["k"] <= e["i"] for o in elegidos):
            elegidos.append(e)
    elegidos.sort(key=lambda e: e["i"])
    for e in elegidos:
        e["texto"] = texto(tipo, e["a"], e["b"])
    if len(elegidos) < 2:
        return elegidos[0] if elegidos else None
    return {"tipo": tipo, "ejemplos": elegidos, "texto": SEPARADOR.join(e["texto"] for e in elegidos)}


def documentos_de(doc: str, rng, rafaga: tuple[int, int] | None = None) -> list[dict]:
    if rafaga is None:
        return ejemplos_de(doc, rng)
    salida = []
    for tipo in TIPOS:
        d = rafaga_de(doc, tipo, rng, *rafaga)
        if d is not None:
            salida.append(d)
    return salida


def fronteras(d: dict) -> list[str]:
    if "ejemplos" not in d:
        return [hasta_segunda_etiqueta(d)]
    salida: list[str] = []
    previo = ""
    for n, e in enumerate(d["ejemplos"]):
        if n:
            salida.append(previo)
            previo += SEPARADOR
        salida.append(previo + hasta_segunda_etiqueta(e))
        previo += e["texto"]
    return salida

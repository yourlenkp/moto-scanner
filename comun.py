import re
import unicodedata
from difflib import SequenceMatcher

from catalogo import catalogo

GENERICOS = {"disco", "escape", "freno", "motor", "rpm", "volante"}
NEGACIONES = {"ni", "no", "nunca", "sin", "tampoco"}
RELLENO = {
    "al",
    "algo",
    "busca",
    "buscame",
    "cual",
    "cuales",
    "de",
    "del",
    "dime",
    "donde",
    "el",
    "ensename",
    "esa",
    "ese",
    "eso",
    "esta",
    "este",
    "esto",
    "favor",
    "foto",
    "hay",
    "identificar",
    "imagen",
    "informacion",
    "la",
    "las",
    "los",
    "mi",
    "mis",
    "mostrar",
    "mostrarme",
    "muestra",
    "muestrame",
    "o",
    "parte",
    "pieza",
    "por",
    "puedes",
    "que",
    "quiero",
    "reconocer",
    "sobre",
    "su",
    "sus",
    "tu",
    "tus",
    "un",
    "una",
    "unas",
    "unos",
    "ver",
    "y",
}


def normalizar(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = "".join(c if c.isalnum() or c.isspace() else " " for c in texto)
    return " ".join(texto.split())


nombres = {}
for clave, ficha in catalogo.items():
    posibles = [clave.replace("_", " "), ficha["nombre"]] + ficha.get("alias", [])
    for nombre in posibles:
        nombres[normalizar(nombre)] = clave


def buscar(frase):
    if not frase:
        return None
    frase = normalizar(frase)
    if not frase or NEGACIONES.intersection(frase.split()):
        return None
    frases = sorted(
        (nombre for nombre in nombres if " " in nombre), key=len, reverse=True
    )
    for nombre in frases:
        if re.search(rf"(?<!\w){re.escape(nombre)}(?!\w)", frase):
            return nombres[nombre]
    palabras = sorted(
        (nombre for nombre in nombres if " " not in nombre), key=len, reverse=True
    )
    for nombre in palabras:
        if nombre in GENERICOS:
            continue
        if re.search(rf"(?<!\w){re.escape(nombre)}(?!\w)", frase):
            return nombres[nombre]
    utiles = [palabra for palabra in frase.split() if palabra not in RELLENO]
    if len(utiles) == 1 and utiles[0] in nombres:
        return nombres[utiles[0]]
    for nombre in sorted(nombres, key=len, reverse=True):
        if len(nombre) < 5 or abs(len(nombre) - len(frase)) > 4:
            continue
        if SequenceMatcher(None, frase, nombre).ratio() >= 0.9:
            return nombres[nombre]
    return None


def videos(ficha):
    vistos = []
    for campo, valor in ficha.items():
        if campo.startswith("video") and isinstance(valor, str) and valor not in vistos:
            vistos.append(valor)
    return vistos

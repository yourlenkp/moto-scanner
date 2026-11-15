import unicodedata
from difflib import get_close_matches

from catalogo import catalogo


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
    for nombre in sorted(nombres, key=len, reverse=True):
        if nombre in frase:
            return nombres[nombre]
    for palabra in frase.split():
        cerca = get_close_matches(palabra, nombres.keys(), n=1, cutoff=0.8)
        if cerca:
            return nombres[cerca[0]]
    return None


def videos(ficha):
    vistos = []
    for campo, valor in ficha.items():
        if campo.startswith("video") and isinstance(valor, str) and valor not in vistos:
            vistos.append(valor)
    return vistos

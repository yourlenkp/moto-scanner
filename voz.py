import os

import speech_recognition as sr

from catalogo import AVISO, catalogo
from comun import buscar, videos


def mostrar(clave):
    ficha = catalogo[clave]
    print(f"\nPieza: {ficha['nombre']}")
    print(f"Sistema: {ficha['sistema']}")
    print(f"Función: {ficha['funcion']}")
    print(f"Mantenimiento: {ficha['mantenimiento']}")
    enlaces = videos(ficha)
    for i, enlace in enumerate(enlaces, 1):
        etiqueta = "Video" if len(enlaces) == 1 else f"Video {i}"
        print(f"{etiqueta}: {enlace}")
    print(f"Aviso: {AVISO}")


def main():
    print("Micrófonos disponibles:")
    try:
        for i, nombre in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  {i}: {nombre}")
    except Exception as error:
        print(f"  No pude listar micrófonos: {error}")

    indice = os.getenv("MIC_INDEX")
    opciones = {"device_index": int(indice)} if indice else {}
    reconocedor = sr.Recognizer()

    try:
        with sr.Microphone(**opciones) as micro:
            print("Ajustando ruido ambiental...")
            reconocedor.adjust_for_ambient_noise(micro, duration=1)
            print("Di el nombre de una pieza. Di 'salir' para terminar. Ctrl+C también sirve.")
            while True:
                print("\nEscuchando...")
                try:
                    audio = reconocedor.listen(micro, timeout=8, phrase_time_limit=6)
                    texto = reconocedor.recognize_google(audio, language="es-CO")
                except sr.WaitTimeoutError:
                    print("No te oí. Intenta otra vez.")
                    continue
                except sr.UnknownValueError:
                    print("No te entendí. Intenta otra vez.")
                    continue
                except sr.RequestError as error:
                    print(f"No pude conectar con Google: {error}")
                    print("Revisa el internet o usa la app web, que reconoce la voz sin Google.")
                    break

                print(f"Entendí: {texto}")
                if "salir" in texto.lower():
                    break
                clave = buscar(texto)
                if clave:
                    mostrar(clave)
                else:
                    print("No encontré esa pieza. Prueba con: manillar, disco de freno, pistón o basculante.")
    except OSError as error:
        print(f"No pude abrir el micrófono: {error}")
        print("Revisa que esté conectado y que el volumen no esté en cero.")
    except KeyboardInterrupt:
        print("\nHasta luego.")


if __name__ == "__main__":
    main()

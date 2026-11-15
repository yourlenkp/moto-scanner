import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from catalogo import catalogo, AVISO

# 1. Cargar CLIP (ya lo descargaste, ahora arranca más rápido)
modelo = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
procesador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Las mismas descripciones que usaste en probar_clip.py
descripciones = {
    "piston": "a photo of a motorcycle engine piston with connecting rod",
    "disco_freno": "a photo of a motorcycle brake disc",
    "manillar": "a photo of a motorcycle handlebar",
    "basculante": "a photo of a motorcycle swingarm",
}
claves = list(descripciones.keys())
textos = list(descripciones.values())

# 3. Función: recibe una imagen y devuelve la pieza y el porcentaje
def reconocer(imagen):
    entradas = procesador(text=textos, images=imagen,
                          return_tensors="pt", padding=True)
    with torch.no_grad():
        salida = modelo(**entradas)
    prob = salida.logits_per_image.softmax(dim=1)[0]
    mejor = prob.argmax().item()
    return claves[mejor], prob[mejor].item() * 100

# 4. Abrir la cámara
camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

print("Pulsa ESPACIO para reconocer la pieza. Pulsa Q para salir.")

while True:
    ok, cuadro = camara.read()
    if not ok:
        print("No se pudo leer la cámara.")
        break
    cv2.imshow("Camara - ESPACIO: reconocer, Q: salir", cuadro)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == 32:  # 32 es la tecla ESPACIO
        rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
        clave, porcentaje = reconocer(Image.fromarray(rgb))
        if porcentaje < 60:
            print(f"\nNo estoy seguro (mejor opción: {clave}, {porcentaje:.0f}%). Acerca la pieza e inténtalo otra vez.")
        else:
            ficha = catalogo[clave]
            print(f"\nPieza: {ficha['nombre']} ({porcentaje:.0f}%)")
            print(f"Sistema: {ficha['sistema']}")
            print(f"Función: {ficha['funcion']}")
            print(f"Mantenimiento: {ficha['mantenimiento']}")
            for campo, enlace in ficha.items():
                if campo.startswith("video"):
                   print(f"{campo.replace('_', ' ').capitalize()}: {enlace}")
            print(f"Aviso: {AVISO}")

camara.release()
cv2.destroyAllWindows()
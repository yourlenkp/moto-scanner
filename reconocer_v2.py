import os
import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from catalogo import catalogo, AVISO

UMBRAL = 50       # porcentaje mínimo para dar una respuesta
PESO_REF = 0.5    # 0.5 = mitad texto, mitad fotos de referencia
ESCALA_TEXTO = 100
ESCALA_REF = 60   # estimación mía, se puede ajustar probando

modelo = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
procesador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

descripciones = {
    "piston": "a photo of a motorcycle engine piston with connecting rod",
    "disco_freno": "a photo of a motorcycle brake disc",
    "manillar": "a photo of a motorcycle handlebar",
    "basculante": "a photo of a motorcycle swingarm",
}
claves = list(descripciones.keys())
textos = list(descripciones.values())

def huellas(imagen):
    """Devuelve la huella de la imagen y la de cada descripción."""
    entradas = procesador(text=textos, images=imagen,
                          return_tensors="pt", padding=True)
    with torch.no_grad():
        salida = modelo(**entradas)
    img = salida.image_embeds[0]
    img = img / img.norm()
    txt = salida.text_embeds
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return img, txt

# Cargar las fotos de referencia de fotos_prueba (una sola vez al arrancar)
referencias = {c: [] for c in claves}
for clave in claves:
    ruta = os.path.join("fotos_prueba", clave)
    for archivo in os.listdir(ruta):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            foto = Image.open(os.path.join(ruta, archivo)).convert("RGB")
            referencias[clave].append(huellas(foto)[0])
    print(f"{clave}: {len(referencias[clave])} fotos de referencia")

def reconocer(imagen):
    img, txt = huellas(imagen)
    sim_texto = torch.stack([img @ txt[i] for i in range(len(claves))])
    p_texto = (sim_texto * ESCALA_TEXTO).softmax(dim=0)

    sim_ref = []
    for clave in claves:
        sims = sorted([float(img @ r) for r in referencias[clave]], reverse=True)
        top = sims[:3]
        sim_ref.append(sum(top) / len(top))
    p_ref = (torch.tensor(sim_ref) * ESCALA_REF).softmax(dim=0)

    prob = PESO_REF * p_ref + (1 - PESO_REF) * p_texto
    mejor = prob.argmax().item()
    return claves[mejor], prob[mejor].item() * 100

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
    if tecla == 32:
        rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
        clave, porcentaje = reconocer(Image.fromarray(rgb))
        if porcentaje < UMBRAL:
            print(f"\nNo estoy seguro (mejor opción: {clave}, {porcentaje:.0f}%). Acerca la pieza e inténtalo otra vez.")
        else:
            ficha = catalogo[clave]
            print(f"\nPieza: {ficha['nombre']} ({porcentaje:.0f}%)")
            print(f"Sistema: {ficha['sistema']}")
            print(f"Función: {ficha['funcion']}")
            print(f"Mantenimiento: {ficha['mantenimiento']}")
            print(f"Video: {ficha['video']}")
            print(f"Aviso: {AVISO}")

camara.release()
cv2.destroyAllWindows()
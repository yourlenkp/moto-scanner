import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. Cargar CLIP (la primera vez lo descarga de internet)
modelo = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
procesador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Cada carpeta con su descripción en texto
piezas = {
    "piston": "a photo of a motorcycle engine piston with connecting rod",
    "disco_freno": "a photo of a motorcycle brake disc",
    "manillar": "a photo of a motorcycle handlebar",
    "basculante": "a photo of a motorcycle swingarm",
}
carpetas = list(piezas.keys())
textos = list(piezas.values())

# 3. Probar cada foto y contar aciertos
aciertos = 0
total = 0
for carpeta in carpetas:
    ruta = os.path.join("fotos_prueba", carpeta)
    for archivo in os.listdir(ruta):
        if not archivo.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        imagen = Image.open(os.path.join(ruta, archivo)).convert("RGB")
        entradas = procesador(text=textos, images=imagen,
                              return_tensors="pt", padding=True)
        with torch.no_grad():
            salida = modelo(**entradas)
        probabilidades = salida.logits_per_image.softmax(dim=1)[0]
        mejor = probabilidades.argmax().item()
        porcentaje = probabilidades[mejor].item() * 100
        acerto = carpetas[mejor] == carpeta
        aciertos += acerto
        total += 1
        print(f"{archivo}: dijo {carpetas[mejor]} ({porcentaje:.0f}%) - "
              f"{'OK' if acerto else 'FALLO, era ' + carpeta}")

print(f"\nAciertos: {aciertos} de {total}")
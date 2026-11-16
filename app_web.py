import importlib
import io
import os
import tempfile
from pathlib import Path

import qrcode
import streamlit as st
import torch
from PIL import Image, ImageOps
from faster_whisper import WhisperModel
from transformers import CLIPModel, CLIPProcessor

import catalogo as modulo_catalogo
import comun as modulo_comun

importlib.reload(modulo_catalogo)
importlib.reload(modulo_comun)

from catalogo import AVISO, catalogo
from comun import buscar, videos_etiquetados

BASE = Path(__file__).resolve().parent
MODELO_CLIP = "openai/clip-vit-base-patch32"
CLAVES = list(catalogo.keys())
ALPHA_REFERENCIA = 20.0
TOP_K_REFERENCIA = 3
UMBRAL_SIMILITUD_DESCONOCIDO = 0.65
UMBRAL_MARGEN_AMBIGUO = 0.10
UMBRAL_CONFIANZA_AMBIGUO = 0.50
UMBRAL_SIMILITUD_AMBIGUO = 0.80
ETIQUETAS_VIDEO = {
    "video": "Ver video",
    "video_instalacion": "Ver video de instalación",
}
CREDITOS = [
    "Manillar: Alexa Wirth, CC0.",
    "Disco de freno: Aprilia disc brake (dominio público); Ildar Sagdejev, CC BY-SA 4.0; Brianhe, CC BY-SA 4.0; Malsa, CC BY-SA 1.0.",
    "Pistón: TuckerBag (dominio público); Honda CB77 Restoration, CC BY 2.0.",
    "Basculante: A7N8X / Auge=mit, CC BY-SA 3.0.",
    "Motor: Yanko Malinov, CC BY-SA 4.0; Paul The Writer, CC0; Khaosaming, CC BY-SA 3.0.",
    "Amortiguador: Dennis Bratland, CC BY-SA 4.0; Andy Carter, CC BY 2.0.",
    "Tubo de escape: Microrao, CC BY-SA 4.0; Franken-Guzzista, CC BY-SA 4.0.",
    "Tacómetro: Piero (Wikipedia), dominio público; Centovalli, CC BY-SA 3.0; Ildar Sagdejev, CC BY-SA 4.0.",
    "Referencias adicionales aportadas por el usuario.",
    "Licencias: creativecommons.org/licenses (CC0, CC BY y CC BY-SA).",
]

DESCRIPCIONES = {
    "manillar": [
        "a close-up of a motorcycle handlebar with grips and controls",
        "the metal steering bar of a motorcycle",
        "motorcycle handlebars and mirrors",
    ],
    "disco_freno": [
        "a close-up of a motorcycle brake disc rotor",
        "a shiny round metal brake disk with drilled holes",
        "a motorcycle front brake rotor mounted on the wheel",
        "a metal brake disc next to the brake caliper",
    ],
    "piston": [
        "a motorcycle engine piston with piston rings",
        "a close-up of a metal piston inside a cylinder",
        "a shiny cylindrical engine piston",
    ],
    "basculante": [
        "a motorcycle swingarm holding the rear wheel",
        "the rear metal arm of a motorcycle chassis",
        "a close-up of a motorcycle swingarm with chain and sprocket",
    ],
    "motor": [
        "a motorcycle engine block",
        "a close-up of a motorcycle engine with cylinders and cooling fins",
        "the engine of a motorcycle",
    ],
    "amortiguador": [
        "a motorcycle rear shock absorber",
        "a close-up of a motorcycle suspension spring and shock",
        "a motorcycle suspension damper",
    ],
    "tubo_escape": [
        "a motorcycle exhaust pipe and muffler",
        "a close-up of a metal motorcycle exhaust",
        "the exhaust system of a motorcycle",
    ],
    "tacometro": [
        "a motorcycle tachometer gauge showing RPM",
        "a close-up of a motorcycle dashboard with a tachometer",
        "a round motorcycle gauge with a needle",
    ],
}


def _features(salida):
    return salida.pooler_output if hasattr(salida, "pooler_output") else salida


@st.cache_resource(show_spinner="Cargando CLIP...")
def cargar_clip():
    faltantes = [clave for clave in CLAVES if not DESCRIPCIONES.get(clave)]
    if faltantes:
        raise ValueError("Faltan descripciones para: " + ", ".join(faltantes))
    torch.set_num_threads(2)
    modelo = CLIPModel.from_pretrained(MODELO_CLIP)
    procesador = CLIPProcessor.from_pretrained(MODELO_CLIP)
    modelo.eval()
    with torch.no_grad():
        textos = []
        for clave in CLAVES:
            ent = procesador(text=DESCRIPCIONES[clave], return_tensors="pt", padding=True)
            emb = _features(modelo.get_text_features(**ent))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0, keepdim=True)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            textos.append(emb)
        texto = torch.cat(textos, dim=0)
        referencias = []
        errores = []
        sin_referencias = []
        for clave in CLAVES:
            rutas = []
            carpeta = BASE / "imagenes_moto" / "referencias" / clave
            if carpeta.exists():
                for ruta in sorted(carpeta.iterdir()):
                    if ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        rutas.append(ruta)
            embs = []
            for ruta in rutas:
                try:
                    with Image.open(ruta) as original:
                        imagen = ImageOps.exif_transpose(original).convert("RGB")
                except Exception as error:
                    errores.append(f"{ruta.name}: {error}")
                    continue
                for img in (imagen, ImageOps.mirror(imagen)):
                    ent = procesador(images=img, return_tensors="pt")
                    emb = _features(modelo.get_image_features(**ent))
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embs.append(emb)
            if not embs:
                sin_referencias.append(clave)
            referencias.append(embs)
    if errores:
        st.warning("No se pudieron cargar algunas referencias: " + "; ".join(errores))
    if sin_referencias:
        st.warning("Estas piezas no tienen referencias válidas: " + ", ".join(sin_referencias))
    return modelo, procesador, texto, referencias


@st.cache_resource(show_spinner="Cargando Whisper...")
def cargar_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)


@st.cache_data(show_spinner=False)
def clasificar_imagen_bytes(datos):
    modelo, procesador, texto, referencias = cargar_clip()
    with Image.open(io.BytesIO(datos)) as original:
        imagen = ImageOps.exif_transpose(original).convert("RGB")
    ancho, alto = imagen.size
    recorte = imagen.crop((int(ancho * 0.08), int(alto * 0.08), int(ancho * 0.92), int(alto * 0.92)))
    variantes = [imagen, ImageOps.mirror(imagen), recorte]
    with torch.no_grad():
        embs = []
        for variante in variantes:
            ent = procesador(images=variante, return_tensors="pt")
            emb = _features(modelo.get_image_features(**ent))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb)
        imagen_emb = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
        imagen_emb = imagen_emb / imagen_emb.norm(dim=-1, keepdim=True)
        prob_texto = (modelo.logit_scale.exp() * (imagen_emb @ texto.T)).softmax(dim=-1)[0]
        prob = prob_texto.clone()
        similitud_maxima = None
        indices = [i for i, refs in enumerate(referencias) if refs]
        if indices:
            sims = []
            for i in indices:
                similitudes = (imagen_emb @ torch.cat(referencias[i], dim=0).T)[0]
                k = min(TOP_K_REFERENCIA, similitudes.numel())
                sims.append(torch.topk(similitudes, k).values.mean())
            sims = torch.stack(sims)
            similitud_maxima = float(sims.max())
            factores = torch.ones(len(CLAVES))
            for pos, i in enumerate(indices):
                factores[i] = torch.exp(ALPHA_REFERENCIA * (sims[pos] - similitud_maxima))
            prob = prob_texto * factores
            prob = prob / prob.sum()
    orden = torch.argsort(prob, descending=True)
    top = [(CLAVES[int(i)], float(prob[int(i)])) for i in orden[:3]]
    confianza = float(prob[int(orden[0])])
    if similitud_maxima is not None:
        margen = confianza - float(prob[int(orden[1])])
        if similitud_maxima < UMBRAL_SIMILITUD_DESCONOCIDO or (
            margen < UMBRAL_MARGEN_AMBIGUO
            and confianza < UMBRAL_CONFIANZA_AMBIGUO
            and similitud_maxima < UMBRAL_SIMILITUD_AMBIGUO
        ):
            return None, confianza, top
    return CLAVES[int(orden[0])], confianza, top


@st.cache_data(show_spinner=False)
def transcribir(datos):
    if not datos:
        return ""
    modelo = cargar_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as archivo:
        archivo.write(datos)
        ruta = archivo.name
    try:
        segmentos, _ = modelo.transcribe(
            ruta,
            language="es",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segmento.text.strip() for segmento in segmentos).strip()
    finally:
        os.unlink(ruta)


@st.cache_data(show_spinner=False)
def qr_png(url):
    imagen = qrcode.make(url)
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


def mostrar_candidatos(top):
    for nombre, probabilidad in top:
        st.write(f"{catalogo[nombre]['nombre']}: {probabilidad * 100:.1f}%")


def mostrar_ficha(clave, confianza=None, top=None):
    ficha = catalogo[clave]
    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        ruta = BASE / ficha["imagen"] if ficha.get("imagen") else None
        if ruta and ruta.exists():
            st.image(str(ruta))
            if ficha.get("credito"):
                st.caption(f"Imagen: {ficha['credito']}")
        else:
            st.info("Imagen de referencia pendiente.")
    with derecha:
        st.subheader(ficha["nombre"])
        if confianza is not None:
            st.metric("Confianza", f"{confianza * 100:.0f}%")
            st.progress(max(0.0, min(confianza, 1.0)))
            if confianza < 0.5:
                st.warning("La app no está segura. Prueba con una foto más cercana o elige la pieza manualmente.")
        st.write(f"**Sistema:** {ficha['sistema']}")
        st.write(f"**Función:** {ficha['funcion']}")
        st.write(f"**Mantenimiento:** {ficha['mantenimiento']}")
        for campo, enlace in videos_etiquetados(ficha):
            etiqueta = ETIQUETAS_VIDEO.get(campo)
            if etiqueta is None:
                detalle = campo.removeprefix("video_").replace("_", " ")
                etiqueta = f"Ver video: {detalle}"
            st.link_button(etiqueta, enlace)
        if top:
            with st.expander("Ver las 3 mejores coincidencias", expanded=confianza is not None and confianza < 0.5):
                mostrar_candidatos(top)
    st.caption(AVISO)


st.set_page_config(page_title="Moto Scanner", layout="wide")
st.title("Moto Scanner")
st.caption("Identifica piezas de moto por cámara o voz y consulta su función y mantenimiento.")

tab_camara, tab_voz, tab_qr = st.tabs(["Cámara", "Voz", "Enlace y QR"])

with tab_camara:
    st.subheader("Reconoce una pieza por foto")
    st.caption("Consejo: llena el encuadre con la pieza, con buena luz y sin otras piezas alrededor.")
    foto = st.camera_input("Toma una foto")
    archivo = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    datos = archivo.getvalue() if archivo else (foto.getvalue() if foto else None)
    if datos:
        st.image(datos, caption="Imagen para analizar")
        if st.button("Reconocer pieza", type="primary"):
            with st.spinner("Analizando la imagen..."):
                try:
                    clave, confianza, top = clasificar_imagen_bytes(datos)
                except Exception as error:
                    st.error(f"No pude analizar la imagen: {error}")
                else:
                    if clave is None:
                        st.warning(
                            "No reconocí una pieza del catálogo. Prueba con una foto más cercana o elige la pieza manualmente."
                        )
                        with st.expander("Ver las 3 mejores coincidencias", expanded=True):
                            mostrar_candidatos(top)
                    else:
                        mostrar_ficha(clave, confianza, top)
    st.divider()
    seleccion = st.selectbox(
        "O elige la pieza manualmente",
        ["—"] + [catalogo[clave]["nombre"] for clave in CLAVES],
        key="camara_manual",
    )
    if seleccion != "—":
        clave = next(clave for clave in CLAVES if catalogo[clave]["nombre"] == seleccion)
        mostrar_ficha(clave)

with tab_voz:
    st.subheader("Reconoce una pieza por voz")
    audio = st.audio_input("Graba el nombre de la pieza", sample_rate=16000)
    if audio:
        with st.spinner("Transcribiendo el audio..."):
            try:
                texto = transcribir(audio.getvalue())
            except Exception as error:
                texto = ""
                st.error(f"No pude transcribir el audio: {error}")
        if texto:
            st.write(f"Entendí: **{texto}**")
            clave = buscar(texto)
            if clave:
                mostrar_ficha(clave)
            else:
                st.warning("No reconocí la pieza. Prueba decir: manillar, disco de freno, pistón, basculante, motor, amortiguador, escape o tacómetro.")
        else:
            st.info("No se detectó voz. Intenta grabar de nuevo.")
    st.divider()
    seleccion = st.selectbox(
        "O elige la pieza manualmente",
        ["—"] + [catalogo[clave]["nombre"] for clave in CLAVES],
        key="voz_manual",
    )
    if seleccion != "—":
        clave = next(clave for clave in CLAVES if catalogo[clave]["nombre"] == seleccion)
        mostrar_ficha(clave)

with tab_qr:
    st.subheader("Comparte la app con un QR")
    try:
        url = st.context.url
    except Exception:
        url = ""
    if url:
        st.write("Abre la app desde otro teléfono escaneando este código:")
        qr = qr_png(url)
        izquierda, derecha = st.columns([1, 2])
        with izquierda:
            st.image(qr, width=260)
        with derecha:
            st.code(url, language=None)
            st.download_button("Descargar QR", qr, "qr_moto_scanner.png", "image/png")
            st.link_button("Abrir la app", url)
        if "localhost" in url or "127.0.0.1" in url:
            st.info("Este QR funciona en tu red local. Cuando la app esté publicada, se generará con el enlace público.")
    else:
        st.info("El enlace aparecerá cuando la app esté publicada.")
    st.caption("También puedes compartir el enlace directo desde la barra del navegador.")

with st.sidebar:
    st.header("Piezas disponibles")
    for clave in CLAVES:
        st.write(f"- {catalogo[clave]['nombre']}")
    st.divider()
    with st.expander("¿Cómo mejorar el reconocimiento?"):
        st.write("Acerca la cámara para que la pieza ocupe la mayor parte de la foto.")
        st.write("Usa buena luz y evita que aparezcan otras piezas alrededor.")
        st.write("Con más fotos reales de referencia, el porcentaje de confianza sube.")
        st.write("El porcentaje es la probabilidad del modelo; si es bajo, usa el selector manual.")
    st.divider()
    st.write("La primera vez que uses la cámara o la voz, la app descarga los modelos. Puede tardar un poco.")
    st.divider()
    with st.expander("Créditos de imágenes"):
        for credito in CREDITOS:
            st.write(credito)
    st.caption("Fichas 1–8 ilustrativas; referencias de Wikimedia Commons y aportadas por el usuario.")

import io
import os
import tempfile
from pathlib import Path

import qrcode
import streamlit as st
import torch
from PIL import Image
from faster_whisper import WhisperModel
from transformers import CLIPModel, CLIPProcessor

from catalogo import AVISO, catalogo
from comun import buscar, videos

BASE = Path(__file__).resolve().parent
MODELO_CLIP = "openai/clip-vit-base-patch32"
CLAVES = list(catalogo.keys())

DESCRIPCIONES = {
    "manillar": "a motorcycle handlebar, the steering bar with grips and controls",
    "disco_freno": "a motorcycle brake disc rotor, a round metal brake disk",
    "piston": "a motorcycle engine piston, a metal piston with rings",
    "basculante": "a motorcycle swingarm, the rear suspension arm holding the wheel",
}
TEXTOS = [DESCRIPCIONES[clave] for clave in CLAVES]


def _features(salida):
    return salida.pooler_output if hasattr(salida, "pooler_output") else salida


@st.cache_resource(show_spinner="Cargando CLIP...")
def cargar_clip():
    torch.set_num_threads(2)
    modelo = CLIPModel.from_pretrained(MODELO_CLIP)
    procesador = CLIPProcessor.from_pretrained(MODELO_CLIP)
    modelo.eval()
    with torch.no_grad():
        entradas = procesador(text=TEXTOS, return_tensors="pt", padding=True)
        texto = _features(modelo.get_text_features(**entradas))
        texto = texto / texto.norm(dim=-1, keepdim=True)
        referencias = []
        for clave in CLAVES:
            ruta = BASE / catalogo[clave]["imagen"]
            if ruta.exists():
                imagen = Image.open(ruta).convert("RGB")
                ent = procesador(images=imagen, return_tensors="pt")
                referencia = _features(modelo.get_image_features(**ent))
                referencia = referencia / referencia.norm(dim=-1, keepdim=True)
                referencias.append(referencia)
            else:
                referencias.append(None)
    return modelo, procesador, texto, referencias


@st.cache_resource(show_spinner="Cargando Whisper...")
def cargar_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")


@st.cache_data(show_spinner=False)
def clasificar_imagen_bytes(datos):
    modelo, procesador, texto, referencias = cargar_clip()
    imagen = Image.open(io.BytesIO(datos)).convert("RGB")
    with torch.no_grad():
        ent = procesador(images=imagen, return_tensors="pt")
        imagen_emb = _features(modelo.get_image_features(**ent))
        imagen_emb = imagen_emb / imagen_emb.norm(dim=-1, keepdim=True)
        prob_texto = (modelo.logit_scale.exp() * (imagen_emb @ texto.T)).softmax(dim=-1)[0]
        if any(referencia is not None for referencia in referencias):
            similitudes = []
            for referencia in referencias:
                if referencia is None:
                    similitudes.append(torch.tensor(0.0))
                else:
                    similitudes.append((imagen_emb @ referencia.T).max())
            prob_ref = (torch.stack(similitudes) * 60).softmax(dim=0)
            prob = 0.5 * prob_ref + 0.5 * prob_texto
        else:
            prob = prob_texto
    orden = torch.argsort(prob, descending=True)
    top = [(CLAVES[int(i)], float(prob[int(i)])) for i in orden[:3]]
    return CLAVES[int(orden[0])], float(prob[int(orden[0])]), top


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


def mostrar_ficha(clave, confianza=None, top=None):
    ficha = catalogo[clave]
    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        ruta = BASE / ficha["imagen"]
        if ruta.exists():
            st.image(str(ruta))
    with derecha:
        st.subheader(ficha["nombre"])
        if confianza is not None:
            st.metric("Confianza", f"{confianza:.0f}%")
            st.progress(max(0.0, min(confianza / 100, 1.0)))
        st.write(f"**Sistema:** {ficha['sistema']}")
        st.write(f"**Función:** {ficha['funcion']}")
        st.write(f"**Mantenimiento:** {ficha['mantenimiento']}")
        enlaces = videos(ficha)
        for i, enlace in enumerate(enlaces, 1):
            etiqueta = "Ver video" if len(enlaces) == 1 else f"Ver video {i}"
            st.link_button(etiqueta, enlace)
        if top and len(top) > 1:
            with st.expander("Otras coincidencias"):
                for nombre, probabilidad in top[1:]:
                    st.write(f"{catalogo[nombre]['nombre']}: {probabilidad * 100:.1f}%")
    st.caption(AVISO)


st.set_page_config(page_title="Moto Scanner", layout="wide")
st.title("Moto Scanner")
st.caption("Identifica piezas de moto por cámara o voz y consulta su función y mantenimiento.")

tab_camara, tab_voz, tab_qr = st.tabs(["Cámara", "Voz", "Enlace y QR"])

with tab_camara:
    st.subheader("Reconoce una pieza por foto")
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
                    if confianza < 50:
                        st.warning("La coincidencia es baja. Prueba con otra foto o elige la pieza manualmente.")
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
                st.warning("No reconocí la pieza. Prueba decir: manillar, disco de freno, pistón o basculante.")
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
    st.write("La primera vez que uses la cámara o la voz, la app descarga los modelos. Puede tardar un poco.")
    st.divider()
    st.caption("Hecho para aprender mecánica básica de moto.")

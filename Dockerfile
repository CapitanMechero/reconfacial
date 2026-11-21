# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Bibliotecas de sistema que necesitan OpenCV/MediaPipe en imágenes slim (obtenidas
# de las dependencias NEEDED de libmediapipe.so y de las librerías que trae la rueda de OpenCV):
#   MediaPipe: libEGL.so.1 (libegl1), libGLESv2.so.2 (libgles2)
#   OpenCV/Qt: libGL, glib, X11 (libx11-6, libxext6, libxcb1, libsm6, libice6)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libgles2 libglib2.0-0 libx11-6 libxext6 libxcb1 libsm6 libice6 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Modelo público de MediaPipe: se descarga y se verifica su SHA-256 en la construcción
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py --dest /opt/facerec/face_landmarker.task

COPY recognition/ recognition/

# Comprobación en la construcción: el detector real se carga y procesa una imagen.
# Si falta alguna biblioteca nativa, falla el build (no la primera foto en la app).
RUN python -c "import numpy as np; from recognition.detector import FaceDetector; \
d = FaceDetector('/opt/facerec/face_landmarker.task'); \
print('detector OK, caras en imagen vacía:', len(d.detect(np.zeros((256, 256, 3), np.uint8))))"
COPY database/ database/
COPY evaluation/ evaluation/
COPY data/ data/
COPY app.py sistema.py ./
COPY .streamlit/ .streamlit/

# Usuario sin privilegios; el volumen de datos hereda estos permisos
# Dos volúmenes separados: datos (BD CSV + SVM) y estado (clave HMAC + manifiesto)
RUN useradd --uid 10001 --create-home facerec \
 && mkdir -p /var/lib/facerec/models /var/lib/facerec-state \
 && chown -R facerec:facerec /var/lib/facerec /var/lib/facerec-state \
 && chmod 700 /var/lib/facerec-state
USER facerec

ENV FACEREC_DATA_DIR=/var/lib/facerec \
    FACEREC_STATE_DIR=/var/lib/facerec-state \
    FACEREC_SEED_DB=/app/data/famosos.csv \
    FACE_LANDMARKER_MODEL=/opt/facerec/face_landmarker.task

VOLUME ["/var/lib/facerec", "/var/lib/facerec-state"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

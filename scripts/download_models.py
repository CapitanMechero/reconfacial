"""Descarga el modelo Face Landmarker de MediaPipe a models/."""
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"Ya existe: {DEST}")
    else:
        urllib.request.urlretrieve(URL, DEST)
        print(f"Descargado en {DEST}")
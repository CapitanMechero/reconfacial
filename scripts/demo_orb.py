"""
Script de demostración del reconocimiento mediante ORB.
Detecta y preprocesa el rostro, extrae sus puntos clave ORB, los compara con las
personas registradas (python -m scripts.register_orb) y muestra la imagen original,
la cara con sus puntos clave y la puntuación de cada identidad frente al umbral T2.

Ejecutar: python -m scripts.demo_orb dataset/imagen.jpg [--t2 0.15]
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from recognition.detector import process_image
from recognition.matching import DEFAULT_T2, match_identity
from recognition.orb import OrbExtractionError, extract_orb
from recognition.orb_store import DEFAULT_DIR, load_database
from recognition.preprocessing import load_image


def main(image_path, t2=DEFAULT_T2, directory=DEFAULT_DIR, detector=None, show=True):
    db = load_database(directory)
    if not db:
        raise SystemExit("No hay personas registradas. Usa: python -m scripts.register_orb")

    original = load_image(image_path)
    face = process_image(original, detector=detector).face

    try:
        template = extract_orb(face)
    except OrbExtractionError as e:
        print("Sin puntos clave suficientes:", e)
        template = None

    result = match_identity(template, db, t2=t2)

    print("Imagen:", image_path)
    print("Puntos clave ORB:", 0 if template is None else len(template))
    for name, s in sorted(result.scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:25s} {s:.3f}")
    print("Resultado:", result.identity or "DESCONOCIDO",
          f"(mejor {result.score:.3f}, margen {result.margin:.3f}, T2 {t2})")

    if not show:
        return result

    # Visualización
    face_kp = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    if template is not None:
        kps = [cv2.KeyPoint(float(x), float(y), float(s), float(a))
               for x, y, s, a, _ in template.keypoints]
        face_kp = cv2.drawKeypoints(face_kp, kps, None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen original")
    ax[1].imshow(face_kp[:, :, ::-1])
    ax[1].set_title("Cara y puntos clave ORB")
    for a in ax[:2]:
        a.axis("off")

    ranked = sorted(result.scores.items(), key=lambda kv: kv[1])[-10:]
    ax[2].barh([n for n, _ in ranked], [s for _, s in ranked])
    ax[2].axvline(t2, color="red", linestyle="--", label=f"T2 = {t2}")
    ax[2].set_title("Puntuación ORB por identidad")
    ax[2].legend()

    fig.suptitle(f"{result.identity or 'DESCONOCIDO'} | mejor: {result.score:.3f}")
    plt.tight_layout()
    plt.show()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--t2", type=float, default=DEFAULT_T2)
    args = parser.parse_args()
    main(args.image, t2=args.t2)

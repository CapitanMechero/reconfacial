"""
Script de demostración del reconocimiento facial mediante HOG y SVM.
Detecta y preprocesa el rostro, extrae sus características HOG y utiliza
el modelo SVM previamente entrenado para predecir su identidad.
Muestra la identidad candidata, su puntuación de decisión y las imágenes
obtenidas durante el procesamiento.

Ejecutar: python -m legacy.integrante2.demo_svm dataset/imagen.jpg
"""

import argparse
import cv2
import matplotlib.pyplot as plt

from recognition.detector import process_image
from recognition.preprocessing import load_image
from recognition.hog import extract_hog
from recognition.svm import predict_identity


def main(image_path):
    original = load_image(image_path)

    # Detección y preprocesamiento
    face = process_image(original).face

    # Extracción HOG
    features, hog_image = extract_hog(face, visualize=True)

    # Predicción SVM
    name, score = predict_identity(features)

    print("Imagen:", image_path)
    print("Características:", len(features))
    print("Identidad candidata:", name)
    print("Puntuación SVM:", score)

    # Visualización
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen original")

    ax[1].imshow(face, cmap="gray")
    ax[1].set_title("Cara detectada")

    ax[2].imshow(hog_image, cmap="gray")
    ax[2].set_title("Descriptor HOG")

    for a in ax:
        a.axis("off")

    fig.suptitle(f"{name} | Score: {score:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    main(args.image)

"""
Script de demostración de la extracción de características HOG.
A partir de una imagen, detecta y preprocesa el rostro mediante MediaPipe,
extrae su descriptor HOG de 8100 características y visualiza la imagen
original, el rostro preprocesado y la representación del descriptor HOG.

Ejecutar: python -m scripts.demo_hog dataset/imagen.jpg
"""
import argparse
import matplotlib.pyplot as plt

from recognition.detector import process_image
from recognition.preprocessing import load_image
from recognition.hog import extract_hog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    # Cargar imagen original
    original = load_image(args.image)

    # Detectar, alinear y normalizar la cara
    face = process_image(original).face

    # Extraer características HOG
    features, hog_image = extract_hog(face, visualize=True)

    print("Imagen original:", original.shape)
    print("Cara procesada:", face.shape)
    print("Vector HOG:", features.shape)
    print("Primeras 20 características:", features[:20])

    # Visualizar las tres etapas
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original[:, :, ::-1])
    axes[0].set_title("Imagen original")

    axes[1].imshow(face, cmap="gray")
    axes[1].set_title("Cara preprocesada")

    axes[2].imshow(hog_image, cmap="gray")
    axes[2].set_title("HOG")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

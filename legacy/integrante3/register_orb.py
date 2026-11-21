"""Registra una persona en la base ORB (equivalente ORB de register.py).

Ejecutar: python -m legacy.integrante3.register_orb "Ana" dataset/ana.jpg [--replace]
"""
import argparse

from recognition.detector import process_image
from recognition.orb import extract_orb
from legacy.integrante3.orb_store import DEFAULT_DIR, save_template


def register(name, image_path, directory=DEFAULT_DIR, detector=None, replace=False):
    # Si la foto no es válida, process_image / extract_orb lanzan un error controlado
    face = process_image(image_path, detector=detector).face
    template = extract_orb(face)
    path = save_template(name, template, directory, replace=replace)

    print("Persona inscrita correctamente")
    print("Nombre:", name.strip())
    print("Imagen:", image_path)
    print("Puntos clave ORB:", len(template))
    print("Guardado en:", path)
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("image")
    parser.add_argument("--replace", action="store_true", help="sobrescribir si ya existe")
    args = parser.parse_args()
    register(args.name, args.image, replace=args.replace)

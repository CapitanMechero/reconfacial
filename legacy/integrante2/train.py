"""Entrenamiento del SVM a partir de un CSV de vectores HOG (flujo offline/legado).

En la aplicación el SVM se reentrena desde la base de datos cifrada mediante
``recognition.model_lifecycle``; este script sirve para experimentos con
``legacy/dataset.csv``. Usa rutas explícitas para poder ejecutarse (y probarse)
sin tocar el modelo del proyecto.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from recognition.svm import DEFAULT_MODEL, save_model, train_svm

CSV_PATH = Path("legacy/dataset.csv")
TEST_SIZE = 0.2
C = 0.1
SEED = 42


def train_from_csv(csv_path=CSV_PATH, model_path=DEFAULT_MODEL, test_size=TEST_SIZE,
                   c=C, seed=SEED, save=True) -> dict:
    """Entrena con el CSV y devuelve las métricas. Guarda el modelo si ``save``."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["name"]).to_numpy(dtype=np.float32)
    y = df["name"].to_numpy()

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
    else:
        X_train, y_train, X_test, y_test = X, y, None, None

    model = train_svm(X_train, y_train, C=c)
    metrics: dict = {"C": c, "train_accuracy": float(model.score(X_train, y_train)),
                     "n_train": int(len(y_train))}
    if X_test is not None:
        y_pred = model.predict(X_test)
        metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["test_f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        metrics["n_test"] = int(len(y_test))
    if save:
        save_model(model, model_path)
        metrics["model_path"] = str(model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--C", type=float, default=C)
    args = parser.parse_args()
    for key, value in train_from_csv(args.csv, args.out, args.test_size, args.C).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

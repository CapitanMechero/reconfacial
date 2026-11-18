"""SVM multiclase y puntuación de decisión (no es una probabilidad)."""

from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from recognition.hog import N_FEATURES


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "svm.joblib"


def train_svm(X, y, C=0.1):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2 or X.shape[1] != N_FEATURES or len(X) != len(y):
        raise ValueError("X e y deben contener vectores HOG y sus etiquetas.")

    if not np.isfinite(X).all():
        raise ValueError("El dataset contiene valores inválidos.")

    if len(np.unique(y)) < 2:
        raise ValueError("Se necesitan al menos dos identidades.")

    model = LinearSVC(
        C=C,
        dual="auto",
        penalty="l2",
        class_weight="balanced",
        random_state=42,
        max_iter=20000
    )

    return model.fit(X, y)


def save_model(model, path=DEFAULT_MODEL):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def load_model(path=DEFAULT_MODEL):

    return joblib.load(path)


def predict_identity(hog_features, model=None):
    """Identidad candidata y puntuación de decisión."""

    vector = np.asarray(hog_features, dtype=np.float32)

    if vector.shape != (N_FEATURES,) or not np.isfinite(vector).all():
        raise ValueError("Vector HOG inválido.")

    if model is None:
        model = load_model()

    scores = np.asarray(
        model.decision_function(vector.reshape(1, -1))
    ).ravel()

    # Caso binario
    if len(model.classes_) == 2:
        scores = np.array([-scores[0], scores[0]])

    winner = int(np.argmax(scores))

    return str(model.classes_[winner]), float(scores[winner])


def passes_threshold(score, threshold):
    """T1 debe calibrarse con consultas independientes."""

    return bool(score >= threshold)
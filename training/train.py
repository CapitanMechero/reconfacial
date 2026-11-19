import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from recognition.svm import train_svm, save_model, DEFAULT_MODEL


TEST_SIZE = 0.2
C = 0.1
SEED = 42


def main():

    # 1. Cargar dataset
    df = pd.read_csv("data/dataset.csv")

    X = df.drop(columns=["name"]).to_numpy(dtype=np.float32)
    y = df["name"].to_numpy()

    # 2. Separar entrenamiento y test
    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y
        )
    else:
        X_train, y_train = X, y

    # 3. Entrenar SVM
    model = train_svm(X_train, y_train, C=C)

    # 4. Evaluar
    print("C:", C)
    print("Accuracy entrenamiento:", model.score(X_train, y_train))

    if TEST_SIZE > 0:
        y_pred = model.predict(X_test)

        print("Accuracy test:", accuracy_score(y_test, y_pred))
        print("F1 macro test:", f1_score(y_test, y_pred, average="macro"))

    # 5. Guardar modelo
    save_model(model)

    print("Modelo guardado en:", DEFAULT_MODEL)


if __name__ == "__main__":
    main()
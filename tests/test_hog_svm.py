import numpy as np
import pytest

from recognition.hog import N_FEATURES, extract_hog
from recognition.svm import (
    decision_scores, load_model, passes_threshold, predict_identity, save_model, train_svm,
)
from tests.synthetic import make_face, make_variant


# ---- hog ---------------------------------------------------------------------
def test_hog_shape_dtype_and_determinism():
    f = make_face(1)
    v = extract_hog(f)
    assert v.shape == (N_FEATURES,) and v.dtype == np.float32 and np.isfinite(v).all()
    assert np.array_equal(v, extract_hog(f.copy()))
    vec, img = extract_hog(f, visualize=True)
    assert np.array_equal(vec, v) and img.shape == (128, 128)


@pytest.mark.parametrize("bad", [np.zeros((64, 64), np.uint8), np.zeros((128, 128), np.float32),
                                 np.zeros((128, 128, 3), np.uint8), [[0]], None])
def test_hog_rejects_invalid_faces(bad):
    with pytest.raises(ValueError):
        extract_hog(bad)


def test_hog_differs_between_faces():
    assert not np.allclose(extract_hog(make_face(1)), extract_hog(make_face(2)))


# ---- svm ---------------------------------------------------------------------
@pytest.fixture()
def trained():
    faces = {n: make_face(i) for i, n in enumerate("abc")}
    X = [extract_hog(f) for f in faces.values()] + [extract_hog(make_variant(f, 1)) for f in faces.values()]
    y = list(faces) * 2
    return train_svm(X, y), faces


def test_train_validations():
    good = np.zeros((2, N_FEATURES), np.float32)
    with pytest.raises(ValueError):
        train_svm(good, ["a", "a"])              # una sola identidad
    with pytest.raises(ValueError):
        train_svm(good[:, :10], ["a", "b"])      # dimensión errónea
    with pytest.raises(ValueError):
        train_svm(good, ["a"])                   # etiquetas desparejadas
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        train_svm(bad, ["a", "b"])


def test_predict_and_scores(trained):
    model, faces = trained
    name, score = predict_identity(extract_hog(make_variant(faces["b"], 5)), model)
    assert name == "b" and isinstance(score, float)
    classes, scores = decision_scores(extract_hog(faces["a"]), model)
    assert classes == ["a", "b", "c"] and scores.argmax() == 0


def test_binary_model_scores_are_two_sided():
    faces = [make_face(1), make_face(2)]
    model = train_svm([extract_hog(f) for f in faces], ["x", "y"])
    classes, scores = decision_scores(extract_hog(faces[0]), model)
    assert classes == ["x", "y"] and scores.shape == (2,) and scores[0] > scores[1]


@pytest.mark.parametrize("bad", [np.zeros(10), np.full(N_FEATURES, np.nan)])
def test_predict_rejects_invalid_vectors(trained, bad):
    with pytest.raises(ValueError):
        predict_identity(bad, trained[0])


def test_save_load_roundtrip_and_threshold(trained, tmp_path):
    model, faces = trained
    path = tmp_path / "sub" / "m.joblib"
    save_model(model, path)
    v = extract_hog(faces["c"])
    assert predict_identity(v, load_model(path)) == predict_identity(v, model)
    assert passes_threshold(0.5, 0.5) and not passes_threshold(0.49, 0.5)

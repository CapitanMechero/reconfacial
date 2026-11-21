"""Pruebas de ORB + matching. Usan imágenes de skimage.data como caras de prueba
(NO son caras reales salvo 'astronaut'); validan el comportamiento mecánico del
módulo, no su precisión real de reconocimiento facial."""
import cv2
import numpy as np
import pytest
from skimage import data

from recognition.orb import OrbTemplate, OrbExtractionError, extract_orb
from recognition.matching import (match_identity, calibrate_t2, far_frr)

SIZE = 128


def _patch(img, y, x, s):
    p = img[y:y + s, x:x + s]
    if p.ndim == 3:
        p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    return cv2.resize(p, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _gallery():
    """Parches distintos con textura, hacen de 'identidades'."""
    srcs = [data.astronaut(), data.coffee(), data.chelsea(), data.camera(),
            data.rocket(), data.brick(), data.text(), data.coins()]
    out = {}
    for k, im in enumerate(srcs):
        h, w = im.shape[:2]
        s = min(h, w) // 2
        out[f"p{k}_a"] = _patch(im, h // 8, w // 8, s)
        out[f"p{k}_b"] = _patch(im, h // 2 - s // 4, w // 2 - s // 4, s)
    out["face"] = _patch(data.astronaut(), 0, 150, 200)
    return out


def _perturb(face, angle=3, gain=1.15, noise=4, seed=0):
    rng = np.random.default_rng(seed)
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, 1.0)
    out = cv2.warpAffine(face, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
    out = out.astype(np.float32) * gain + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def db():
    faces = _gallery()
    templates = {}
    for k, f in faces.items():
        try:
            templates[k] = extract_orb(f)
        except OrbExtractionError:
            pass
    return faces, templates


def test_extract_shapes(db):
    _, t = db
    tpl = t["face"]
    assert tpl.descriptors.dtype == np.uint8 and tpl.descriptors.shape[1] == 32
    assert tpl.keypoints.shape == (len(tpl), 5)
    assert len(tpl) >= 15


def test_serialization_roundtrip(db):
    _, t = db
    tpl = t["face"]
    back = OrbTemplate.from_bytes(tpl.to_bytes())
    assert np.array_equal(back.descriptors, tpl.descriptors)
    assert np.allclose(back.keypoints, tpl.keypoints)


def test_corrupt_bytes_rejected(db):
    _, t = db
    with pytest.raises(ValueError):
        OrbTemplate.from_bytes(t["face"].to_bytes()[:-5])


def test_flat_image_rejected():
    with pytest.raises(OrbExtractionError):
        extract_orb(np.full((128, 128), 120, np.uint8))


def test_invalid_input():
    with pytest.raises(ValueError):
        extract_orb(np.array([]))


def test_color_input_converted(db):
    faces, _ = db
    bgr = cv2.cvtColor(faces["face"], cv2.COLOR_GRAY2BGR)
    assert len(extract_orb(bgr)) >= 15


def test_self_match_beats_others(db):
    faces, t = db
    q = extract_orb(_perturb(faces["face"]))
    res = match_identity(q, t, t2=0.1)
    assert res.best_candidate == "face"
    assert res.score > res.second_score


def test_known_accepted_unknown_rejected(db):
    faces, t = db
    t_known = {k: v for k, v in t.items() if k != "p3_b"}
    # conocida
    res = match_identity(extract_orb(_perturb(faces["face"], seed=1)), t_known, t2=0.1)
    assert res.accepted and res.identity == "face"
    # desconocida (no está en la BD)
    unk = extract_orb(_perturb(faces["p3_b"], seed=2))
    res = match_identity(unk, t_known, t2=0.1)
    assert not res.accepted and res.identity is None


def test_empty_database(db):
    _, t = db
    res = match_identity(t["face"], {})
    assert res.identity is None and not res.accepted


def test_descriptors_only_input(db):
    _, t = db
    res = match_identity(t["face"].descriptors, {"face": t["face"]}, t2=0.1)
    assert res.best_candidate == "face"


def test_calibration_and_metrics():
    genuine = [0.5, 0.4, 0.3, 0.6, 0.2]
    impostor = [0.05, 0.1, 0.02, 0.0, 0.12]
    cal = calibrate_t2(genuine, impostor, target_far=0.0)
    assert cal["far"] == 0.0 and cal["frr"] == 0.0
    assert 0.12 < cal["t2"] <= 0.2
    far, frr = far_frr(genuine, impostor, 0.25)
    assert far == 0.0 and frr == pytest.approx(0.2)

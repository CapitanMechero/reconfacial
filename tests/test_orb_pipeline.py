"""Registro ORB -> reconocimiento con los módulos reales de los compañeros
(preprocessing, detector.process_image) y un detector falso en lugar de MediaPipe
(no necesita el modelo)."""
import cv2
import numpy as np
import pytest
from skimage import data

from recognition.detector import FaceDetection
from recognition.errors import NoFaceDetectedError
from recognition.orb_store import delete_template, load_database
from scripts.register_orb import register
from scripts.demo_orb import main as demo


class FakeDetector:
    def __init__(self, eyes):
        self.eyes = eyes

    def detect(self, img):
        (lx, ly), (rx, ry) = self.eyes
        lm = np.zeros((478, 2), np.float32)
        lm[[33, 133]] = [[lx - 8, ly], [lx + 8, ly]]
        lm[[362, 263]] = [[rx - 8, ry], [rx + 8, ry]]
        return [FaceDetection((int(lx - 50), int(ly - 40), 150, 170), lm)]


class NoFaces:
    def detect(self, img):
        return []


def _save(tmp_path, name, rgb):
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return str(p)


def recognize(path, directory, detector):
    return demo(path, directory=directory, detector=detector, show=False)


@pytest.fixture
def setup(tmp_path):
    d = tmp_path / "orb"
    imgs = {"Ana": data.astronaut(), "Luis": data.coffee(), "María José": data.chelsea()}
    eyes = {"Ana": ((190, 78), (232, 76))}
    for n, im in imgs.items():
        h, w = im.shape[:2]
        e = eyes.get(n, ((w * .4, h * .4), (w * .6, h * .4)))
        register(n, _save(tmp_path, n.replace(" ", ""), im), d, FakeDetector(e))
    return d, tmp_path, eyes


def test_register_and_load(setup):
    d, _, _ = setup
    assert set(load_database(d)) == {"Ana", "Luis", "María José"}


def test_recognize_known(setup):
    d, tmp, eyes = setup
    M = cv2.getRotationMatrix2D((256, 256), 5, 1.04)
    q = cv2.convertScaleAbs(cv2.warpAffine(data.astronaut(), M, (512, 512)), alpha=1.15, beta=-8)
    qe = tuple(tuple((M @ np.array([x, y, 1.0])).tolist()) for x, y in eyes["Ana"])
    res = recognize(_save(tmp, "q", q), d, FakeDetector(qe))
    assert res.identity == "Ana" and res.accepted


def test_recognize_unknown(setup):
    d, tmp, _ = setup
    im = data.rocket()
    h, w = im.shape[:2]
    res = recognize(_save(tmp, "u", im), d, FakeDetector(((w * .4, h * .4), (w * .6, h * .4))))
    assert res.identity is None


def test_duplicate_name_rejected(setup):
    d, tmp, eyes = setup
    with pytest.raises(ValueError):
        register("Ana", _save(tmp, "x", data.astronaut()), d, FakeDetector(eyes["Ana"]))


def test_replace_allowed(setup):
    d, tmp, eyes = setup
    register("Ana", _save(tmp, "x", data.astronaut()), d, FakeDetector(eyes["Ana"]), replace=True)


def test_delete(setup):
    d, _, _ = setup
    assert delete_template("Luis", d) and "Luis" not in load_database(d)
    assert not delete_template("Luis", d)


def test_no_face_error_propagates(setup):
    d, tmp, _ = setup
    with pytest.raises(NoFaceDetectedError):
        recognize(_save(tmp, "n", data.astronaut()), d, NoFaces())


def test_empty_database(tmp_path):
    with pytest.raises(SystemExit):
        recognize(_save(tmp_path, "a", data.astronaut()), tmp_path / "vacio", NoFaces())

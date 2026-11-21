"""build_orb_dataset + demo_orb_dataset sobre un dataset sintético (carpeta por famoso, 5 fotos)."""
import csv
import cv2
import numpy as np
import pytest
from skimage import data

from recognition.detector import FaceDetection
from legacy.integrante3.orb_store import load_dataset_csv
from legacy.integrante3.build_orb_dataset import build
from legacy.integrante3.demo_orb_dataset import main as demo


class Fake:
    def detect(self, img):
        lm = np.zeros((478, 2), np.float32)
        lm[[33, 133]] = [[72, 80], [88, 80]]
        lm[[362, 263]] = [[112, 80], [128, 80]]
        return [FaceDetection((20, 20, 160, 160), lm)]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    srcs = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket(), data.brick(), data.coins()]
    rng = np.random.default_rng(1)
    k = 0
    for im in srcs:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
        h, w = g.shape
        for fy, fx in [(0.0, 0.0), (0.5, 0.0)]:
            s = min(h, w) // 2
            base = cv2.resize(g[int(fy * (h - s)):int(fy * (h - s)) + s, int(fx * (w - s)):int(fx * (w - s)) + s], (200, 200))
            d = root / f"Famoso Numero{k}"
            d.mkdir()
            for j in range(5):
                M = cv2.getRotationMatrix2D((100, 100), rng.uniform(-6, 6), rng.uniform(.95, 1.05))
                f = cv2.warpAffine(base, M, (200, 200), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                f = f * rng.uniform(.8, 1.2) + rng.normal(0, 4, f.shape)
                cv2.imwrite(str(d / f"foto{j + 1}.jpg"), np.clip(f, 0, 255).astype(np.uint8))
            k += 1
    return root


def test_build_csv(dataset, tmp_path):
    out = tmp_path / "orb.csv"
    rows, failed = build(dataset, out, detector=Fake())
    assert len(rows) == 60 and not failed
    with open(out, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["name", "image", "n_keypoints", "template"]
    db = load_dataset_csv(out)
    assert len(db) == 12 and all(len(v) == 5 for v in db.values())
    assert "Famoso Numero0" in db


def test_build_is_idempotent(dataset, tmp_path):
    out = tmp_path / "orb.csv"
    build(dataset, out, detector=Fake())
    build(dataset, out, detector=Fake())
    assert sum(len(v) for v in load_dataset_csv(out).values()) == 60


def test_demo_recognizes_most(dataset, tmp_path):
    out = tmp_path / "orb.csv"
    build(dataset, out, detector=Fake())
    res = demo(dataset, out, detector=Fake(), verbose=False)
    assert len(res) == 12
    assert sum(r["ok"] for r in res) >= 10


def test_include_self_is_trivially_perfect(dataset, tmp_path):
    out = tmp_path / "orb.csv"
    build(dataset, out, detector=Fake())
    res = demo(dataset, out, detector=Fake(), include_self=True, verbose=False)
    assert all(r["ok"] for r in res)

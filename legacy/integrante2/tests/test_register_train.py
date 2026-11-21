"""Tests del flujo CSV anterior (integrante 2): train.py y register.py. No se ejecutan en la CI.

Ejecutar: python -m pytest legacy/integrante2/tests
"""
import csv

import numpy as np
import pytest

from legacy.integrante2 import register as register_mod
from legacy.integrante2.train import train_from_csv
from recognition.hog import N_FEATURES, extract_hog
from tests.synthetic import make_face, make_variant


# ---- train.py (sobre un CSV temporal; nunca toca models/) -----------------------------
def _write_csv(path, n_ids=4, per_id=5):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + [f"hog_{i}" for i in range(N_FEATURES)])
        for i in range(n_ids):
            base = make_face(i)
            for s in range(per_id):
                w.writerow([f"p{i}"] + extract_hog(make_variant(base, s)).tolist())


def test_train_from_csv(tmp_path):
    csv_path, out = tmp_path / "d.csv", tmp_path / "m.joblib"
    _write_csv(csv_path)
    m = train_from_csv(csv_path, out, test_size=0.2)
    assert out.exists() and m["n_test"] > 0 and 0 <= m["test_accuracy"] <= 1
    out.unlink()
    train_from_csv(csv_path, out, test_size=0, save=False)
    assert not out.exists()


# ---- register.py ------------------------------------------------------------------------
class _P:
    def __init__(self, face):
        self.face = face


def test_register_appends_rows_and_header_once(tmp_path, monkeypatch):
    target = tmp_path / "ds.csv"
    monkeypatch.setattr(register_mod, "CSV_PATH", target)
    monkeypatch.setattr(register_mod, "process_image", lambda p: _P(make_face(3)))
    f1 = register_mod.register("  Ana ", "x.jpg")
    register_mod.register("Luis", "y.jpg")
    rows = list(csv.reader(open(target)))
    assert len(rows) == 3 and rows[0][0] == "name" and rows[1][0] == "Ana"
    assert np.allclose(np.array(rows[1][1:], dtype=np.float32), f1, atol=1e-6)


def test_register_rejects_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(register_mod, "CSV_PATH", tmp_path / "ds.csv")
    with pytest.raises(ValueError):
        register_mod.register("   ", "x.jpg")
    assert not (tmp_path / "ds.csv").exists()

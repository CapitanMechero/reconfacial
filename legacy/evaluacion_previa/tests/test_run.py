"""Test del estudio previo de evaluación. No se ejecuta en la CI.

Ejecutar: python -m pytest legacy/evaluacion_previa/tests
"""
from evaluation.splits import Sample
from legacy.evaluacion_previa.run import run
from recognition.hog import extract_hog
from recognition.orb import extract_orb
from tests.synthetic import make_face, make_variant


def _samples(n_ids=10, per_id=4, with_orb=False):
    out = []
    for i in range(n_ids):
        base = make_face(i)
        for k in range(per_id):
            img = make_variant(base, k, (0.5 * k, -0.5 * k))
            out.append(Sample(f"{i}-{k}", f"id{i}", extract_hog(img),
                              extract_orb(img) if with_orb else None))
    return out


def test_full_pipeline_on_synthetic_data_with_orb(tmp_path):
    rep = run(_samples(n_ids=10, per_id=4, with_orb=True), seed=0, target_far=0.0,
              min_margin=0.01, modes=["hog", "orb", "cascade"], out_dir=tmp_path)
    for mode in ("hog", "orb", "cascade"):
        assert rep["modes"][mode]["calibration"]["far"]["k"] == 0
    # en datos sintéticos ORB separa perfectamente: comprueba que la tubería funciona
    assert rep["modes"]["orb"]["test"]["ir"]["value"] > 0.9
    assert (tmp_path / "calibration_hog.png").exists()

import math
from dataclasses import replace

import numpy as np
import pytest

from evaluation import cascade
from evaluation.metrics import Rate, confusion, open_set_metrics, per_identity
from evaluation.splits import LeakageError, Sample, make_split, validate_split
from recognition.config import Thresholds
from recognition.hog import extract_hog
from recognition.orb import extract_orb
from recognition.recognition import RecognitionEngine
from recognition.types import HogPrediction, OrbOutcome
from tests.synthetic import make_face, make_variant


def test_open_set_metric_definitions():
    m = open_set_metrics(["a", "a", "b", "b"], ["a", None, "a", "b"], [None, "a", None, None])
    assert (m.ir.k, m.frr.k, m.fir.k, m.ir.n) == (2, 1, 1, 4)
    assert m.ir.value + m.frr.value + m.fir.value == 1
    assert (m.far.k, m.far.n) == (1, 4)
    with pytest.raises(ValueError):
        open_set_metrics(["a"], [], [])


def test_wilson_interval():
    lo, hi = Rate(0, 50).wilson()
    assert lo == 0 and hi == pytest.approx(0.0713, abs=1e-3)
    assert all(math.isnan(x) for x in Rate(0, 0).wilson())
    lo, hi = Rate(5, 10).wilson()
    assert lo < 0.5 < hi


def test_per_identity_and_confusion():
    assert per_identity(["a", "a"], ["a", None])["a"] == {
        "n": 2, "correct": 1, "rejected": 1, "misidentified": 0}
    cm = confusion(["a"], ["b"], ["a", None])
    assert cm[("a", "b")] == 1 and cm[("<impostor>", "a")] == 1


def _samples(n_ids=10, per_id=4, with_orb=False):
    out = []
    for i in range(n_ids):
        base = make_face(i)
        for k in range(per_id):
            img = make_variant(base, k, (0.5 * k, -0.5 * k))
            out.append(Sample(f"{i}-{k}", f"id{i}", extract_hog(img),
                              extract_orb(img) if with_orb else None))
    return out


def test_split_has_no_leakage_and_is_reproducible():
    s = _samples()
    a, b = make_split(s, seed=1), make_split(s, seed=1)
    assert [x.sample_id for x in a.test_genuine] == [x.sample_id for x in b.test_genuine]
    known = set(a.known)
    assert not ({x.identity for x in a.calib_impostor} & known)
    assert not ({x.identity for x in a.test_impostor} & {x.identity for x in a.calib_impostor})
    one = make_split(s, seed=1, max_train_per_identity=1)
    assert len(one.train) == len(one.known)
    assert [x.sample_id for x in one.test_genuine] == [x.sample_id for x in a.test_genuine]


def test_validate_split_detects_leaks():
    split = make_split(_samples(), seed=3)
    leaked = replace(split, test_genuine=split.test_genuine + [split.train[0]])
    with pytest.raises(LeakageError):
        validate_split(leaked)
    clone = Sample("nuevo-id", split.train[0].identity, split.train[0].hog.copy())
    with pytest.raises(LeakageError):
        validate_split(replace(split, calib_genuine=split.calib_genuine + [clone]))
    wrong = replace(split, test_impostor=split.test_impostor + [split.calib_impostor[0]])
    with pytest.raises(LeakageError):
        validate_split(wrong)


def test_split_requires_enough_data():
    with pytest.raises(ValueError):
        make_split(_samples(n_ids=3))


class _Svm:
    def __init__(self, p):
        self.p = p

    def predict(self, face):
        return self.p


class _Orb:
    def __init__(self, o):
        self.o = o

    def match(self, face, refs):
        return self.o


class _Tpl:
    def __init__(self, names):
        self.names = names

    def load_orb_templates(self):
        return {n: [object()] for n in self.names}


def test_offline_decision_matches_engine():
    """La regla offline debe coincidir con RecognitionEngine en todos los casos."""
    rng = np.random.default_rng(0)
    known = ["a", "b", "c"]
    for _ in range(300):
        svm_label = rng.choice(known + ["borrado"])
        best = float(rng.choice([0.0, rng.uniform(0, 0.3)]))     # incluye entradas sin evidencia
        q = cascade.QueryScores("a", svm_label, float(rng.normal()), rng.choice(known),
                                best, float(rng.uniform(0, best)) if best else 0.0)
        t = Thresholds(t1=float(rng.normal()), t2=float(rng.uniform(0, 0.3)),
                       orb_min_margin=float(rng.uniform(0, 0.05)))
        accepted = q.orb_best >= t.t2 and q.orb_margin >= t.orb_min_margin
        eng = RecognitionEngine(lambda i: None, _Svm(HogPrediction(q.svm_label, q.svm_score)),
                                _Orb(OrbOutcome(q.orb_label, q.orb_best, q.orb_best - q.orb_margin,
                                                q.orb_margin, accepted, 3)),
                                _Tpl(known), t)
        expected = eng.recognize_face(np.zeros((128, 128), np.uint8)).identity
        assert cascade.decide(q, t.t1, t.t2, t.orb_min_margin, frozenset(known),
                              t.orb_min_evidence) == expected


def test_threshold_selection_respects_far_target_on_calibration():
    g = [cascade.QueryScores("a", "a", 1.0, "a", 0.3, 0.3),
         cascade.QueryScores("b", "b", 0.2, "b", 0.1, 0.1)]
    i = [cascade.QueryScores("x", "a", 0.5, "b", 0.05, 0.05)]
    sel = cascade.select_thresholds(g, i, ["a", "b"], "hog", 0.0, 0.0)
    assert sel.metrics.far.k == 0 and sel.t1 > 0.5 and sel.metrics.ir.k == 1
    sel = cascade.select_thresholds(g, i, ["a", "b"], "cascade", 0.0, 0.0)
    assert sel.metrics.far.k == 0 and sel.metrics.ir.k == 2

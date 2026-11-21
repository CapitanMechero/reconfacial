"""Cascada T1 -> T2 -> desconocido con dependencias inyectadas y con algoritmos reales."""
from dataclasses import dataclass

import numpy as np
import pytest

from recognition.config import Thresholds
from recognition.errors import (
    KeyConfigurationError, MultipleFacesError, NoFaceDetectedError, ModelNotFoundError,
    InvalidImageError, StorageError, SvmUnavailableError, TemplateCorruptError,
)
from recognition.hog import extract_hog
from recognition.orb import extract_orb
from recognition.recognition import OrbDatabaseMatcher, RecognitionEngine, SvmClassifier
from recognition.svm import save_model, train_svm
from recognition.types import (
    ErrorCode, HogPrediction, OrbOutcome, RecognitionStatus, ScoreKind, Stage,
)
from tests.synthetic import make_face, make_variant

FACE = make_face(0)


@dataclass
class FakeProcessed:
    face: np.ndarray


class FakeSvm:
    def __init__(self, prediction=None, exc=None):
        self.prediction, self.exc = prediction, exc

    def predict(self, face):
        if self.exc:
            raise self.exc
        return self.prediction


class FakeOrb:
    def __init__(self, outcome=None, exc=None):
        self.outcome, self.exc = outcome, exc

    def match(self, face, references):
        if self.exc:
            raise self.exc
        return self.outcome


class FakeTemplates:
    def __init__(self, names=("ana", "luis"), exc=None):
        self.names, self.exc = names, exc

    def load_orb_templates(self):
        if self.exc:
            raise self.exc
        return {n: [] for n in self.names}


T = Thresholds(t1=0.5, t2=0.1, orb_min_margin=0.02, calibrated=True)


def engine(svm=None, orb=None, templates=None, processor=None):
    return RecognitionEngine(
        processor or (lambda img: FakeProcessed(FACE)),
        svm or FakeSvm(HogPrediction("ana", 0.0, -1.0)),
        orb or FakeOrb(OrbOutcome("luis", 0.01, 0.0, 0.01, False, 2)),   # evidencia mínima, < T2
        templates or FakeTemplates(), T)


def test_t1_accepts_when_orb_evidence_exists():
    r = engine(svm=FakeSvm(HogPrediction("ana", 0.9, 0.1))).recognize(b"x")
    assert r.status is RecognitionStatus.IDENTIFIED and r.identity == "ana"
    assert r.stage is Stage.HOG_SVM and r.score_kind is ScoreKind.SVM_DECISION
    assert r.thresholds_calibrated


def test_t1_boundary_is_inclusive():
    r = engine(svm=FakeSvm(HogPrediction("ana", 0.5, 0.0))).recognize(b"x")
    assert r.is_identified and r.stage is Stage.HOG_SVM


def test_falls_back_to_orb_and_accepts():
    orb = FakeOrb(OrbOutcome("luis", 0.3, 0.05, 0.25, True, 2))
    r = engine(svm=FakeSvm(HogPrediction("ana", 0.4, 0.0)), orb=orb).recognize(b"x")
    assert r.identity == "luis" and r.stage is Stage.ORB
    assert r.score_kind is ScoreKind.ORB_INLIER_RATIO
    assert any("T1 sugería" in n for n in r.notes)


def test_orb_below_t2_is_unknown():
    orb = FakeOrb(OrbOutcome(None, 0.02, 0.0, 0.02, False, 2))
    r = engine(svm=FakeSvm(HogPrediction("ana", 0.1, 0.0)), orb=orb).recognize(b"x")
    assert r.status is RecognitionStatus.UNKNOWN and r.identity is None
    assert r.hog is not None and r.orb is not None


def test_orb_ambiguous_is_uncertain_not_identified():
    orb = FakeOrb(OrbOutcome(None, 0.3, 0.29, 0.01, False, 2))
    r = engine(svm=FakeSvm(HogPrediction("ana", 0.1, 0.0)), orb=orb).recognize(b"x")
    assert r.status is RecognitionStatus.UNCERTAIN


def test_orb_insufficient_keypoints_is_uncertain():
    orb = FakeOrb(OrbOutcome(None, 0.0, 0.0, 0.0, False, 2, insufficient_keypoints=True))
    assert engine(orb=orb).recognize(b"x").status is RecognitionStatus.UNCERTAIN


@pytest.mark.parametrize("outcome", [
    OrbOutcome(None, 0.0, 0.0, 0.0, False, 2),                               # sin coincidencias
    OrbOutcome(None, 0.0, 0.0, 0.0, False, 2, insufficient_keypoints=True),  # sin puntos
])
def test_no_orb_evidence_blocks_even_a_confident_svm(outcome):
    """Mitigación A2/B4: una no-cara con margen SVM alto no se identifica."""
    r = engine(svm=FakeSvm(HogPrediction("ana", 99.0, 0.0)), orb=FakeOrb(outcome)).recognize(b"x")
    assert r.status is RecognitionStatus.UNCERTAIN and r.identity is None
    assert any("no es una cara" in n for n in r.notes)


def test_svm_unavailable_degrades_visibly():
    orb = FakeOrb(OrbOutcome("ana", 0.4, 0.0, 0.4, True, 2))
    r = engine(svm=FakeSvm(exc=SvmUnavailableError("sin modelo")), orb=orb).recognize(b"x")
    assert r.identity == "ana" and r.hog is None
    assert any("T1 omitido" in n for n in r.notes)


def test_stale_svm_identity_is_never_accepted():
    # El SVM aún conoce a «borrado», que ya no está en la base de datos.
    r = engine(svm=FakeSvm(HogPrediction("borrado", 9.0, 0.0))).recognize(b"x")
    assert r.status is RecognitionStatus.UNKNOWN
    assert any("desactualizado" in n for n in r.notes)


def test_svm_value_error_is_error_not_unknown():
    r = engine(svm=FakeSvm(exc=ValueError("vector inválido"))).recognize(b"x")
    assert r.status is RecognitionStatus.ERROR and r.error.code is ErrorCode.SVM_FAILURE


def test_empty_database_is_unknown_with_note():
    r = engine(templates=FakeTemplates(names=())).recognize(b"x")
    assert r.status is RecognitionStatus.UNKNOWN and r.notes


@pytest.mark.parametrize("exc,code", [
    (NoFaceDetectedError("x"), ErrorCode.NO_FACE),
    (MultipleFacesError(3), ErrorCode.MULTIPLE_FACES),
    (InvalidImageError("x"), ErrorCode.INVALID_IMAGE),
    (ModelNotFoundError("x"), ErrorCode.DETECTOR_MODEL_MISSING),
])
def test_processing_errors_are_typed(exc, code):
    def processor(_):
        raise exc
    r = engine(processor=processor).recognize(b"x")
    assert r.status is RecognitionStatus.ERROR and r.error.code is code


@pytest.mark.parametrize("exc,code", [
    (TemplateCorruptError("x"), ErrorCode.TEMPLATE_CORRUPT),
    (KeyConfigurationError("x"), ErrorCode.KEY_ERROR),
    (StorageError("x"), ErrorCode.STORAGE_FAILURE),
])
def test_storage_errors_are_typed(exc, code):
    r = engine(templates=FakeTemplates(exc=exc)).recognize(b"x")
    assert r.status is RecognitionStatus.ERROR and r.error.code is code


def test_unexpected_exceptions_propagate():
    def processor(_):
        raise ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        engine(processor=processor).recognize(b"x")


# --------------------- integración con algoritmos reales ---------------------
class RealTemplates:
    def __init__(self, faces):
        self.faces = faces

    def load_orb_templates(self):
        return {n: [extract_orb(f)] for n, f in self.faces.items()}


@pytest.fixture()
def real_engine(tmp_path):
    faces = {f"p{i}": make_face(i) for i in range(4)}
    X = np.array([extract_hog(f) for f in faces.values()] +
                 [extract_hog(make_variant(f, s)) for f in faces.values() for s in (1, 2)])
    y = list(faces) + [n for n in faces for _ in (1, 2)]
    model_path = tmp_path / "svm.joblib"
    save_model(train_svm(X, y), model_path)
    t = Thresholds(t1=0.0, t2=0.05, orb_min_margin=0.01, calibrated=False)
    eng = RecognitionEngine(lambda img: FakeProcessed(img), SvmClassifier(model_path),
                            OrbDatabaseMatcher(t), RealTemplates(faces), t)
    return eng, faces


def test_real_cascade_identifies_registered_person(real_engine):
    eng, faces = real_engine
    r = eng.recognize_face(make_variant(faces["p2"], 77))
    assert r.is_identified and r.identity == "p2"


def test_real_cascade_rejects_stranger_with_strict_t1(real_engine):
    eng, _ = real_engine
    strict = RecognitionEngine(eng._processor, eng._svm, eng._orb, eng._templates,
                               Thresholds(t1=50.0, t2=0.05, orb_min_margin=0.01))
    r = strict.recognize_face(make_face(999))
    assert not r.is_identified
    assert not r.thresholds_calibrated


def test_real_orb_rescues_when_t1_is_too_strict(real_engine):
    eng, faces = real_engine
    strict = RecognitionEngine(eng._processor, eng._svm, eng._orb, eng._templates,
                               Thresholds(t1=50.0, t2=0.05, orb_min_margin=0.01))
    r = strict.recognize_face(make_variant(faces["p1"], 5))
    assert r.identity == "p1" and r.stage is Stage.ORB


def test_svm_classifier_reloads_when_model_changes(tmp_path):
    path = tmp_path / "m.joblib"
    clf = SvmClassifier(path)
    with pytest.raises(SvmUnavailableError):
        clf.predict(FACE)
    faces = [make_face(i) for i in range(3)]
    save_model(train_svm([extract_hog(f) for f in faces], ["a", "b", "c"]), path)
    assert clf.predict(faces[1]).identity == "b"
    path.write_bytes(b"corrupto")
    with pytest.raises(SvmUnavailableError):
        clf.predict(faces[1])

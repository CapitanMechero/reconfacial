"""Flujo extremo a extremo con CSV real (HMAC), SVM real y ORB real.

El detector MediaPipe se sustituye por un procesador que recibe ya la cara
normalizada (las fotos reales se prueban en test_real_photos.py). Todo ocurre en
``tmp_path``: nunca toca var/, data/ ni models/ del proyecto.
"""
import json
from dataclasses import dataclass

import numpy as np
import pytest

from recognition.bootstrap import build_service
from recognition.config import Thresholds
from recognition.errors import IdentityNotFoundError, MultipleFacesError, NoFaceDetectedError
from recognition.model_lifecycle import ModelStatus, read_meta
from recognition.service import DuplicateFaceError, PhotoRejectedError
from recognition.template import extract_template, to_base64
from recognition.types import RecognitionStatus, Stage, VerificationStatus
from tests.synthetic import make_face, make_variant, tmp_settings

T = Thresholds(t1=0.0, t2=0.05, orb_min_margin=0.01, t1_verify=0.0, t2_verify=0.05)
STRICT = Thresholds(t1=1e9, t2=0.05, orb_min_margin=0.01, t1_verify=1e9, t2_verify=0.05)


@dataclass
class P:
    face: np.ndarray


def processor(image):
    if isinstance(image, str):
        raise {"none": NoFaceDetectedError("sin cara"), "many": MultipleFacesError(2)}[image]
    return P(image)


def make_service(tmp_path, thresholds=T):
    return build_service(tmp_settings(tmp_path, thresholds), processor=processor)


@pytest.fixture()
def service(tmp_path):
    return make_service(tmp_path)


def photos(seed):
    base = make_face(seed)
    return base, [base, make_variant(base, 1), make_variant(base, 2, (-1.0, 2.0))]


def test_full_lifecycle(service, tmp_path):
    faces = {n: photos(i) for i, n in enumerate(["Ana", "Luis", "Eva"])}
    for name, (_, imgs) in faces.items():
        out = service.enroll(name, imgs)
        assert out.identity.name == name
    assert out.training.status is ModelStatus.UP_TO_DATE
    assert [i.name for i in service.list_identities()] == ["Ana", "Eva", "Luis"]

    query = make_variant(faces["Luis"][0], 50, (0.5, 0.5))         # foto no inscrita
    r = service.identify(query)
    assert r.status is RecognitionStatus.IDENTIFIED and r.identity == "Luis"

    strict = make_service(tmp_path, STRICT)                         # misma BD, T1 inalcanzable
    unknown = strict.identify(make_face(12345))   # sin coincidencias ORB: lo frena el filtro
    assert not unknown.is_identified and unknown.candidate is not None

    out = service.delete("luis")
    assert out.templates_removed and out.absent_from_model
    assert strict.identify(query).identity != "Luis"
    assert not service.identify(query).is_identified
    assert not service.verify("Luis", query).accepted
    with pytest.raises(IdentityNotFoundError):
        service.delete("Luis")


def test_verification(service):
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    ana_query = make_variant(photos(0)[0], 60)
    ok = service.verify("ana", ana_query)
    assert ok.status is VerificationStatus.ACCEPTED and bool(ok) and ok.claimed == "Ana"
    assert ok.svm_score is not None
    assert not service.verify("Nadie", ana_query).accepted
    assert service.verify("Ana", "none").status is VerificationStatus.ERROR


def test_verification_by_orb_rejects_impostor(tmp_path):
    s = make_service(tmp_path, STRICT)
    for i, n in enumerate(["Ana", "Luis"]):
        s.enroll(n, photos(i)[1])
    q = make_variant(photos(0)[0], 61)
    assert s.verify("Ana", q).accepted and s.verify("Ana", q).stage is Stage.ORB
    assert not s.verify("Luis", q).accepted


def test_three_input_levels_are_consistent(service):
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    q = make_variant(photos(1)[0], 70)
    t = extract_template(q)
    assert service.identify(q).identity == "Luis"
    assert service.identify_template(to_base64(t)).identity == "Luis"
    assert service.identify_template(t).identity == "Luis"
    assert service.verify_template("Luis", to_base64(t)).accepted
    # _sd: la «foto» ya es la cara; se reescala y se convierte a gris
    import cv2
    big = cv2.resize(cv2.cvtColor(q, cv2.COLOR_GRAY2BGR), (256, 256))
    assert service.identify_without_detector(big).status in set(RecognitionStatus)
    ok, png = cv2.imencode(".png", q)
    assert service.identify_without_detector(png.tobytes()).identity == "Luis"
    assert service.identify_without_detector(b"basura").status is RecognitionStatus.ERROR


def test_enroll_templates_and_without_detector(service):
    out = service.enroll_templates("Ana", [to_base64(extract_template(make_face(1)))])
    assert out.identity.n_templates == 1
    import cv2
    ok, png = cv2.imencode(".png", make_face(2))
    assert service.enroll("Luis", [png.tobytes()], detector=False).identity.n_templates == 1
    with pytest.raises(PhotoRejectedError):
        service.enroll_templates("Eva", ["no-es-un-template"])


def test_enrolling_existing_name_is_rejected(service):
    from recognition.errors import DuplicateIdentityError
    service.enroll("Ana", [make_face(1)])
    with pytest.raises(DuplicateIdentityError):
        service.enroll("  ANA ", [make_variant(make_face(1), 3)])
    assert service.store.get_identity("Ana").n_templates == 1


def test_duplicate_face_is_rejected(service):
    _, imgs = photos(1)
    service.enroll("Ana", imgs)
    with pytest.raises(DuplicateFaceError) as e:
        service.enroll("Otra", [make_variant(imgs[0], 9)])
    assert e.value.existing == "Ana" and [i.name for i in service.list_identities()] == ["Ana"]
    service.enroll("Otra", [make_variant(imgs[0], 9)], allow_duplicate_face=True)


def test_enrollment_is_all_or_nothing(service):
    with pytest.raises(PhotoRejectedError) as e:
        service.enroll("Ana", [make_face(1), "many"])
    assert e.value.index == 1
    with pytest.raises(PhotoRejectedError):
        service.enroll("Ana", [np.zeros((128, 128), np.uint8)])   # sin textura para ORB
    with pytest.raises(ValueError):
        service.enroll("Ana", [])
    assert service.list_identities() == []


def test_recognition_errors_are_results(service):
    service.enroll("Ana", photos(1)[1])
    assert service.identify("none").error.code.value == "no_face"
    assert service.identify("many").error.code.value == "multiple_faces"


def test_single_identity_disables_svm_but_orb_works(service):
    out = service.enroll("Ana", photos(1)[1])
    assert out.training.status is ModelStatus.NOT_ENOUGH_IDENTITIES
    r = service.identify(make_variant(photos(1)[0], 77))
    assert r.identity == "Ana" and r.stage is Stage.ORB


def test_failed_retrain_keeps_model_and_stale_model_never_decides(service, monkeypatch):
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    before = service.settings.paths.svm_model.read_bytes()
    monkeypatch.setattr("recognition.model_lifecycle.train_svm",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("fallo simulado")))
    out = service.enroll("Eva", photos(5)[1])
    assert out.training.error and out.training.status is ModelStatus.STALE
    assert service.settings.paths.svm_model.read_bytes() == before
    r = service.identify(make_variant(photos(5)[0], 40))
    assert r.identity == "Eva" and any("T1 omitido" in n for n in r.notes)
    monkeypatch.undo()
    assert service.ensure_model_current() is ModelStatus.UP_TO_DATE


def test_failed_retrain_after_delete_discards_model(service, monkeypatch):
    for i, n in enumerate(["Ana", "Luis", "Eva"]):
        service.enroll(n, photos(i)[1])
    monkeypatch.setattr("recognition.model_lifecycle.train_svm",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    out = service.delete("Luis")
    assert out.training.error and out.absent_from_model
    assert not service.settings.paths.svm_model.exists()


def test_replaced_model_and_meta_without_key_are_rejected(service):
    """Atacante que escribe el volumen: sustituye modelo y recalcula el hash del meta."""
    import hashlib
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    paths = service.settings.paths
    paths.svm_model.write_bytes(paths.svm_model.read_bytes() + b"x")
    meta = json.loads(paths.svm_meta.read_text())
    meta["sha256"] = hashlib.sha256(paths.svm_model.read_bytes()).hexdigest()
    paths.svm_meta.write_text(json.dumps(meta))                 # sin la clave no puede re-firmar
    assert service.model_status() is ModelStatus.INVALID
    r = service.identify(make_variant(photos(0)[0], 3))
    assert r.hog is None and any("T1 omitido" in n for n in r.notes)
    assert service.ensure_model_current() is ModelStatus.UP_TO_DATE


def test_interrupted_publish_between_model_and_meta_recovers(service):
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    paths = service.settings.paths
    meta = read_meta(paths)
    meta["sha256"] = "0" * 64
    paths.svm_meta.write_text(json.dumps(meta))
    assert service.model_status() is ModelStatus.INVALID
    assert service.ensure_model_current() is ModelStatus.UP_TO_DATE


def test_tampered_database_gives_errors_not_unknown(service):
    service.enroll("Ana", photos(1)[1])
    db = service.settings.paths.db_path
    db.write_text(db.read_text().replace(",Ana,", ",Intruso,", 1))
    r = service.identify(make_face(1))
    assert r.status is RecognitionStatus.ERROR and r.error.code.value == "template_corrupt"
    assert service.verify("Ana", make_face(1)).status is VerificationStatus.ERROR


def test_orb_evidence_gate_on_templates_and_non_faces(service):
    """Mitigación A2/B4 con algoritmos reales: template aleatorio e imagen lisa."""
    from recognition.orb import OrbTemplate
    from recognition.types import RegisteredFace
    for i, n in enumerate(["Ana", "Luis"]):
        service.enroll(n, photos(i)[1])
    rng = np.random.default_rng(0)
    kps = np.column_stack([rng.uniform(10, 118, (300, 2)), np.full(300, 15.0),
                           rng.uniform(0, 360, 300), np.zeros(300)]).astype(np.float32)
    fake = RegisteredFace(rng.uniform(0, 0.4, 8100).astype(np.float32),
                          OrbTemplate(kps, rng.integers(0, 256, (300, 32), dtype=np.uint8)))
    r = service.identify_template(fake)
    assert not r.is_identified and any("no es una cara" in n for n in r.notes)
    assert not service.verify_template("Ana", fake).accepted
    import cv2
    flat = cv2.imencode(".png", np.full((128, 128), 128, np.uint8))[1].tobytes()
    assert not service.identify_without_detector(flat).is_identified


def test_cannot_add_any_face_to_existing_identity(service):
    """V-01: no se pueden añadir fotos (de nadie) a una identidad ya inscrita, por ninguna vía."""
    from recognition.errors import DuplicateIdentityError
    service.enroll("Ana", photos(0)[1])
    service.enroll("Luis", photos(1)[1])
    before = service.store.get_identity("Ana").n_templates
    intruder = make_variant(photos(1)[0], 31)                      # cara de Luis
    for attempt in (
        lambda: service.enroll("ana", [intruder]),
        lambda: service.enroll("ana", [intruder], allow_duplicate_face=True),
        lambda: service.enroll("ana", [intruder], detector=False),
        lambda: service.enroll_templates("ana", [to_base64(extract_template(intruder))]),
        lambda: service.enroll("Ana", [make_variant(photos(0)[0], 6)]),   # ni su propia cara
    ):
        with pytest.raises(DuplicateIdentityError):
            attempt()
    assert service.store.get_identity("Ana").n_templates == before
    assert not service.verify("Ana", make_variant(photos(1)[0], 32)).accepted

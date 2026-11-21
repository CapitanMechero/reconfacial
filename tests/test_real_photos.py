"""Pipeline real (MediaPipe + BD cifrada + SVM + ORB) con fotografías de data/fotos.

Las fotos no están en git (datos biométricos): sin ellas, o sin el modelo de
MediaPipe, estas pruebas se omiten. Se comprueba el CONTRATO del sistema, no su
precisión (que se mide en evaluation/ con particiones independientes).
"""
from pathlib import Path

import pytest

from recognition.config import PROJECT_ROOT
from recognition.errors import IdentityNotFoundError
from recognition.model_lifecycle import ModelStatus
from recognition.types import RecognitionStatus

PHOTOS = PROJECT_ROOT / "data" / "fotos"
pytestmark = pytest.mark.skipif(
    not PHOTOS.is_dir() or not (PROJECT_ROOT / "models" / "face_landmarker.task").is_file(),
    reason="requiere data/fotos y el modelo de MediaPipe")


def _people(n: int, per_person: int) -> dict[str, list[Path]]:
    out = {}
    for d in sorted(p for p in PHOTOS.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(imgs) >= per_person:
            out[d.name] = imgs
        if len(out) == n:
            break
    return out


@pytest.fixture()
def service(tmp_path):
    from recognition.bootstrap import build_service
    from tests.synthetic import tmp_settings
    return build_service(tmp_settings(tmp_path))


def _register_valid(service, name, images, k=3):
    """Registra con las primeras k fotos que el detector acepte."""
    from recognition.errors import FaceProcessingError
    from recognition.orb import OrbExtractionError
    chosen = []
    for img in images:
        try:
            service.template_from_photo(img.read_bytes())
        except (FaceProcessingError, OrbExtractionError):
            continue
        chosen.append(img)
        if len(chosen) == k:
            break
    assert len(chosen) == k, f"{name}: no hay {k} fotos válidas"
    service.enroll(name, [p.read_bytes() for p in chosen], allow_duplicate_face=True)
    return chosen


def test_real_register_recognize_delete(service):
    people = _people(4, 5)
    registered = list(people)[:3]
    used = {n: _register_valid(service, n, people[n]) for n in registered}
    assert service.model_status() is ModelStatus.UP_TO_DATE

    for name in registered:
        query = next(p for p in people[name] if p not in used[name])   # foto independiente
        a = service.analyze(query.read_bytes())
        v = service.verify(name, query.read_bytes())
        assert v.status.value in {"accepted", "rejected", "error"}
        assert a.result.status in set(RecognitionStatus)
        if a.result.status is not RecognitionStatus.ERROR:
            assert a.processed is not None and a.processed.face.shape == (128, 128)

    stranger = people[list(people)[3]][0].read_bytes()
    r = service.identify(stranger)
    assert r.status in set(RecognitionStatus) and (r.identity is None) == (not r.is_identified)
    victim = registered[0]
    out = service.delete(victim)
    assert out.templates_removed and out.absent_from_model
    for p in used[victim]:
        assert service.identify(p.read_bytes()).identity != victim
    with pytest.raises(IdentityNotFoundError):
        service.delete(victim)


def test_real_photo_with_two_faces_is_rejected_with_typed_error(service):
    from recognition.detector import process_image
    from recognition.errors import FaceProcessingError, MultipleFacesError
    for img in sorted(PHOTOS.rglob("*")):
        if img.is_file():
            try:
                process_image(img)
            except MultipleFacesError:
                r = service.identify(img.read_bytes())
                assert r.status is RecognitionStatus.ERROR and r.error.code.value == "multiple_faces"
                return
            except FaceProcessingError:
                continue
    pytest.skip("no hay fotos con varias caras")


def test_nine_methods_with_real_photos(tmp_path):
    """Los 9 métodos de la práctica sobre el sistema desplegado (semilla + umbrales reales).

    Para tener una persona «nueva», se elimina primero de una copia temporal de la BD
    (operación interna; la API no ofrece eliminar) y se inscribe con su foto de
    calibración. Se comprueba el contrato, no la precisión (ver evaluation/deployed.py).
    """
    import csv

    import cv2

    import sistema
    from recognition.bootstrap import build_service
    from recognition.config import load_settings
    from recognition.detector import process_image
    from recognition.errors import DuplicateIdentityError
    from tests.synthetic import tmp_settings

    real = load_settings()
    settings = tmp_settings(tmp_path, real.thresholds, seed=real.paths.seed_db)
    s = sistema.SistemaBiometrico(build_service(settings))
    assert len(s.identidades()) == 51

    holdout: dict[str, dict[str, Path]] = {}
    with open(real.paths.holdout, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            holdout.setdefault(r["identity"], {})[r["split"]] = PHOTOS / r["photo"]
    who = "henry_cavill"
    s.service.delete(who)
    cal, test = holdout[who]["calibracion"], holdout[who]["test"]
    crop = tmp_path / "crop.png"                                   # cara ya recortada para _sd
    cv2.imwrite(str(crop), process_image(test).face)

    # inscribir / inscribir_sd / inscribir_t (identidades nuevas)
    assert s.inscribir("nuevo", cal).n_templates == 1
    assert s.inscribir_sd("nuevo_sd", crop).n_templates == 1
    t = s.extraer_template(test)
    s.service.delete("adele")                                      # otra persona «nueva» para _t
    assert s.inscribir_t("nuevo_t", s.extraer_template(holdout["adele"]["calibracion"])) \
        .n_templates == 1
    with pytest.raises(DuplicateIdentityError):
        s.inscribir("adam_pearson", cal)

    # verificar / verificar_sd / verificar_t
    for v in (s.verificar("nuevo", test), s.verificar_sd("nuevo", crop), s.verificar_t("nuevo", t)):
        assert v.status.value in {"accepted", "rejected"} and v.error is None
    assert not s.verificar("adam_pearson", test).accepted           # impostor

    # identificar / identificar_sd / identificar_t
    for r in (s.identificar(test), s.identificar_sd(crop), s.identificar_t(t)):
        assert r.error is None and r.candidate is not None
        assert r.identity in (None, "nuevo", "nuevo_sd", "nuevo_t")

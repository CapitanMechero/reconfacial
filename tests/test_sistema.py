"""API de la práctica: los 9 métodos con sus firmas y su comportamiento básico."""
import inspect
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

import sistema
from recognition.bootstrap import build_service
from recognition.config import Thresholds
from recognition.service import PhotoRejectedError
from recognition.types import RecognitionStatus, VerificationStatus
from tests.synthetic import make_face, make_variant, tmp_settings

METHODS = {
    "inscribir": ["id", "foto"], "verificar": ["id", "foto"], "identificar": ["foto"],
    "inscribir_sd": ["id", "foto"], "verificar_sd": ["id", "foto"], "identificar_sd": ["foto"],
    "inscribir_t": ["id", "template"], "verificar_t": ["id", "template"],
    "identificar_t": ["template"],
}


@dataclass
class P:
    face: np.ndarray


def _png(face):
    return cv2.imencode(".png", face)[1].tobytes()


@pytest.fixture()
def s(tmp_path):
    t = Thresholds(t1=0.0, t2=0.05, orb_min_margin=0.01, t1_verify=0.0, t2_verify=0.05)
    # el «detector» de prueba decodifica la imagen (ya es la cara alineada)
    from recognition.preprocessing import load_image
    svc = build_service(tmp_settings(tmp_path, t),
                        processor=lambda img: P(cv2.cvtColor(load_image(img), cv2.COLOR_BGR2GRAY)))
    return sistema.SistemaBiometrico(svc)


@pytest.mark.parametrize("name,params", METHODS.items())
def test_signatures(name, params):
    assert list(inspect.signature(getattr(sistema, name)).parameters) == params
    assert list(inspect.signature(getattr(sistema.SistemaBiometrico, name)).parameters)[1:] == params


def test_nine_methods_work(s):
    faces = {n: make_face(i) for i, n in enumerate(["ana", "luis", "eva"])}
    s.inscribir("ana", _png(faces["ana"]))
    s.inscribir_sd("luis", _png(faces["luis"]))
    s.inscribir_t("eva", s.extraer_template(_png(faces["eva"])))
    assert s.identidades() == ["ana", "eva", "luis"]
    q = {n: _png(make_variant(f, 5)) for n, f in faces.items()}

    assert s.identificar(q["ana"]).identity == "ana"
    assert s.identificar_sd(q["luis"]).identity == "luis"
    assert s.identificar_t(s.extraer_template(q["eva"])).identity == "eva"

    assert s.verificar("ana", q["ana"]) and not s.verificar("luis", q["ana"]).accepted
    assert s.verificar_sd("luis", q["luis"]).status is VerificationStatus.ACCEPTED
    assert s.verificar_t("eva", s.extraer_template(q["eva"], detector=False)).accepted
    assert not s.verificar_t("ana", s.extraer_template(q["eva"])).accepted

    from recognition.errors import DuplicateIdentityError
    for metodo, arg in ((s.inscribir, q["ana"]), (s.inscribir_sd, q["ana"]),
                        (s.inscribir_t, s.extraer_template(q["ana"]))):
        with pytest.raises(DuplicateIdentityError):          # solo se inscriben personas nuevas
            metodo("ana", arg)
    assert s.service.store.get_identity("ana").n_templates == 1


def test_invalid_inputs(s):
    s.inscribir("ana", _png(make_face(1)))
    with pytest.raises(PhotoRejectedError):
        s.inscribir_t("x", "no-es-template")
    with pytest.raises(ValueError):
        s.verificar_t("ana", "no-es-template")
    assert s.identificar_sd(b"no es imagen").status is RecognitionStatus.ERROR
    assert s.verificar_sd("ana", b"no es imagen").status is VerificationStatus.ERROR
    assert not s.verificar("nadie", _png(make_face(1))).accepted


def test_numeric_ids_are_accepted(s):
    s.inscribir(7, _png(make_face(1)))
    assert s.verificar(7, _png(make_variant(make_face(1), 2))).accepted
    assert s.verificar_sd("7", _png(make_variant(make_face(1), 3))).accepted
    assert not s.verificar(8, _png(make_face(1))).accepted


def test_command_line(s, capsys, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(sistema, "_default", s)
    foto = tmp_path / "f.png"
    foto.write_bytes(_png(make_face(1)))
    q = tmp_path / "q.png"
    q.write_bytes(_png(make_variant(make_face(1), 4)))

    def run(*argv):
        code = sistema.main(list(argv))
        out = capsys.readouterr().out.strip()
        return code, out

    assert run("inscribir", "ana", str(foto)) == (0, json.dumps({"id": "ana", "muestras": 1}))
    code, out = run("identificar", str(q))
    assert code == 0 and json.loads(out)["identidad"] == "ana"
    code, out = run("verificar_sd", "ana", str(q))
    assert code == 0 and json.loads(out)["aceptado"] is True
    code, t = run("extraer_template", str(q))
    (tmp_path / "t.txt").write_text(t)
    code, out = run("verificar_t", "ana", "@" + str(tmp_path / "t.txt"))
    assert json.loads(out)["aceptado"] is True
    code, out = run("identificar_t", t)
    assert json.loads(out)["identidad"] == "ana"
    code, out = run("inscribir", "ana", str(foto))          # id existente
    assert code == 1 and "DuplicateIdentityError" in json.loads(out)["error"]
    code, out = run("identidades")
    assert json.loads(out) == ["ana"]

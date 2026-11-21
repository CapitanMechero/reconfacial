"""API del sistema biométrico exigida por la práctica (9 métodos).

Uso desde Python::

    import sistema
    r = sistema.identificar("foto.jpg")         # r.identity (None = no aceptada), r.candidate
    v = sistema.verificar("adele", "foto.jpg")  # v.accepted  (bool(v) == v.accepted)
    sistema.inscribir("persona_nueva", "foto.jpg")

    # o con una instancia explícita:
    s = sistema.SistemaBiometrico()
    s.identificar("foto.jpg")

Uso desde la línea de comandos (salida JSON; en Docker, ver README)::

    python -m sistema identificar foto.jpg
    python -m sistema verificar adele foto.jpg
    python -m sistema inscribir persona_nueva foto.jpg
    python -m sistema extraer_template foto.jpg > t.txt
    python -m sistema verificar_t adele @t.txt        # @archivo = template leído de un archivo

Los 9 métodos
-------------
* ``inscribir(id, foto)``: inscribe a una persona NUEVA con su nombre y una foto.
* ``verificar(id, foto)``: ¿la foto es de la persona ``id``?
* ``identificar(foto)``: ¿de quién es la foto?
* ``inscribir_sd``, ``verificar_sd``, ``identificar_sd``: igual, omitiendo el detector
  de caras (la foto se trata como una cara ya recortada: gris, 128x128, sin alinear).
* ``inscribir_t(id, template)``, ``verificar_t(id, template)``,
  ``identificar_t(template)``: igual, omitiendo detector y extractor.

Entradas
--------
* ``id``: nombre de la persona (se convierte a texto; mayúsculas indiferentes).
* ``foto``: ruta (``str``/``Path``), bytes de un JPEG/PNG o ``np.ndarray`` BGR/gris.
* ``template``: template público en base64 (``str``), sus bytes o un
  ``RegisteredFace`` (formato en ``recognition/template.py``); se obtiene con
  ``extraer_template(foto)``.

Salidas
-------
* ``inscribir*`` -> ``Identity``. Solo personas NUEVAS: un id existente lanza
  ``DuplicateIdentityError``. Otros errores: ``ValueError`` (nombre o template
  inválido), ``PhotoRejectedError`` (sin cara, varias caras, sin textura),
  ``DuplicateFaceError`` (la cara ya está inscrita como otra persona).
* ``verificar*`` -> ``VerificationResult``: ``accepted``, ``status``
  (accepted/rejected/error), ``svm_score``, ``orb_score`` y ``notes``.
* ``identificar*`` -> ``RecognitionResult``: ``identity`` (o ``None``), ``status``
  (identified/uncertain/unknown/error), ``candidate`` (el más parecido aunque no se
  acepte), ``stage`` y ``score``. Las puntuaciones NO son probabilidades.

La práctica no pide eliminar personas y este módulo no lo ofrece.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from recognition.bootstrap import build_service
from recognition.service import FaceRecognitionService
from recognition.template import TemplateInput, to_base64
from recognition.types import Identity, ImageInput, RecognitionResult, VerificationResult


class SistemaBiometrico:
    def __init__(self, service: Optional[FaceRecognitionService] = None) -> None:
        self._s = service or build_service()
        self._s.ensure_model_current()

    @property
    def service(self) -> FaceRecognitionService:
        return self._s

    # ---- con detector -----------------------------------------------------------
    def inscribir(self, id: str, foto: ImageInput) -> Identity:  # noqa: A002 - nombre exigido
        return self._s.enroll(str(id), [foto], detector=True).identity

    def verificar(self, id: str, foto: ImageInput) -> VerificationResult:  # noqa: A002
        return self._s.verify(str(id), foto)

    def identificar(self, foto: ImageInput) -> RecognitionResult:
        return self._s.identify(foto)

    # ---- sin detector (_sd) ------------------------------------------------------
    def inscribir_sd(self, id: str, foto: ImageInput) -> Identity:  # noqa: A002
        return self._s.enroll(str(id), [foto], detector=False).identity

    def verificar_sd(self, id: str, foto: ImageInput) -> VerificationResult:  # noqa: A002
        return self._s.verify_without_detector(str(id), foto)

    def identificar_sd(self, foto: ImageInput) -> RecognitionResult:
        return self._s.identify_without_detector(foto)

    # ---- con template (_t) -----------------------------------------------------------
    def inscribir_t(self, id: str, template: TemplateInput) -> Identity:  # noqa: A002
        return self._s.enroll_templates(str(id), [template]).identity

    def verificar_t(self, id: str, template: TemplateInput) -> VerificationResult:  # noqa: A002
        return self._s.verify_template(str(id), template)

    def identificar_t(self, template: TemplateInput) -> RecognitionResult:
        return self._s.identify_template(template)

    # ---- utilidades -----------------------------------------------------------------------
    def extraer_template(self, foto: ImageInput, detector: bool = True) -> str:
        """Template público (base64) de una foto, para usarlo con los métodos ``_t``."""
        return to_base64(self._s.template_from_photo(foto, detector))

    def identidades(self) -> list[str]:
        return [i.name for i in self._s.list_identities()]


_default: Optional[SistemaBiometrico] = None


def _sistema() -> SistemaBiometrico:
    global _default
    if _default is None:
        _default = SistemaBiometrico()
    return _default


def inscribir(id: str, foto: ImageInput) -> Identity:  # noqa: A002
    return _sistema().inscribir(id, foto)


def verificar(id: str, foto: ImageInput) -> VerificationResult:  # noqa: A002
    return _sistema().verificar(id, foto)


def identificar(foto: ImageInput) -> RecognitionResult:
    return _sistema().identificar(foto)


def inscribir_sd(id: str, foto: ImageInput) -> Identity:  # noqa: A002
    return _sistema().inscribir_sd(id, foto)


def verificar_sd(id: str, foto: ImageInput) -> VerificationResult:  # noqa: A002
    return _sistema().verificar_sd(id, foto)


def identificar_sd(foto: ImageInput) -> RecognitionResult:
    return _sistema().identificar_sd(foto)


def inscribir_t(id: str, template: TemplateInput) -> Identity:  # noqa: A002
    return _sistema().inscribir_t(id, template)


def verificar_t(id: str, template: TemplateInput) -> VerificationResult:  # noqa: A002
    return _sistema().verificar_t(id, template)


def identificar_t(template: TemplateInput) -> RecognitionResult:
    return _sistema().identificar_t(template)


def extraer_template(foto: ImageInput, detector: bool = True) -> str:
    return _sistema().extraer_template(foto, detector)


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
def to_dict(result: Any) -> dict:
    """Representación JSON de los resultados (también útil para registrar pruebas)."""
    if isinstance(result, Identity):
        return {"id": result.name, "muestras": result.n_templates}
    if isinstance(result, VerificationResult):
        return {"aceptado": result.accepted, "estado": result.status.value, "id": result.claimed,
                "etapa": result.stage.value, "svm_score": result.svm_score,
                "orb_score": result.orb_score, "notas": list(result.notes),
                "error": result.error.message if result.error else None}
    if isinstance(result, RecognitionResult):
        return {"identidad": result.identity, "estado": result.status.value,
                "candidato": result.candidate, "etapa": result.stage.value,
                "puntuacion": result.score, "tipo_puntuacion": result.score_kind.value,
                "notas": list(result.notes),
                "error": result.error.message if result.error else None}
    raise TypeError(type(result).__name__)


def _template_arg(value: str) -> str:
    return Path(value[1:]).read_text(encoding="utf-8").strip() if value.startswith("@") else value


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sistema", description="API de la práctica (salida JSON).")
    sub = ap.add_subparsers(dest="metodo", required=True)
    for name in ("inscribir", "verificar", "inscribir_sd", "verificar_sd"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("foto")
    for name in ("identificar", "identificar_sd"):
        sub.add_parser(name).add_argument("foto")
    for name in ("inscribir_t", "verificar_t"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("template", help="base64, o @archivo con el base64")
    sub.add_parser("identificar_t").add_argument("template", help="base64, o @archivo con el base64")
    p = sub.add_parser("extraer_template")
    p.add_argument("foto")
    p.add_argument("--sd", action="store_true", help="sin detector de caras")
    sub.add_parser("identidades")
    args = ap.parse_args(argv)

    import logging
    logging.disable(logging.ERROR)    # la salida es solo el JSON
    s = _sistema()
    m = args.metodo
    try:
        if m == "extraer_template":
            print(s.extraer_template(args.foto, detector=not args.sd))
            return 0
        if m == "identidades":
            print(json.dumps(s.identidades(), ensure_ascii=False))
            return 0
        if m.endswith("_t"):
            t = _template_arg(args.template)
            out = getattr(s, m)(args.id, t) if m != "identificar_t" else s.identificar_t(t)
        elif m.startswith("identificar"):
            out = getattr(s, m)(args.foto)
        else:
            out = getattr(s, m)(args.id, args.foto)
    except (ValueError, LookupError, OSError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(to_dict(out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Interfaz Streamlit. Solo presentación: toda la lógica vive en ``recognition.service``.

Ofrece exactamente las operaciones de la práctica: identificar, verificar e inscribir
(con y sin detector). No hay eliminación: la práctica no la pide y exponerla
permitiría borrar a una persona e inscribirse con su nombre (V-01/V-02).
"""
from __future__ import annotations

import logging

import cv2
import streamlit as st

from recognition.bootstrap import build_service
from recognition.config import configure_logging, load_settings
from recognition.detector import draw_detection
from recognition.errors import (
    KeyConfigurationError, ModelNotFoundError, StorageError,
    TemplateCorruptError,
)
from recognition.model_lifecycle import ModelStatus
from recognition.preprocessing import load_image
from recognition.service import (
    DuplicateFaceError, FaceRecognitionService, PhotoRejectedError, face_without_detector,
)
from recognition.types import RecognitionStatus, ScoreKind, Stage, VerificationStatus

MAX_PHOTOS = 10
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
STATUS_TEXT = {
    ModelStatus.UP_TO_DATE: "SVM al día",
    ModelStatus.STALE: "SVM desactualizado (T1 desactivado hasta reentrenar)",
    ModelStatus.MISSING: "SVM no entrenado",
    ModelStatus.INVALID: "SVM no verificado (T1 desactivado)",
    ModelStatus.NOT_ENOUGH_IDENTITIES: "SVM inactivo: se necesitan ≥ 2 identidades",
}
SCORE_LABEL = {
    ScoreKind.SVM_DECISION: "Margen de decisión del SVM",
    ScoreKind.ORB_INLIER_RATIO: "Fracción de puntos ORB con coincidencia geométrica",
}
STAGE_TEXT = {Stage.HOG_SVM: "HOG + SVM (T1)", Stage.ORB: "ORB (T2)", Stage.NONE: "—"}


@st.cache_resource(show_spinner="Cargando la base de datos y los modelos…")
def get_service() -> FaceRecognitionService:
    settings = load_settings()
    configure_logging(settings.log_level)
    service = build_service(settings)
    service.ensure_model_current()   # entrena el SVM en el primer arranque / tras interrupciones
    from recognition.detector import get_default_detector
    get_default_detector()           # carga MediaPipe ya: los fallos se ven al arrancar, no al subir una foto
    return service


def show_training(report) -> None:
    for w in report.warnings:
        st.warning(w)
    if report.error:
        st.error(f"No se pudo reentrenar el SVM: {report.error}")
    elif report.trained:
        st.caption(f"SVM reentrenado con {report.n_samples} muestras de "
                   f"{report.n_identities} identidades.")


def uncalibrated_warning(calibrated: bool) -> None:
    if not calibrated:
        st.warning("Umbrales provisionales (sin calibrar): no interpretes el resultado como una "
                   "garantía. Las puntuaciones no son probabilidades.")


def photo_input(key: str):
    upload = st.file_uploader("Fotografía", type=IMAGE_TYPES, key=f"{key}_file")
    detector = st.toggle("Usar detector de caras", value=True, key=f"{key}_det",
                         help="Desactívalo (modo _sd) si la imagen ya es un recorte de la cara.")
    return upload, detector


def preview(data: bytes, detector: bool, processed=None) -> None:
    if processed is not None:
        img = draw_detection(load_image(data), processed)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Cara detectada", width=320)
    elif not detector:
        st.image(face_without_detector(data), caption="Imagen usada como cara (sin detector)",
                 width=160)


def tab_identify(service: FaceRecognitionService) -> None:
    st.subheader("¿Quién es?")
    upload, detector = photo_input("idf")
    if upload is None or not st.button("Identificar", type="primary"):
        return
    data = upload.getvalue()
    with st.spinner("Analizando…"):
        if detector:
            analysis = service.analyze(data)
            res, processed = analysis.result, analysis.processed
        else:
            res, processed = service.identify_without_detector(data), None
    if res.status is RecognitionStatus.ERROR:
        st.error(res.error.message)
        return
    preview(data, detector, processed)
    if res.status is RecognitionStatus.IDENTIFIED:
        st.success(f"Es **{res.identity}**")
    elif res.status is RecognitionStatus.UNCERTAIN:
        st.warning("Resultado incierto: no se puede confirmar ninguna identidad.")
    else:
        st.info("Persona desconocida (ninguna coincidencia supera los umbrales).")
    if res.candidate and not res.is_identified:
        st.write(f"Candidato más parecido (no aceptado): **{res.candidate}**")
    st.write(f"**Algoritmo decisivo:** {STAGE_TEXT[res.stage]}")
    if res.score is not None:
        st.write(f"**{SCORE_LABEL[res.score_kind]}:** {res.score:.3f}")
    for note in res.notes:
        st.caption(note)
    uncalibrated_warning(res.thresholds_calibrated)


def tab_verify(service: FaceRecognitionService) -> None:
    st.subheader("¿Es quien dice ser?")
    names = [i.name for i in service.list_identities()]
    if not names:
        st.info("No hay identidades inscritas.")
        return
    claimed = st.selectbox("Identidad que se reclama", names, key="ver_name")
    upload, detector = photo_input("ver")
    if upload is None or not st.button("Verificar", type="primary"):
        return
    data = upload.getvalue()
    with st.spinner("Verificando…"):
        res = (service.verify(claimed, data) if detector
               else service.verify_without_detector(claimed, data))
    if res.status is VerificationStatus.ERROR:
        st.error(res.error.message)
        return
    preview(data, detector)
    if res.accepted:
        st.success(f"Correcto: la foto corresponde a **{res.claimed}**")
    else:
        st.error(f"Incorrecto: la foto no corresponde a **{claimed}**")
    st.write(f"**Algoritmo decisivo:** {STAGE_TEXT[res.stage]}")
    if res.svm_score is not None:
        st.write(f"**Margen del SVM para {res.claimed}:** {res.svm_score:.3f} "
                 f"(T1v = {service.settings.thresholds.t1_verify:g})")
    if res.orb_score is not None:
        st.write(f"**Puntuación ORB:** {res.orb_score:.3f} "
                 f"(T2v = {service.settings.thresholds.t2_verify:g})")
    for note in res.notes:
        st.caption(note)
    uncalibrated_warning(res.thresholds_calibrated)


def tab_enroll(service: FaceRecognitionService) -> None:
    st.subheader("Inscribir a una persona")
    name = st.text_input("Identificador (nombre)", max_chars=100)
    uploads = st.file_uploader(f"Fotografías (hasta {MAX_PHOTOS})", type=IMAGE_TYPES,
                               accept_multiple_files=True, key="enr_files")
    detector = st.toggle("Usar detector de caras", value=True, key="enr_det")
    exists = bool(name.strip()) and any(
        i.name.casefold() == name.strip().casefold() for i in service.list_identities())
    if exists:
        st.warning("Esa identidad ya existe. Solo se pueden inscribir personas nuevas: "
                   "elige otro identificador.")
    allow_dup = st.checkbox("Inscribir aunque la cara ya coincida con otra identidad")
    if not st.button("Inscribir", type="primary"):
        return
    if not name.strip():
        st.error("Indica un identificador.")
    elif exists:
        st.error("Esa identidad ya existe; no se ha inscrito nada.")
    elif not uploads:
        st.error("Selecciona al menos una fotografía.")
    elif len(uploads) > MAX_PHOTOS:
        st.error(f"Máximo {MAX_PHOTOS} fotografías.")
    else:
        try:
            with st.spinner("Procesando y reentrenando…"):
                out = service.enroll(name, [u.getvalue() for u in uploads], detector=detector,
                                     allow_duplicate_face=allow_dup)
        except (PhotoRejectedError, DuplicateFaceError, ValueError) as exc:
            st.error(str(exc))
            return
        except (TemplateCorruptError, StorageError) as exc:
            st.error(f"Error de la base de datos: {exc}")
            return
        st.success(f"«{out.identity.name}» inscrita con {out.identity.n_templates} foto(s). "
                   "Las imágenes no se han guardado.")
        show_training(out.training)


def sidebar(service: FaceRecognitionService) -> None:
    st.sidebar.header("Estado")
    st.sidebar.write(f"Identidades inscritas: **{len(service.list_identities())}**")
    status = service.model_status()
    (st.sidebar.success if status is ModelStatus.UP_TO_DATE else st.sidebar.warning)(
        STATUS_TEXT[status])
    t = service.settings.thresholds
    st.sidebar.write(f"Identificación: T1 = {t.t1:g} · T2 = {t.t2:g}")
    st.sidebar.write(f"Verificación: T1v = {t.t1_verify:g} · T2v = {t.t2_verify:g}")
    if t.calibrated:
        st.sidebar.caption(f"Umbrales calibrados: {t.source}")
    else:
        st.sidebar.warning("Umbrales sin calibrar (provisionales).")
    if status in (ModelStatus.STALE, ModelStatus.INVALID, ModelStatus.MISSING):
        if st.sidebar.button("Reentrenar SVM"):
            show_training(service.retrain())
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Reconocimiento facial", page_icon="🙂")
    st.title("Reconocimiento facial")
    try:
        service = get_service()
    except TemplateCorruptError as exc:
        st.error(f"Fallo de integridad de la base de datos: {exc}")
        st.stop()
        return   # sin contexto de Streamlit (p. ej. al importarse), st.stop() no detiene
    except (KeyConfigurationError, StorageError) as exc:
        st.error(f"No se puede abrir la base de datos: {exc}")
        st.stop()
        return   # sin contexto de Streamlit (p. ej. al importarse), st.stop() no detiene
    except (OSError, ModelNotFoundError) as exc:
        st.error(f"No se puede cargar el detector de caras (MediaPipe): {exc}")
        st.stop()
        return   # sin contexto de Streamlit (p. ej. al importarse), st.stop() no detiene
    except Exception:  # noqa: BLE001 - última barrera de la UI: se registra completa
        logging.getLogger(__name__).exception("Fallo al iniciar la aplicación")
        st.error("Error inesperado al iniciar la aplicación; revisa los registros.")
        st.stop()
        return   # sin contexto de Streamlit (p. ej. al importarse), st.stop() no detiene

    try:
        sidebar(service)
        idf, ver, enr = st.tabs(["Identificar", "Verificar", "Inscribir"])
        with idf:
            tab_identify(service)
        with ver:
            tab_verify(service)
        with enr:
            tab_enroll(service)
    except TemplateCorruptError as exc:     # la BD se manipuló mientras la app estaba abierta
        st.error(f"Fallo de integridad de la base de datos: {exc}")


main()

"""Pruebas de la interfaz con ``streamlit.testing`` (sin navegador)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from recognition.config import ENV_PREFIX, PROJECT_ROOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    if not (PROJECT_ROOT / "models" / "face_landmarker.task").is_file():
        pytest.skip("falta el modelo de MediaPipe")
    monkeypatch.setenv(ENV_PREFIX + "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(ENV_PREFIX + "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv(ENV_PREFIX + "SEED_DB", str(tmp_path / "sin_semilla.csv"))
    from streamlit.runtime.caching import cache_resource
    cache_resource.clear()        # el servicio se cachea por proceso: aislar cada test
    yield monkeypatch
    cache_resource.clear()


def run():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def test_app_renders_tabs_with_empty_database(env):
    at = run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Identificar", "Verificar", "Inscribir"]
    assert any("No hay identidades" in i.value for i in at.info)
    assert not any("Eliminar" in b.label for b in at.button)   # la práctica no pide eliminar


def test_tampered_database_shows_integrity_error(env, tmp_path):
    run()                                                   # primer arranque: crea clave y BD
    (tmp_path / "state" / "manifest.json").write_text("{}")
    from streamlit.runtime.caching import cache_resource
    cache_resource.clear()
    at = run()
    assert not at.exception
    assert any("integridad" in e.value.lower() or "manifiesto" in e.value.lower() for e in at.error)


def test_detector_failure_shows_error_instead_of_crashing(env, tmp_path):
    env.setenv("FACE_LANDMARKER_MODEL", str(tmp_path / "no_existe.task"))
    from recognition.detector import get_default_detector
    get_default_detector.cache_clear()
    try:
        at = run()
    finally:
        get_default_detector.cache_clear()
    assert not at.exception
    assert any("detector de caras" in e.value for e in at.error)
    assert not at.tabs

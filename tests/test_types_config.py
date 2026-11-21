import json

import pytest

from recognition.config import ConfigError, load_settings
from recognition.types import (
    ErrorCode, HogPrediction, ProcessingError, RecognitionResult, RecognitionStatus,
)


def test_identified_requires_identity():
    with pytest.raises(ValueError):
        RecognitionResult(status=RecognitionStatus.IDENTIFIED)
    with pytest.raises(ValueError):
        RecognitionResult(status=RecognitionStatus.UNKNOWN, identity="ana")


def test_error_requires_error_info():
    with pytest.raises(ValueError):
        RecognitionResult(status=RecognitionStatus.ERROR)
    err = ProcessingError(ErrorCode.NO_FACE, "sin cara")
    assert RecognitionResult(status=RecognitionStatus.ERROR, error=err).error is err
    with pytest.raises(ValueError):
        RecognitionResult(status=RecognitionStatus.UNCERTAIN, error=err)


def test_hog_margin():
    assert HogPrediction("a", 1.0, 0.25).margin == 0.75
    assert HogPrediction("a", 1.0).margin is None


def test_defaults_are_flagged_uncalibrated(tmp_path):
    s = load_settings({"FACEREC_SEED_DB": str(tmp_path / "famosos.csv")})
    assert not s.thresholds_calibrated


def test_thresholds_file_then_env_override(tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps(
        {"t1": 0.3, "t2": 0.1, "t1_verify": 0.2, "t2_verify": 0.04, "source": "x"}))
    env = {"FACEREC_SEED_DB": str(tmp_path / "famosos.csv")}
    s = load_settings(env)
    assert (s.thresholds.t1, s.thresholds.t2, s.thresholds_calibrated) == (0.3, 0.1, True)
    assert (s.thresholds.t1_verify, s.thresholds.t2_verify) == (0.2, 0.04)
    s = load_settings({**env, "FACEREC_T1": "0.9"})
    assert s.thresholds.t1 == 0.9 and not s.thresholds_calibrated


def test_invalid_config_raises(tmp_path):
    env = {"FACEREC_SEED_DB": str(tmp_path / "famosos.csv")}
    with pytest.raises(ConfigError):
        load_settings({**env, "FACEREC_T2": "abc"})
    (tmp_path / "thresholds.json").write_text("{roto")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_runtime_svm_never_collides_with_versioned_legacy_model():
    from recognition.svm import DEFAULT_MODEL
    s = load_settings({})
    assert s.paths.svm_model.resolve() != DEFAULT_MODEL.resolve()

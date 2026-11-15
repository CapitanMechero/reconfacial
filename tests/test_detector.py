import cv2
import numpy as np
import pytest

from recognition.detector import FaceDetection, process_image
from recognition.errors import FaceTooSmallError, MultipleFacesError
from recognition.detector import DEFAULT_MODEL_PATH, detect_and_align
from recognition.errors import InvalidImageError, NoFaceDetectedError
from recognition.preprocessing import (
    DEFAULT_CONFIG,
    PreprocessingConfig,
    align_and_crop,
    compute_alignment_matrix,
    eye_centers,
    load_image,
    normalize_illumination,
)

needs_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.is_file(), reason="Falta models/face_landmarker.task"
)


def _fake_landmarks(left_center, right_center, swap=False):
    lm = np.zeros((478, 2), dtype=np.float32)
    lx, ly = left_center
    rx, ry = right_center
    a, b = ((rx, ry), (lx, ly)) if swap else ((lx, ly), (rx, ry))
    lm[33], lm[133] = (a[0] - 10, a[1]), (a[0] + 10, a[1])
    lm[362], lm[263] = (b[0] - 10, b[1]), (b[0] + 10, b[1])
    return lm


# ---------------------------- preprocesamiento ---------------------------- #
@pytest.mark.parametrize(
    "left,right",
    [((100, 120), (180, 120)), ((100, 140), (180, 100)), ((90, 100), (190, 150)), ((300, 200), (340, 210))],
)
def test_alignment_places_eyes_in_canonical_positions(left, right):
    cfg = DEFAULT_CONFIG
    m = compute_alignment_matrix(left, right, cfg)
    size = cfg.output_size
    out_left = m @ np.array([*left, 1.0])
    out_right = m @ np.array([*right, 1.0])
    np.testing.assert_allclose(out_left, [cfg.left_eye_x * size, cfg.eye_y * size], atol=1e-6)
    np.testing.assert_allclose(out_right, [cfg.right_eye_x * size, cfg.eye_y * size], atol=1e-6)


def test_eye_centers_are_ordered_left_to_right():
    for swap in (False, True):
        left, right = eye_centers(_fake_landmarks((100, 100), (200, 100), swap=swap))
        assert left[0] < right[0]
        np.testing.assert_allclose(left, [100, 100])
        np.testing.assert_allclose(right, [200, 100])


def test_align_and_crop_output_shape_and_dtype():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face = align_and_crop(image, (250, 200), (350, 210))
    assert face.shape == (128, 128)
    assert face.dtype == np.uint8


def test_clahe_keeps_shape_and_is_optional():
    face = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    assert normalize_illumination(face, DEFAULT_CONFIG) is face
    out = normalize_illumination(face, PreprocessingConfig(use_clahe=True))
    assert out.shape == face.shape and out.dtype == np.uint8


# ------------------------------ carga de imagen --------------------------- #
def test_load_image_rejects_corrupt_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"esto no es una imagen")


def test_load_image_rejects_empty_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"")


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "no_existe.jpg")


def test_load_image_rejects_tiny_image():
    with pytest.raises(InvalidImageError):
        load_image(np.zeros((10, 10, 3), dtype=np.uint8))


def test_load_image_accepts_gray_and_bgra_arrays():
    assert load_image(np.zeros((64, 64), dtype=np.uint8)).shape == (64, 64, 3)
    assert load_image(np.zeros((64, 64, 4), dtype=np.uint8)).shape == (64, 64, 3)


def test_load_image_decodes_png_bytes():
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    assert load_image(buf.tobytes()).shape == (64, 64, 3)


# ------------------------- detección (requiere modelo) -------------------- #
@needs_model
def test_blank_image_raises_no_face():
    with pytest.raises(NoFaceDetectedError):
        detect_and_align(np.full((480, 640, 3), 128, dtype=np.uint8))


@needs_model
def test_corrupt_file_raises_controlled_error():
    with pytest.raises(InvalidImageError):
        detect_and_align(b"corrupto")

# ------------------------ política de una sola cara ----------------------- #
class _FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image_bgr):
        return self._detections


def _fake_detection(size=100, x=10, y=10):
    lm = _fake_landmarks((x + 30, y + 40), (x + 70, y + 40))
    return FaceDetection((x, y, size, size), lm)


_IMAGE = np.full((240, 320, 3), 128, dtype=np.uint8)


def test_single_face_is_processed():
    result = process_image(_IMAGE, detector=_FakeDetector([_fake_detection()]))
    assert result.face.shape == (128, 128)
    assert result.bbox == (10, 10, 100, 100)


def test_multiple_faces_raise_error_with_count():
    detector = _FakeDetector([_fake_detection(x=10), _fake_detection(x=150)])
    with pytest.raises(MultipleFacesError) as exc:
        process_image(_IMAGE, detector=detector)
    assert exc.value.count == 2


def test_small_background_face_is_ignored():
    detector = _FakeDetector([_fake_detection(size=100), _fake_detection(size=20, x=200)])
    result = process_image(_IMAGE, detector=detector)
    assert result.bbox[2] == 100


def test_only_tiny_faces_raise_too_small():
    with pytest.raises(FaceTooSmallError):
        process_image(_IMAGE, detector=_FakeDetector([_fake_detection(size=20)]))
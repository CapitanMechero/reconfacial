import numpy as np
import pytest

from recognition.orb import OrbTemplate
from recognition.template import (
    TemplateFormatError, as_template, extract_template, from_base64, from_bytes, to_base64, to_bytes,
)
from recognition.types import RegisteredFace
from tests.synthetic import make_face

T = extract_template(make_face(1))


def test_roundtrip_all_representations():
    for value in (T, to_bytes(T), bytearray(to_bytes(T)), to_base64(T)):
        t = as_template(value)
        assert np.array_equal(t.hog, T.hog) and np.array_equal(t.orb.descriptors, T.orb.descriptors)
    assert to_bytes(T)[:4] == b"FRT\x01"


@pytest.mark.parametrize("bad", [b"", b"FRT\x01", b"XXX\x01" + to_bytes(T)[4:],
                                 b"FRT\x09" + to_bytes(T)[4:], to_bytes(T)[:-5]])
def test_malformed_bytes(bad):
    with pytest.raises(TemplateFormatError):
        from_bytes(bad)


@pytest.mark.parametrize("hog", [np.full(8100, 1e30), np.full(8100, -0.1), np.full(8100, np.nan),
                                 np.zeros(10)])
def test_out_of_range_hog(hog):
    with pytest.raises(TemplateFormatError):
        as_template(RegisteredFace(hog.astype(np.float32), T.orb))


def test_orb_points_outside_face_and_empty():
    kps = T.orb.keypoints.copy()
    kps[0, 0] = 500
    with pytest.raises(TemplateFormatError):
        as_template(RegisteredFace(T.hog, OrbTemplate(kps, T.orb.descriptors)))
    with pytest.raises(TemplateFormatError):
        as_template(RegisteredFace(T.hog, OrbTemplate(np.zeros((0, 5), np.float32),
                                                       np.zeros((0, 32), np.uint8))))


def test_other_inputs():
    with pytest.raises(TemplateFormatError):
        from_base64("no base64!!")
    with pytest.raises(TemplateFormatError):
        as_template(123)

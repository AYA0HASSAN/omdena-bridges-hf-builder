import numpy as np
import pytest

from pipeline.postprocess import obb_corners, postprocess


def test_postprocess_synthetic_single_class_1_6_n():
    """The exact layout confirmed against a real yolo26s-obb ONNX export: (1, 6, N)."""
    n = 20
    raw = np.zeros((1, 6, n), dtype=np.float32)
    raw[0, :, 5] = [120.0, 80.0, 30.0, 15.0, 0.9, 0.2]
    dets = postprocess(raw, {"confidence_threshold": 0.25, "iou_threshold": 0.45})

    assert len(dets) == 1
    d = dets[0]
    assert d["cx"] == pytest.approx(120.0)
    assert d["cy"] == pytest.approx(80.0)
    assert d["w"] == pytest.approx(30.0)
    assert d["h"] == pytest.approx(15.0)
    assert d["confidence"] == pytest.approx(0.9)
    assert d["angle"] == pytest.approx(0.2)
    assert d["class_id"] == 0
    assert len(obb_corners(d)) == 4


def test_postprocess_below_threshold_returns_empty():
    raw = np.zeros((1, 6, 10), dtype=np.float32)
    raw[0, 4, :] = 0.05
    assert postprocess(raw, {"confidence_threshold": 0.25}) == []


def test_postprocess_handles_channels_last_layout():
    raw_channels_first = np.zeros((1, 6, 10), dtype=np.float32)
    raw_channels_first[0, :, 2] = [50.0, 60.0, 10.0, 8.0, 0.8, -0.3]
    raw_channels_last = np.transpose(raw_channels_first, (0, 2, 1))

    dets = postprocess(raw_channels_last, {"confidence_threshold": 0.25})
    assert len(dets) == 1
    assert dets[0]["cx"] == pytest.approx(50.0)


def test_postprocess_reads_nc_from_channel_count():
    """nc = C - 5, not hard-coded: a 3-class (C=8) tensor should decode with argmax over classes.

    N (anchor count) must stay comfortably above C (channel count) - see postprocess()'s docstring -
    so N=20 here, not some small number close to C=8.
    """
    n = 20
    nc = 3
    raw = np.zeros((1, 4 + nc + 1, n), dtype=np.float32)
    raw[0, :, 0] = [100.0, 100.0, 20.0, 10.0, 0.1, 0.2, 0.7, 0.0]  # class 2 scores highest
    dets = postprocess(raw, {"confidence_threshold": 0.5})
    assert len(dets) == 1
    assert dets[0]["class_id"] == 2
    assert dets[0]["confidence"] == pytest.approx(0.7)


def test_rotated_nms_suppresses_overlapping_lower_confidence():
    n = 20  # N must stay well above C=6 - see postprocess()'s docstring
    raw = np.zeros((1, 6, n), dtype=np.float32)
    raw[0, :, 0] = [10.0, 10.0, 4.0, 4.0, 0.9, 0.0]
    raw[0, :, 1] = [10.1, 10.1, 4.0, 4.0, 0.5, 0.0]  # near-duplicate of the first, lower confidence
    dets = postprocess(raw, {"confidence_threshold": 0.25, "iou_threshold": 0.45})
    assert len(dets) == 1
    assert dets[0]["confidence"] == pytest.approx(0.9)

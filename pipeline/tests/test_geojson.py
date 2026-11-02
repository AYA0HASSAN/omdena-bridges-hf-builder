import json

import numpy as np
import pytest
from PIL import Image

from pipeline import serve as serve_mod
from pipeline.detector import Detections
from pipeline.serve import _angle_from_corners


class _FakeDetector:
    """Stands in for OnnxDetector so this test needs no real ONNX model - it only checks that
    serve() turns Detections into valid, correctly-shaped GeoJSON."""

    def __init__(self, *_a, **_k):
        self.imgsz = 64

    def predict(self, chips, conf=0.25, iou=0.45):
        corners = np.array([[[10, 10], [40, 10], [40, 30], [10, 30]]], dtype=float)
        return [Detections(corners=corners, scores=np.array([0.8]), class_ids=np.array([0])) for _ in chips]


def test_serve_writes_valid_geojson(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_mod, "OnnxDetector", _FakeDetector)
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(chips_dir / "OAM-100-200-18.png")

    out = tmp_path / "out.geojson"
    serve_mod.serve(
        model_path="unused.onnx", chips_dir=str(chips_dir), out=str(out), conf=0.25, iou=0.45, batch=4, imgsz=64
    )

    fc = json.loads(out.read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1

    feat = fc["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # closed ring, per RFC 7946
    assert set(feat["properties"]) >= {"class", "label", "confidence", "angle", "chip_id"}
    lon, lat = ring[0]
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90


def test_serve_skips_non_oam_named_chips(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_mod, "OnnxDetector", _FakeDetector)
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(chips_dir / "not_oam_named.png")

    out = tmp_path / "out.geojson"
    serve_mod.serve(
        model_path="unused.onnx", chips_dir=str(chips_dir), out=str(out), conf=0.25, iou=0.45, batch=4, imgsz=64
    )
    fc = json.loads(out.read_text())
    assert fc["features"] == []


def test_angle_from_corners_axis_aligned():
    assert _angle_from_corners([(0, 0), (10, 0), (10, 5), (0, 5)]) == pytest.approx(0.0)

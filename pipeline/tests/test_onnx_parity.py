"""ONNX/PyTorch parity, three levels, using a tiny nc=1 model built from the stock
`yolo26n-obb.yaml` architecture (random weights - nothing downloaded, no HF_TOKEN needed, fast
enough for CI). A second test with the real `AyaHassan/fair-bridges-yolo` weights runs only when
HF_TOKEN is set (`@pytest.mark.slow`, excluded from CI by `-m "not slow"`).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _nc1_model(tmp_path: Path):
    """Ultralytics YOLO, scale 'n', task obb, with nc overridden to 1 - random weights, no network
    access.

    There is no literal `yolo26n-obb.yaml` file in the package: ultralytics' own `yaml_model_load`
    strips the scale letter, loads the base `cfg/models/26/yolo26-obb.yaml` (nc=80, a `scales:` dict
    for n/s/m/l/x), and re-attaches `scale: "n"` from the originally requested name - verified by
    reading ultralytics/nn/tasks.py's `yaml_model_load`/`guess_model_scale` in this exact installed
    version. We call that same function, override `nc`, and dump the result back to a file named
    the same way (`yolo26n-obb.yaml`) so `guess_model_scale` re-derives "n" correctly on reload.
    """
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load

    cfg = yaml_model_load("yolo26n-obb.yaml")
    cfg["nc"] = 1
    custom = tmp_path / "yolo26n-obb.yaml"
    custom.write_text(yaml.safe_dump(cfg))
    return YOLO(str(custom), task="obb")


@pytest.fixture(scope="module")
def nc1_onnx(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("nc1_model")
    model = _nc1_model(tmp_path)
    onnx_path = model.export(format="onnx", imgsz=64, opset=17, simplify=True)
    return model, Path(onnx_path)


def _torch_raw(model, x):
    import torch

    model.model.eval()
    with torch.no_grad():
        raw = model.model(x)
    return (raw[0] if isinstance(raw, (list, tuple)) else raw).numpy()


def test_raw_tensor_parity(nc1_onnx):
    import onnxruntime as ort
    import torch

    model, onnx_path = nc1_onnx
    x = torch.rand(1, 3, 64, 64)
    torch_raw = _torch_raw(model, x)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_raw = sess.run(None, {input_name: x.numpy()})[0]

    assert torch_raw.shape == onnx_raw.shape
    assert np.abs(torch_raw - onnx_raw).max() < 1e-3


def test_postprocess_parity(nc1_onnx):
    import onnxruntime as ort
    import torch

    from pipeline.postprocess import obb_corners, postprocess

    model, onnx_path = nc1_onnx
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    for _ in range(10):
        x_np = rng.random((1, 3, 64, 64), dtype=np.float32)
        torch_raw = _torch_raw(model, torch.from_numpy(x_np))
        onnx_raw = sess.run(None, {input_name: x_np})[0]

        torch_dets = sorted(
            postprocess(torch_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45}),
            key=lambda d: -d["confidence"],
        )
        onnx_dets = sorted(
            postprocess(onnx_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45}),
            key=lambda d: -d["confidence"],
        )
        assert len(torch_dets) == len(onnx_dets)
        for td, od in zip(torch_dets, onnx_dets):
            assert abs(td["confidence"] - od["confidence"]) < 1e-3
            assert _rotated_iou(obb_corners(td), obb_corners(od)) >= 0.99


def _rotated_iou(corners_a, corners_b) -> float:
    from shapely.geometry import Polygon

    pa, pb = Polygon(corners_a), Polygon(corners_b)
    if not pa.is_valid or not pb.is_valid or pa.area == 0 or pb.area == 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def test_geojson_parity(nc1_onnx):
    """Same (already parity-checked) detections -> the same GeoJSON geometry from each backend."""
    import onnxruntime as ort
    import torch

    from pipeline.geo import chip_corners_to_lonlat
    from pipeline.postprocess import obb_corners, postprocess

    model, onnx_path = nc1_onnx
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(1)
    x_np = rng.random((1, 3, 64, 64), dtype=np.float32)
    torch_raw = _torch_raw(model, torch.from_numpy(x_np))
    onnx_raw = sess.run(None, {input_name: x_np})[0]

    torch_dets = sorted(
        postprocess(torch_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45}), key=lambda d: -d["confidence"]
    )
    onnx_dets = sorted(
        postprocess(onnx_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45}), key=lambda d: -d["confidence"]
    )
    assert len(torch_dets) == len(onnx_dets)

    def to_features(dets):
        feats = []
        for d in dets:
            lonlat = chip_corners_to_lonlat(obb_corners(d), 10, 20, 14, 64, 64)
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"confidence": round(d["confidence"], 4)},
                    "geometry": {"type": "Polygon", "coordinates": [[*lonlat, lonlat[0]]]},
                }
            )
        return feats

    torch_feats, onnx_feats = to_features(torch_dets), to_features(onnx_dets)
    assert len(torch_feats) == len(onnx_feats)
    for tf, of in zip(torch_feats, onnx_feats):
        assert tf["properties"]["confidence"] == pytest.approx(of["properties"]["confidence"], abs=1e-3)
        t_ring = np.array(tf["geometry"]["coordinates"][0])
        o_ring = np.array(of["geometry"]["coordinates"][0])
        assert np.abs(t_ring - o_ring).max() < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN to download real weights")
def test_real_weights_parity(tmp_path):
    """Downloads AyaHassan/fair-bridges-yolo's real ONNX + PyTorch weights and runs the same checks.
    Not run in CI; run this on Kaggle (or anywhere HF_TOKEN is set)."""
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    from pipeline.postprocess import obb_corners, postprocess

    onnx_path = hf_hub_download("AyaHassan/fair-bridges-yolo", "v0/bridges-v0.onnx")
    pt_path = hf_hub_download("AyaHassan/fair-bridges-yolo", "exp2_clean_512/stage2/best.pt")

    model = YOLO(pt_path, task="obb")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    for _ in range(10):
        x_np = rng.random((1, 3, 512, 512), dtype=np.float32)
        torch_raw = _torch_raw(model, torch.from_numpy(x_np))
        onnx_raw = sess.run(None, {input_name: x_np})[0]
        assert np.abs(torch_raw - onnx_raw).max() < 1e-3

        torch_dets = postprocess(torch_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45})
        onnx_dets = postprocess(onnx_raw, {"confidence_threshold": 0.001, "iou_threshold": 0.45})
        assert len(torch_dets) == len(onnx_dets)
        for td, od in zip(
            sorted(torch_dets, key=lambda d: -d["confidence"]), sorted(onnx_dets, key=lambda d: -d["confidence"])
        ):
            assert abs(td["confidence"] - od["confidence"]) < 1e-3
            assert _rotated_iou(obb_corners(td), obb_corners(od)) >= 0.99

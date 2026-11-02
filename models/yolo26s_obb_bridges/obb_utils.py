"""OBB pre/post-processing helpers with no zenml/fair dependency, so both the fAIr pipeline.py and the
lean pipeline/ serving tool can import this module without pulling in ZenML.

ONNX output layout - CONFIRMED (not just assumed) two ways: (1) against a real exported
yolo26s-obb ONNX graph (`bridges-v0.onnx`): input "images" is (batch, 3, height, width), output
"output0" is (batch, 6, anchors) with no NonMaxSuppression/TopK node in the graph; (2) against
`yolo26n-obb.yaml` (nc=80 random weights -> (1, 85, 5376)) and `yolo26n-obb.pt` (DOTA, nc=15 ->
(1, 20, 5376)) at imgsz=512, opset=17, ultralytics 8.4.197. Channels = 4 box (cx, cy, w, h, already
decoded to input-pixel units) + nc class scores (already sigmoid-applied) + 1 angle (radians, last
channel); N = 64^2 + 32^2 + 16^2 = 5376 at 512. The export is raw (not end-to-end): no NMS is baked
in, so `postprocess` always applies its own confidence filter + rotated NMS.
"""

import math
from typing import Any

CLASS_NAME = "bridge"
CLASS_VALUE = 0
MODEL_IMGSZ = 512  # ONNX input side; fAIr's native chip is 256x256, resized up before inference


def preprocess(image_path: Any) -> Any:
    """One RGB chip as a normalised (1, 3, MODEL_IMGSZ, MODEL_IMGSZ) float32 batch."""
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(image_path) as src:
        arr = src.read([1, 2, 3])
    rgb = np.ascontiguousarray(np.transpose(arr, (1, 2, 0))).astype(np.uint8)
    resized = Image.fromarray(rgb, mode="RGB").resize((MODEL_IMGSZ, MODEL_IMGSZ), Image.Resampling.BILINEAR)
    return np.asarray(resized).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0


def obb_corners(det: dict[str, float]) -> list[tuple[float, float]]:
    cx, cy, w, h, angle = det["cx"], det["cy"], det["w"], det["h"], det["angle"]
    dx, dy = w / 2, h / 2
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a) for lx, ly in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy))]


def rotated_nms(dets: list[dict[str, float]], iou_threshold: float) -> list[dict[str, float]]:
    from shapely.geometry import Polygon

    ordered = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    polys = [Polygon(obb_corners(d)) for d in ordered]
    keep: list[int] = []
    for i, poly_i in enumerate(polys):
        if not poly_i.is_valid or poly_i.area <= 0:
            continue
        if any(
            (union := poly_i.area + polys[j].area - poly_i.intersection(polys[j]).area) > 0
            and poly_i.intersection(polys[j]).area / union > iou_threshold
            for j in keep
        ):
            continue
        keep.append(i)
    return [ordered[i] for i in keep]


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Raw YOLO26-OBB ONNX output -> detections (cx, cy, w, h, angle radians, confidence, class_id) in
    model-input pixel space (confidence filter + rotated NMS; no CRS involved).

    `nc` (number of classes) is read from the channel count (C = 4 box + nc scores + 1 angle, so
    nc = C - 5) rather than hard-coded, so this also works for a multi-class export, not just our
    single-class (nc=1, C=6) bridge model.

    The channels-first-vs-channels-last check below assumes the anchor count N is larger than the
    channel count C, which always holds for a real YOLO head (N is in the thousands; C is at most a
    few hundred even for an 80-class model) but is ambiguous for a hand-built tensor with very few
    anchors - keep N > C in tests.
    """
    import numpy as np

    arr = np.asarray(raw_output)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[1] <= arr.shape[2]:  # (batch, channels, N) -> (batch, N, channels)
        arr = np.transpose(arr, (0, 2, 1))
    rows = arr[0]
    nc = rows.shape[-1] - 5
    if nc < 1:
        return []

    params = params or {}
    conf_threshold = float(params.get("confidence_threshold", 0.25))
    iou_threshold = float(params.get("iou_threshold", 0.45))

    box, scores, angle = rows[:, :4], rows[:, 4 : 4 + nc], rows[:, 4 + nc]
    class_id = scores.argmax(axis=1)
    confidence = scores[np.arange(len(scores)), class_id]
    keep = confidence >= conf_threshold
    if not keep.any():
        return []
    dets = [
        {
            "cx": float(box[i, 0]),
            "cy": float(box[i, 1]),
            "w": float(box[i, 2]),
            "h": float(box[i, 3]),
            "angle": float(angle[i]),
            "confidence": float(confidence[i]),
            "class_id": int(class_id[i]),
        }
        for i in np.nonzero(keep)[0]
    ]
    return rotated_nms(dets, iou_threshold)

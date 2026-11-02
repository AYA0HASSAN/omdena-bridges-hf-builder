"""Model-agnostic detector interface. `serve.py` (live/batch serving) uses only `OnnxDetector` -
onnxruntime, CPU by default, no torch, no ultralytics - because Ultralytics is AGPL-3.0 and the
licensing question for this project is still open. `UltralyticsDetector` exists only for the
ONNX/PyTorch parity tests and training-time checks, never for serving.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .postprocess import MODEL_IMGSZ, obb_corners, postprocess


@dataclass
class Detections:
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2)))  # (K, 4, 2) px, model-input space
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (K,)
    class_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # (K,)


class Detector(Protocol):
    imgsz: int

    def predict(self, chips: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[Detections]:
        """chips: N x imgsz x imgsz x 3 uint8 RGB. One (already NMS'd) Detections per chip."""
        ...


class OnnxDetector:
    """onnxruntime CPU inference + our own confidence filter / rotated NMS (postprocess.postprocess)."""

    def __init__(self, model_path: str, imgsz: int = MODEL_IMGSZ, providers: list[str] | None = None):
        import onnxruntime as ort

        self.imgsz = imgsz
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, chips: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[Detections]:
        out = []
        for chip in chips:
            batch = chip.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
            raw = self.session.run(None, {self.input_name: batch})[0]
            dets = postprocess(raw, {"confidence_threshold": conf, "iou_threshold": iou})
            out.append(_to_detections(dets))
        return out


class UltralyticsDetector:
    """Ultralytics YOLO (.pt) inference - parity tests and training-time checks only, never serving."""

    def __init__(self, weights_path: str, imgsz: int = MODEL_IMGSZ):
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.model = YOLO(weights_path, task="obb")

    def predict(self, chips: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[Detections]:
        out = []
        for chip in chips:
            result = self.model.predict(chip, imgsz=self.imgsz, conf=conf, iou=iou, verbose=False)[0]
            if result.obb is not None and len(result.obb):
                corners = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
                scores = result.obb.conf.cpu().numpy()
                class_ids = result.obb.cls.cpu().numpy().astype(int)
                out.append(Detections(corners, scores, class_ids))
            else:
                out.append(Detections())
        return out


def _to_detections(dets: list[dict[str, Any]]) -> Detections:
    if not dets:
        return Detections()
    return Detections(
        corners=np.array([obb_corners(d) for d in dets]),
        scores=np.array([d["confidence"] for d in dets]),
        class_ids=np.array([d["class_id"] for d in dets]),
    )

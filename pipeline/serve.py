"""Serving path: a folder of `OAM-x-y-z` chips -> ONNX predictions -> GeoJSON.

    python -m pipeline.serve --model bridges-v0.onnx --chips /path/chips --out bridges.geojson \
        --conf 0.10 --batch 16
"""
import argparse
import json
import math
from pathlib import Path

from .detector import OnnxDetector
from .geo import chip_corners_to_lonlat, parse_oam_name
from .postprocess import CLASS_NAME, CLASS_VALUE


def _angle_from_corners(corners) -> float:
    """Bearing (degrees, 0-360) of the first edge of a 4-corner box - a stand-in for the model's own
    angle output when corners already come from `postprocess`'s `cx/cy/w/h/angle` form."""
    import numpy as np

    c = np.asarray(corners)
    edge = c[1] - c[0]
    return math.degrees(math.atan2(edge[1], edge[0])) % 360


def serve(model_path: str, chips_dir: str, out: str, conf: float, iou: float, batch: int, imgsz: int) -> dict:
    import numpy as np
    from PIL import Image

    detector = OnnxDetector(model_path, imgsz=imgsz)
    files = sorted(p for pat in ("*.png", "*.jpg", "*.jpeg") for p in Path(chips_dir).glob(pat))
    if not files:
        raise FileNotFoundError(f"No chip images found in {chips_dir}")

    features = []
    skipped = 0
    for i in range(0, len(files), batch):
        chunk = files[i : i + batch]
        images = [Image.open(f).convert("RGB") for f in chunk]
        native_sizes = [im.size for im in images]  # (w, h)
        chips = np.stack([np.array(im.resize((imgsz, imgsz), Image.Resampling.BILINEAR)) for im in images])
        for f, (w, h), result in zip(chunk, native_sizes, detector.predict(chips, conf=conf, iou=iou)):
            xyz = parse_oam_name(f.name)
            if xyz is None:
                skipped += 1
                continue
            x, y, z = xyz
            sx, sy = w / imgsz, h / imgsz
            for corners, score, class_id in zip(result.corners, result.scores, result.class_ids):
                native = corners * [sx, sy]
                lonlat = chip_corners_to_lonlat(native, x, y, z, w, h)
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "class": int(class_id) if len(result.class_ids) else CLASS_VALUE,
                            "label": CLASS_NAME,
                            "confidence": round(float(score), 4),
                            "angle": round(_angle_from_corners(native), 2),
                            "chip_id": f.stem,
                        },
                        "geometry": {"type": "Polygon", "coordinates": [[*lonlat, lonlat[0]]]},
                    }
                )

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "model": str(model_path),
        "confidence_threshold": conf,
    }
    Path(out).write_text(json.dumps(fc))
    print(f"[serve] {len(features)} features from {len(files)} chips ({skipped} skipped, not OAM-x-y-z named) -> {out}")
    return fc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--chips", required=True)
    p.add_argument("--out", default="bridges.geojson")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=512)
    args = p.parse_args()
    serve(args.model, args.chips, args.out, args.conf, args.iou, args.batch, args.imgsz)


if __name__ == "__main__":
    main()

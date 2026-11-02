"""Serving-only entrypoint: rasterio/pyproj/shapely/PIL/numpy + fair.utils/fair.stac only - never
`zenml` - so the inference container can import this module without the ZenML-decorated training
pipeline (ticket [BRIDGES-XX], agent feedback #858). `pipeline.py` re-exports `predict` from here for
its own `run_inference` step; the STAC item's `mlm:entrypoint` and the Dockerfile's `inference` stage
both point `MODEL_MODULE` at this module, not at `pipeline`.
"""

import math
from typing import Any

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory

from .obb_utils import CLASS_NAME, CLASS_VALUE, MODEL_IMGSZ, obb_corners, postprocess


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Module-level serving entrypoint for `fair.serve.base`. ONNX Runtime + rasterio/pyproj/shapely/PIL
    only - no torch, no ultralytics, no zenml, so this runs in the distroless inference image."""
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import Polygon, mapping
    from shapely.geometry.polygon import orient

    if "confidence_threshold" not in params:
        raise ValueError("predict requires a 'confidence_threshold' parameter")

    chips_dir = resolve_directory(input_images[IMAGE_INPUT])
    chip_paths = sorted(p for pat in ("*.tif", "*.tiff") for p in chips_dir.rglob(pat))
    if not chip_paths:
        raise FileNotFoundError(f"No georeferenced (.tif/.tiff) chips found in {chips_dir}")

    input_name = session.get_inputs()[0].name
    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            transform, crs, width, height = src.transform, src.crs, src.width, src.height
            arr = src.read([1, 2, 3])
        rgb = np.ascontiguousarray(np.transpose(arr, (1, 2, 0))).astype(np.uint8)
        resized = Image.fromarray(rgb, mode="RGB").resize((MODEL_IMGSZ, MODEL_IMGSZ), Image.Resampling.BILINEAR)
        batch = np.asarray(resized).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        raw = session.run(None, {input_name: batch})[0]
        detections = postprocess(raw, params)
        if not detections:
            continue

        sx, sy = width / MODEL_IMGSZ, height / MODEL_IMGSZ
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs is not None else None
        for det in detections:
            native = [(px * sx, py * sy) for px, py in obb_corners(det)]
            geo = [transform * (px, py) for px, py in native]
            lonlat = [to_wgs84.transform(x, y) for x, y in geo] if to_wgs84 is not None else geo
            polygon = orient(Polygon([*lonlat, lonlat[0]]))
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class": CLASS_VALUE,
                        "label": CLASS_NAME,
                        "confidence": round(det["confidence"], 4),
                        "angle": round(math.degrees(det["angle"]) % 360, 2),
                        "source": chip_path.name,
                    },
                    "geometry": mapping(polygon),
                }
            )
    return {"type": "FeatureCollection", "features": features}

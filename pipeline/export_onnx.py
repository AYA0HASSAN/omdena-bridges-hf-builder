"""Export Ultralytics YOLO (task=obb) weights to ONNX + a small metadata sidecar.

    python -m pipeline.export_onnx --weights best.pt --imgsz 512 --opset 17 --out bridges-v0.onnx
"""
import argparse
import hashlib
import json
from pathlib import Path


def export(weights: str, imgsz: int, opset: int, out: str, simplify: bool = True) -> str:
    import onnx
    from ultralytics import YOLO

    model = YOLO(weights, task="obb")
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify)
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(onnx_path).replace(out_path)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {"weights": str(weights), "imgsz": imgsz, "opset": opset, "simplify": simplify, "chip_size": 256, "sha256": sha256},
            indent=2,
        )
    )
    print(f"[export_onnx] {out_path} ({sha256[:12]}...), metadata -> {meta_path}")
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", default="bridges-v0.onnx")
    p.add_argument("--no-simplify", action="store_true")
    args = p.parse_args()
    export(args.weights, args.imgsz, args.opset, args.out, simplify=not args.no_simplify)


if __name__ == "__main__":
    main()

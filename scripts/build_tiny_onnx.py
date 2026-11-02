"""Build a tiny nc=1, random-weight YOLO26n-OBB ONNX export for CI (no network, no HF_TOKEN) - same
construction as pipeline/tests/test_onnx_parity.py's `_nc1_model`, used here to exercise the real
inference image's /predict endpoint end-to-end without downloading the real trained weights.

    python scripts/build_tiny_onnx.py --out ci-artifacts/tiny_nc1.onnx --imgsz 64
"""

import argparse
import shutil
from pathlib import Path


def build(out: str, imgsz: int) -> None:
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "_tiny_nc1_src"
    tmp_dir.mkdir(exist_ok=True)

    cfg = yaml_model_load("yolo26n-obb.yaml")
    cfg["nc"] = 1
    custom = tmp_dir / "yolo26n-obb.yaml"
    custom.write_text(yaml.safe_dump(cfg))

    model = YOLO(str(custom), task="obb")
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    shutil.move(onnx_path, out_path)
    print(f"[build_tiny_onnx] wrote {out_path} (imgsz={imgsz}, nc=1)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="ci-artifacts/tiny_nc1.onnx")
    p.add_argument("--imgsz", type=int, default=64)
    args = p.parse_args()
    build(args.out, args.imgsz)


if __name__ == "__main__":
    main()

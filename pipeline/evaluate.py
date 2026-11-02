"""Evaluation harness on a YOLO-OBB split (`images/<split>/*.png` + `labels/<split>/*.txt`, as
`bridgebuild to-yolo` exports): runs the ONNX detector over every image, greedy-matches predictions
to ground truth at IoU 0.3 and 0.5, sweeps confidence, and writes `metrics.json` + a sweep CSV.

    python -m pipeline.evaluate --model bridges-v0.onnx --images-dir <export>/images/test \
        --labels-dir <export>/labels/test --out metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .detector import OnnxDetector
from .metrics import at_conf, best_conf, match, sweep


def run(model_path: str, images_dir: str, labels_dir: str, imgsz: int, batch: int, out: str) -> dict:
    from PIL import Image

    detector = OnnxDetector(model_path, imgsz=imgsz)
    files = sorted(Path(images_dir).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png images found in {images_dir}")

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(0, len(files), batch):
        chunk = files[i : i + batch]
        chips = np.stack(
            [np.array(Image.open(f).convert("RGB").resize((imgsz, imgsz), Image.Resampling.BILINEAR)) for f in chunk]
        )
        for f, d in zip(chunk, detector.predict(chips, conf=0.001)):
            preds[f.stem] = (d.corners, d.scores)

    det, gt = match(preds, labels_dir)
    sw = sweep(det, gt)
    conf = best_conf(sw, iou_thr=0.5)
    rows_at_conf = at_conf(sw, conf)

    metrics = {
        "model": str(model_path),
        "threshold": conf,
        "n_images": len(files),
        "by_iou": rows_at_conf.to_dict("records"),
    }
    out_path = Path(out)
    out_path.write_text(json.dumps(metrics, indent=2))
    sweep_csv = out_path.with_suffix(".sweep.csv")
    sw.to_csv(sweep_csv, index=False)
    print(f"[evaluate] {len(files)} images, threshold={conf:.2f} -> {out_path}, sweep -> {sweep_csv}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="metrics.json")
    args = p.parse_args()
    run(args.model, args.images_dir, args.labels_dir, args.imgsz, args.batch, args.out)


if __name__ == "__main__":
    main()

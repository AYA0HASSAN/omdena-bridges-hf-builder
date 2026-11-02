"""Rotated-box evaluation: greedy IoU matching, a confidence sweep, precision/recall/F1.

Ported (not imported) from `experiments/bridges-hf-builder/bridgebuild/experiment.py`'s proven
`match`/`sweep`/`best_conf`/`at_conf` - ported rather than imported so `pipeline/` doesn't pull in
bridgebuild's heavier optional deps (`datasets`, `huggingface_hub`) that it has no other use for.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _polys(arr):
    from shapely.geometry import Polygon

    return [Polygon(p) for p in arr]


def _iou(a, b) -> float:
    if not a.intersects(b):
        return 0.0
    i = a.intersection(b).area
    return i / (a.area + b.area - i + 1e-9)


def read_gt(lbl_path, size: int = 256) -> np.ndarray:
    """A YOLO-OBB label file -> (N, 4, 2) pixel corners."""
    if not Path(lbl_path).exists():
        return np.zeros((0, 4, 2))
    rows = [list(map(float, line.split()[1:])) for line in Path(lbl_path).read_text().splitlines() if line.strip()]
    return np.array(rows).reshape(-1, 4, 2) * size if rows else np.zeros((0, 4, 2))


def match(preds: dict[str, tuple[np.ndarray, np.ndarray]], lbl_dir, ious: tuple[float, ...] = (0.3, 0.5)):
    """preds: {stem: (corners_px (K,4,2), scores (K,))}. Greedy matching per image (highest
    confidence first). Returns (detections table, ground-truth table)."""
    det, gt = [], []
    for stem, (corners, scores) in preds.items():
        g = read_gt(Path(lbl_dir) / f"{stem}.txt")
        gt_polys, pred_polys = _polys(g), _polys(corners)
        iou_matrix = (
            np.array([[_iou(a, b) for b in gt_polys] for a in pred_polys])
            if pred_polys and gt_polys
            else np.zeros((len(pred_polys), len(gt_polys)))
        )
        order = np.argsort(-scores) if len(scores) else np.array([], dtype=int)
        for thr in ious:
            used: set[int] = set()
            hit_conf: dict[int, float] = {}
            for i in order:
                candidates = (
                    [j for j in np.argsort(-iou_matrix[i]) if j not in used and iou_matrix[i, j] >= thr]
                    if iou_matrix.shape[1]
                    else []
                )
                tp = bool(candidates)
                if tp:
                    used.add(candidates[0])
                    hit_conf[candidates[0]] = scores[i]
                det.append({"stem": stem, "iou_thr": thr, "conf": float(scores[i]), "tp": tp})
            for j in range(len(gt_polys)):
                gt.append({"stem": stem, "box": j, "iou_thr": thr, "hit_conf": hit_conf.get(j, -1.0)})
    return pd.DataFrame(det), pd.DataFrame(gt)


def sweep(det: pd.DataFrame, gt: pd.DataFrame, confs: np.ndarray = np.round(np.arange(0.05, 0.95, 0.05), 2)) -> pd.DataFrame:
    rows = []
    for thr in sorted(gt["iou_thr"].unique()) if len(gt) else []:
        d, g = det[det.iou_thr == thr], gt[gt.iou_thr == thr]
        for c in confs:
            tp = int(((d.conf >= c) & d.tp).sum())
            fp = int(((d.conf >= c) & ~d.tp).sum())
            rec = int((g.hit_conf >= c).sum()) / max(len(g), 1)
            prec = tp / max(tp + fp, 1)
            rows.append(
                {
                    "iou_thr": thr,
                    "conf": c,
                    "precision": prec,
                    "recall": rec,
                    "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                    "tp": tp,
                    "fp": fp,
                    "gt": len(g),
                }
            )
    return pd.DataFrame(rows)


def best_conf(sw: pd.DataFrame, iou_thr: float = 0.5) -> float:
    s = sw[sw.iou_thr == iou_thr]
    return float(s.loc[s.f1.idxmax(), "conf"]) if len(s) else 0.25


def at_conf(sw: pd.DataFrame, conf: float) -> pd.DataFrame:
    return sw[np.isclose(sw.conf, conf)].reset_index(drop=True)

"""Training experiments on the dataset that is already on the Hub - no rebuild, no source re-download.

1. load_tables      index tables only (a few MB): chips + bridges
2. box_table        one row per label box with its size, bridge length, footbridge flag, visible share
3. ignore_rules     marks boxes the model cannot be expected to find (culverts, slivers, footbridges)
4. spatial_split    re-split: whole test *cities* (several countries), z14 blocks for val, bridge grouping,
                    1-chip buffer, pHash, then the 5 leakage tests
5. build_pool       every chip image once on local disk; reuses an earlier YOLO export and the HF cache,
                    downloads only shards that still hold missing chips
6. make_experiment  symlinked YOLO folders + yaml files: train (OAM repeated), train_oam, val (OAM),
                    val_france, test (clean), test_raw
7. evaluate         rotated-box precision / recall / F1 at IoU 0.3 and 0.5 over a confidence sweep,
                    recall per label category (why a bridge was missed)
"""
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .finalize import assign_split, leakage_tests, load_index

FOOT_HIGHWAY = {"footway", "path", "pedestrian", "steps", "cycleway", "bridleway", "corridor"}
SETS = ("train", "train_oam", "val", "val_france", "val_all", "test", "test_raw")


# ------------------------------------------------------------------ 1. tables
def city_of(region):
    return re.sub(r"_\d+$", "", str(region))


def stem_of(chip_id):
    src, name = chip_id.split("/", 1)
    return name if src == "oam_osm" else f"{src}_{name}"      # same names as to_yolo / the old export


def load_tables(hub, work="bb_work/exp"):
    ch, br = load_index(hub, Path(work) / "_idx")
    ch["city"] = ch["region"].map(city_of)
    ch["stem"] = ch["chip_id"].map(stem_of)
    if "bounds_3857" in ch and isinstance(ch["bounds_3857"].iloc[0], str):
        ch["bounds_3857"] = ch["bounds_3857"].map(json.loads)
    br = br.drop_duplicates("bridge_id").reset_index(drop=True) if len(br) else br
    return ch, br


# ------------------------------------------------------------------ 2. boxes
def _tags(s):
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:
        return {}


def box_table(ch, br):
    """One row per label box: chip, box index, pixel size, edge contact, bridge length, footbridge, visible share."""
    import geopandas as gpd
    from shapely.geometry import box as sbox
    deck, info = {}, {}
    if len(br):
        g = gpd.GeoSeries.from_wkt(br["deck_wkt"].fillna("POLYGON EMPTY"), crs=4326).to_crs(3857)
        for bid, geom, L, tj, cls in zip(br["bridge_id"], g, br["length_m"], br["tags_json"], br["cls"]):
            t = _tags(tj)
            foot = str(t.get("highway", "")).lower() in FOOT_HIGHWAY or \
                str(t.get("nature", "")).lower().startswith("passerelle")
            deck[bid] = geom
            info[bid] = (float(L) if L == L else np.nan, foot, cls)
    rows = []
    for r in ch[ch["n_boxes"] > 0].itertuples():
        boxes = json.loads(r.boxes_px)
        ids = list(r.bridge_ids) if len(r.bridge_ids) == len(boxes) else [None] * len(boxes)
        cb = sbox(*r.bounds_3857)
        for k, (c, bid) in enumerate(zip(boxes, ids)):
            p = np.array(c, float).reshape(4, 2)
            e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(2)]
            L, foot, cls = info.get(bid, (np.nan, False, None))
            d = deck.get(bid)
            vis = (d.intersection(cb).area / d.area) if (d is not None and not d.is_empty and d.area > 0) else np.nan
            rows.append({"chip_id": r.chip_id, "stem": r.stem, "source": r.source, "city": r.city, "box": k,
                         "bridge_id": bid, "long_px": max(e), "short_px": min(e),
                         "touches_edge": bool((p <= 1).any() or (p >= 255).any()),
                         "length_m": L, "footbridge": foot, "cls": cls, "visible_share": vis})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. ignore rules
def ignore_rules(boxes, min_len_m=12.0, min_long_px=16, min_short_px=4, sliver_visible=0.25, sliver_px=48,
                 drop_footbridges=True):
    """Adds `ignore` (reason or ""). A chip with any ignored box is left out of training and of the clean test
    set (not used as a negative either: the bridge is still in the picture)."""
    b = boxes.copy()

    def why(r):
        if r.length_m == r.length_m and r.length_m < min_len_m:
            return f"short bridge <{min_len_m:g} m (culvert?)"
        if r.long_px < min_long_px or r.short_px < min_short_px:
            return f"tiny box <{min_long_px}x{min_short_px} px"
        if drop_footbridges and r.footbridge:
            return "footbridge"
        if r.touches_edge and r.visible_share == r.visible_share and r.visible_share < sliver_visible and r.long_px < sliver_px:
            return "edge sliver"
        return ""
    b["ignore"] = [why(r) for r in b.itertuples()] if len(b) else []
    return b


def chip_ignore(boxes):
    """chip_id -> main ignore reason ("" if every box is fine)."""
    if not len(boxes):
        return {}
    bad = boxes[boxes["ignore"] != ""]
    return bad.groupby("chip_id")["ignore"].agg(lambda v: Counter(v).most_common(1)[0][0]).to_dict()


# ------------------------------------------------------------------ 4. spatial split
def city_table(ch, source="oam_osm"):
    c = ch[ch["source"] == source]
    t = c.groupby(["city", "country"]).agg(chips=("chip_id", "nunique"), positive=("n_boxes", lambda v: int((v > 0).sum())),
                                           boxes=("n_boxes", "sum"), images=("image_id", "nunique"),
                                           regions=("region", "nunique")).reset_index()
    return t.sort_values("positive", ascending=False).reset_index(drop=True)


def suggest_test_cities(ch, share=0.25, source="oam_osm"):
    """Whole cities held out, at most one per country, the largest city always kept for training,
    until ~`share` of the source's positive chips is in test."""
    t = city_table(ch, source)
    if len(t) < 2:
        return list(t["city"])
    total, got, picked, countries = t["positive"].sum(), 0, [], set()
    for r in t.iloc[1:].itertuples():                  # skip the biggest city
        if r.country in countries or r.positive == 0:
            continue
        picked.append(r.city); countries.add(r.country); got += r.positive
        if got >= share * total:
            break
    return picked


def spatial_split(ch, br, test_cities, val_fraction=None, buffer_chips=1, phash_max=4, test_source="oam_osm"):
    val_fraction = val_fraction or {"oam_osm": 0.3, "default": 0.15}
    regs = ch.loc[(ch["source"] == test_source) & ch["city"].isin(test_cities), ["source", "region"]].drop_duplicates()
    cfg = {"split": {"test_regions": regs.to_dict("records"), "val_fraction": val_fraction,
                     "buffer_chips": buffer_chips, "phash_max": phash_max}}
    s = assign_split(ch, br, cfg)
    # join on the row index (assign_split keeps ch's index): a merge on chip_id/image_id would duplicate rows when the
    # same tile from the same image was cut twice (overlapping regions) and break "every chip in one split"
    s = s.join(ch[["stem", "city", "obb_yolo"]])
    leak = leakage_tests(s, cfg)
    return s, leak, cfg


# ------------------------------------------------------------------ 5. image pool (download once)
def build_pool(hub, s, pool, reuse_dirs=(), batch_rows=64):
    """All kept chips as pool/images/<stem>.png + pool/labels/<stem>.txt.
    Order of sources: already in the pool -> an earlier YOLO export (reuse_dirs) -> HF shard (HF cache first)."""
    import pyarrow.parquet as pq
    pool = Path(pool)
    (pool / "images").mkdir(parents=True, exist_ok=True)
    (pool / "labels").mkdir(parents=True, exist_ok=True)
    keep = s[s["split"].isin(["train", "val", "test"])].drop_duplicates("stem")
    for r in keep.itertuples():                                  # labels come from the index: always cheap
        (pool / "labels" / f"{r.stem}.txt").write_text((r.obb_yolo or "") + ("\n" if r.obb_yolo else ""))
    have = {p.stem for p in (pool / "images").glob("*.png")}
    old = {}
    for d in reuse_dirs:
        if d and Path(d).exists():
            old.update({p.stem: p for p in Path(d).rglob("*.png") if "/images/" in str(p)})
    reused = 0
    for st in keep["stem"]:
        if st not in have and st in old:
            try:
                os.link(old[st], pool / "images" / f"{st}.png")
            except OSError:
                shutil.copyfile(old[st], pool / "images" / f"{st}.png")
            have.add(st); reused += 1
    missing = keep[~keep["stem"].isin(have)]
    print(f"[pool] {len(keep)} chips: {len(keep) - len(missing) - reused} already in pool, {reused} reused from "
          f"earlier export, {len(missing)} to read from {missing['shard'].nunique()} shard(s)")
    want = dict(zip(missing["chip_id"], missing["stem"]))
    for k, sh in enumerate(sorted(missing["shard"].unique())):
        pf = pq.ParquetFile(hub.get(sh))                         # HF cache: no download if already cached
        for b in pf.iter_batches(batch_size=batch_rows, columns=["chip_id", "image"]):
            for r in b.to_pylist():
                st = want.get(r["chip_id"])
                if st and st not in have:
                    (pool / "images" / f"{st}.png").write_bytes(r["image"]["bytes"]); have.add(st)
        del pf
        print(f"[pool] shard {k + 1}/{missing['shard'].nunique()} {sh}")
    lost = set(keep["stem"]) - have
    if lost:
        print(f"[pool] WARNING {len(lost)} chips not found in any shard, e.g. {list(lost)[:3]}")
    return pool


# ------------------------------------------------------------------ 6. experiment folders
def _link(src, dst):
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def make_experiment(pool, s, ignored, out, oam_repeat=3, val_sources=("oam_osm",), neg_keep=1.0, seed=0):
    """Symlink folders (no copies) + yaml files. `ignored`: chip_id -> reason."""
    pool, out = Path(pool), Path(out)
    shutil.rmtree(out, ignore_errors=True)
    rng = np.random.default_rng(seed)
    k = s[s["split"].isin(["train", "val", "test"])].drop_duplicates("stem").copy()
    k["ignored"] = k["chip_id"].map(lambda c: ignored.get(c, ""))
    k = k[k["stem"].map(lambda st: (pool / "images" / f"{st}.png").exists())]
    neg = k["n_boxes"] == 0
    k = k[~neg | (rng.random(len(k)) < neg_keep)]
    clean = k["ignored"] == ""
    sets = {
        "train": k[(k.split == "train") & clean],
        "train_oam": k[(k.split == "train") & clean & (k.source == "oam_osm")],
        "val": k[(k.split == "val") & clean & k.source.isin(val_sources)],
        "val_france": k[(k.split == "val") & clean & (k.source == "france")],
        "val_all": k[(k.split == "val") & clean],
        "test": k[(k.split == "test") & clean],
        "test_raw": k[k.split == "test"],
    }
    counts = []
    for name, d in sets.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for r in d.itertuples():
            reps = oam_repeat if (name == "train" and r.source == "oam_osm") else 1
            for j in range(reps):
                nm = r.stem if j == 0 else f"{r.stem}__r{j}"
                _link(pool / "images" / f"{r.stem}.png", out / "images" / name / f"{nm}.png")
                _link(pool / "labels" / f"{r.stem}.txt", out / "labels" / name / f"{nm}.txt")
        counts.append({"set": name, "chips": len(d), "positive": int((d.n_boxes > 0).sum()),
                       "negative": int((d.n_boxes == 0).sum()), "boxes": int(d.n_boxes.sum()),
                       "oam_chips": int((d.source == "oam_osm").sum()), "files": len(os.listdir(out / "images" / name))})

    def yaml(fn, train, val):
        (out / fn).write_text(f"path: {out.resolve()}\ntrain: images/{train}\nval: images/{val}\nnames:\n  0: bridge\n")
    yaml("data_stage1.yaml", "train", "val")
    yaml("data_stage2.yaml", "train_oam", "val")
    for name in sets:
        yaml(f"eval_{name}.yaml", "train", name)
    return pd.DataFrame(counts)


# ------------------------------------------------------------------ 7. evaluation
def _polys(arr):
    from shapely.geometry import Polygon
    return [Polygon(p) for p in arr]


def _iou(a, b):
    if not a.intersects(b):
        return 0.0
    i = a.intersection(b).area
    return i / (a.area + b.area - i + 1e-9)


def read_gt(lbl_path, size=256):
    if not Path(lbl_path).exists():
        return np.zeros((0, 4, 2))
    rows = [list(map(float, l.split()[1:])) for l in Path(lbl_path).read_text().splitlines() if l.strip()]
    return np.array(rows).reshape(-1, 4, 2) * size if rows else np.zeros((0, 4, 2))


def predict_set(model, img_dir, imgsz=512, batch=32, conf=0.001):
    """Predictions for every image of a set: {stem: (polys px in 256 space, conf)}."""
    from PIL import Image
    files = sorted(Path(img_dir).glob("*.png"))
    out = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        for f, r in zip(chunk, model.predict([str(f) for f in chunk], imgsz=imgsz, conf=conf, verbose=False)):
            if r.obb is None or not len(r.obb):
                out[f.stem] = (np.zeros((0, 4, 2)), np.zeros(0))
            else:
                h, w = r.orig_shape
                pts = r.obb.xyxyxyxy.cpu().numpy() * (256 / max(h, w))
                out[f.stem] = (pts, r.obb.conf.cpu().numpy())
    return out


def match(preds, lbl_dir, ious=(0.3, 0.5)):
    """Greedy matching per image (highest confidence first). Returns detections table and GT table."""
    det, gt = [], []
    for stem, (pp, cc) in preds.items():
        g = read_gt(Path(lbl_dir) / f"{stem}.txt")
        gp, ppoly = _polys(g), _polys(pp)
        M = np.array([[_iou(a, b) for b in gp] for a in ppoly]) if len(ppoly) and len(gp) else np.zeros((len(ppoly), len(gp)))
        order = np.argsort(-cc)
        for thr in ious:
            used = set()
            hit_gt = {}
            for i in order:
                best = [jj for jj in np.argsort(-M[i]) if jj not in used and M[i, jj] >= thr] if M.shape[1] else []
                tp = bool(best)
                if tp:
                    used.add(best[0]); hit_gt[best[0]] = cc[i]
                det.append({"stem": stem.split("__r")[0], "iou_thr": thr, "conf": float(cc[i]), "tp": tp})
            for j in range(len(gp)):
                gt.append({"stem": stem.split("__r")[0], "box": j, "iou_thr": thr, "hit_conf": hit_gt.get(j, -1.0)})
    return pd.DataFrame(det), pd.DataFrame(gt)


def sweep(det, gt, confs=np.round(np.arange(0.05, 0.95, 0.05), 2)):
    rows = []
    for thr in sorted(gt["iou_thr"].unique()) if len(gt) else []:
        d, g = det[det.iou_thr == thr], gt[gt.iou_thr == thr]
        for c in confs:
            tp = int(((d.conf >= c) & d.tp).sum()); fp = int(((d.conf >= c) & ~d.tp).sum())
            rec = int((g.hit_conf >= c).sum()) / max(len(g), 1)
            prec = tp / max(tp + fp, 1)
            rows.append({"iou_thr": thr, "conf": c, "precision": prec, "recall": rec,
                         "f1": 2 * prec * rec / max(prec + rec, 1e-9), "tp": tp, "fp": fp, "gt": len(g)})
    return pd.DataFrame(rows)


def best_conf(sw, iou_thr=0.5):
    s = sw[sw.iou_thr == iou_thr]
    return float(s.loc[s.f1.idxmax(), "conf"]) if len(s) else 0.25


def at_conf(sw, conf):
    return sw[np.isclose(sw.conf, conf)].reset_index(drop=True)


def recall_by_category(gt, boxes, conf, iou_thr=0.3):
    """Why bridges are missed: recall per ignore category ("" = clean label) and per box size."""
    g = gt[gt.iou_thr == iou_thr].merge(boxes[["stem", "box", "ignore", "long_px", "source"]], on=["stem", "box"], how="left")
    g["found"] = g.hit_conf >= conf
    g["category"] = g["ignore"].fillna("").replace("", "clean label")
    g["size"] = pd.cut(g["long_px"], [0, 16, 32, 64, 128, 400], labels=["<16", "16-32", "32-64", "64-128", ">128"])
    return (g.groupby("category").agg(boxes=("found", "size"), recall=("found", "mean")).reset_index(),
            g.groupby("size", observed=True).agg(boxes=("found", "size"), recall=("found", "mean")).reset_index())

"""Offline end-to-end test: fake source (synthetic bridges + rendered imagery) -> build (stopped, then
resumed) -> finalize (split + leakage tests) -> stats -> to-yolo. Uses a local folder as the "Hub".

  python tests/test_offline_pipeline.py
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box
from shapely import affinity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgebuild import sources, build as B, finalize as F, stats as S, to_yolo as Y   # noqa: E402
from bridgebuild.sources import CellData, Imagery                                         # noqa: E402
from bridgebuild.geo import bbox_3857                                                     # noqa: E402


def fake_cell(region, cell_id, bbox, opts):
    rng = np.random.default_rng(abs(hash(cell_id)) % 2**32)
    x0, y0, x1, y1 = bbox_3857(bbox)
    n = int(rng.integers(0, 6))                     # some cells have no bridges
    decks, srcs = [], []
    for k in range(n):
        cx, cy = rng.uniform(x0 + 200, x1 - 200), rng.uniform(y0 + 200, y1 - 200)
        L, W, a = rng.uniform(25, 160), rng.uniform(6, 18), rng.uniform(0, 180)
        line = affinity.rotate(LineString([(cx - L / 2, cy), (cx + L / 2, cy)]), a, origin=(cx, cy))
        decks.append({"src_id": f"fake/{cell_id}/{k}", "cls": "F", "width_m": W, "geometry": line.buffer(W / 2, cap_style=2)})
        srcs.append({"src_id": f"fake/{cell_id}/{k}", "bridge": "yes", "highway": "primary", "geometry": line})
    br = gpd.GeoDataFrame(decks, crs=3857) if decks else sources.empty_bridges()
    src = gpd.GeoDataFrame(srcs, crs=3857).to_crs(4326) if srcs else gpd.GeoDataFrame({"src_id": []}, geometry=[], crs=4326)
    water = gpd.GeoDataFrame(geometry=[LineString([(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2 + 300)]).buffer(40)], crs=3857)
    roads = gpd.GeoDataFrame(geometry=[LineString([((x0 + x1) / 2, y0), ((x0 + x1) / 2 + 200, y1)])], crs=3857)
    cell = CellData("oam_osm", region, cell_id, bbox, br, src, water, roads, licence_lbl="ODbL 1.0", label_source="fake OSM")

    def reader(bb, sz, decks=br):
        w, h = sz
        img = (rng.normal(110, 25, (h, w, 3))).clip(1, 255).astype(np.uint8)
        if len(decks):
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            m = rasterize([(g, 1) for g in decks.geometry], out_shape=(h, w), transform=from_bounds(*bb, w, h))
            img[m == 1] = (200, 200, 190)
        return img
    if n:
        cell.imagery.append(Imagery(f"img-{region['name']}", reader, box(x0, y0, x1, y1), 0.3, "CC-BY 4.0",
                                    "fake provider", "2024-01-01", "https://example/fake.tif", "fake"))
    if cell_id.endswith("_001_000"):
        raise IOError("simulated service failure")   # one failing cell must not stop the run
    return cell


def main():
    work = ROOT / "tests" / "_out"
    shutil.rmtree(work, ignore_errors=True)
    sources.SOURCES["oam_osm"] = fake_cell
    B.SOURCES["oam_osm"] = fake_cell
    cfg = {
        "hub": {"local_dir": str(work / "hub")}, "work_dir": str(work / "work"),
        "build": {"cell_deg": 0.01, "shard_size": 40, "label_margin_m": 300, "state_every_cells": 5},
        "chips": {"min_valid": 0.9, "neg_ratio": 0.5, "hard_negative_share": 0.5, "max_pos_per_cell": 400},
        "sources": {"oam_osm": {"enabled": True, "regions": [
            {"name": "town_a", "bbox": [39.20, -6.83, 39.23, -6.80], "country": "TZ", "global_south": True},
            {"name": "town_b", "bbox": [39.23, -6.83, 39.26, -6.80], "country": "TZ", "global_south": True},
            {"name": "town_t", "bbox": [39.30, -6.83, 39.32, -6.81], "country": "TZ", "global_south": True}]}},
        "split": {"test_regions": [{"source": "oam_osm", "region": "town_t"}], "val_fraction": 0.3, "buffer_chips": 1,
                  "phash_max": 4},
    }
    # 1. build part of it (simulates a session that stops), then resume
    st1 = B.build(cfg, max_cells=7)
    done1 = list(st1["done_cells"])
    st2 = B.build(cfg)
    assert len(st2["done_cells"]) == len(set(st2["done_cells"])), "a cell was processed twice"
    assert set(done1) <= set(st2["done_cells"])
    assert len(st2["failed_cells"]) == 3, st2["failed_cells"]        # the *_001_000 cell of each region
    n_cells = sum(1 for r in cfg["sources"]["oam_osm"]["regions"] for _ in sources.cells_for_region(r, 0.01))
    assert len(st2["done_cells"]) + len(st2["failed_cells"]) == n_cells
    st3 = B.build(cfg)                                                # nothing left: no new shards
    assert st3["next_shard"] == st2["next_shard"]
    hub = work / "hub"
    shards = sorted((hub / "data").glob("*.parquet"))
    assert shards and not list((work / "work").glob("shard-*.parquet")), "local shards must be deleted after upload"
    # chips unique, positives have labels
    ch, br = F.load_index(B.open_hub(cfg), work / "x")
    assert ch["chip_id"].is_unique, "the same tile was cut twice"
    assert (ch.loc[ch.kind == "positive", "n_boxes"] > 0).all()
    assert (ch.loc[ch.kind != "positive", "n_boxes"] == 0).all()
    # 2. HF datasets can read the shards and decode images
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=[str(p) for p in shards], split="train")
    assert ds[0]["image"].size == (256, 256) and len(ds) == len(ch)
    # 3. finalize
    s, leak = F.finalize(cfg)
    assert leak["passed"].all(), leak
    assert (s.loc[s.region == "town_t", "split"].isin(["test", "buffer", "drop"])).all()
    bad = s.copy()                                                    # the tests must catch a leaky split
    i = bad.index[bad.split == "train"][0]
    bad.loc[i, "split"] = "val"; bad = pd.concat([bad, bad.loc[[i]].assign(split="train")])
    assert not F.leakage_tests(bad, cfg)["passed"].all(), "leakage tests did not catch a duplicated chip"
    for f in ("README.md", "MANIFEST.csv", "dataset_record.md", "splits/split.parquet"):
        assert (hub / f).exists(), f
    man = pd.read_csv(hub / "MANIFEST.csv")
    import hashlib
    for r in man.itertuples():
        if r.file.startswith("data/"):
            assert hashlib.sha256((hub / r.file).read_bytes()).hexdigest() == r.sha256
    # 4. stats and YOLO export
    S.overlays(cfg, n=12)
    S.stats(cfg, push=True)
    assert (hub / 'reports' / 'overlay_samples.png').exists()
    counts = Y.to_yolo(cfg, work / "yolo")
    n_lbl = sum(1 for _ in (work / "yolo" / "labels").rglob("*.txt"))
    n_img = sum(1 for _ in (work / "yolo" / "images").rglob("*.png"))
    assert n_lbl == n_img == (s["split"].isin(["train", "val", "test"])).sum()
    one = next(p for p in (work / "yolo" / "labels" / "train").glob("*.txt") if p.read_text().strip())
    vals = one.read_text().split("\n")[0].split()
    assert vals[0] == "0" and len(vals) == 9 and all(0 <= float(v) <= 1 for v in vals[1:])
    print("\nsplit:", s.groupby("split").size().to_dict())
    print("yolo:", counts)
    print("OFFLINE PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()

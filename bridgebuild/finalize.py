"""Finalize step (run after one or more build sessions): deduplicate, split, leakage tests, manifest,
dataset card and dataset record. Reads only the small index files, never the images.

Split rules (same as the analysis pipeline, doc 05):
- test   = whole regions listed in split.test_regions (OAM only, fixed before training)
- train/val by z14 block (~2.4 km) with a stable hash; all chips of a bridge follow the bridge's home block
- buffer = a chip within `buffer_chips` of a chip in another split is dropped (test > train > val decides who drops)
- near-duplicate chips (pHash) across splits: the non-train copy is dropped
"""
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mercantile
import numpy as np
import pandas as pd

from .build import STATE_PATH, config_hash, sha256_file
from .hub import open_hub


def _frac(key, salt="bridges"):
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def load_index(hub, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    chips, bridges = [], []
    for f in hub.list("index/"):
        p = hub.get(f)
        (chips if "/chips-" in f else bridges).append(pd.read_parquet(p))
    ch = pd.concat(chips, ignore_index=True) if chips else pd.DataFrame()
    br = pd.concat(bridges, ignore_index=True) if bridges else pd.DataFrame()
    if len(ch):
        ch["bridge_ids"] = ch["bridge_ids"].apply(lambda s: json.loads(s) if isinstance(s, str) else list(s))
    return ch, br


def assign_split(ch, br, cfg):
    sc = cfg.get("split", {})
    val_frac, buf, ph_max = sc.get("val_fraction", 0.2), sc.get("buffer_chips", 1), sc.get("phash_max", 4)
    tests = {(t["source"], t["region"]) for t in sc.get("test_regions", [])}
    s = ch[["chip_id", "source", "region", "image_id", "x", "y", "z", "block_z14", "kind", "n_boxes", "bridge_ids",
            "phash", "gsd_native_m", "shard"]].copy()
    # 1. duplicates: the same tile cut twice (from two images / re-runs) -> keep the finest image, most boxes
    s = s.sort_values(["gsd_native_m", "n_boxes"], ascending=[True, False])
    dup = s.duplicated(subset=["chip_id"], keep="first")
    s["split"], s["reason"] = "", ""
    s.loc[dup, ["split", "reason"]] = ["drop", "duplicate tile"]
    live = ~dup
    # 2. test regions, then block hash
    is_test = s.apply(lambda r: (r["source"], r["region"]) in tests, axis=1)
    s.loc[live & is_test, ["split", "reason"]] = ["test", "test region"]
    rest = live & ~is_test
    s.loc[rest, "split"] = ["val" if _frac(b) < val_frac else "train" for b in s.loc[rest, "block_z14"]]
    s.loc[rest, "reason"] = "z14 block hash"
    # 3. bridge grouping: chips follow the bridge's home block (block of its centroid)
    if len(br):
        brd = br.drop_duplicates("bridge_id").set_index("bridge_id")
        home = {bid: f"{mercantile.tile(r.lon, r.lat, 14).x}-{mercantile.tile(r.lon, r.lat, 14).y}" for bid, r in brd.iterrows()}
        block_split = s[rest].groupby("block_z14")["split"].first().to_dict()
        for i, r in s[rest & (s["n_boxes"] > 0)].iterrows():
            sp = {block_split.get(home.get(b), r["split"]) for b in r["bridge_ids"]}
            if len(sp) > 1:
                s.loc[i, ["split", "reason"]] = ["buffer", "bridges of two splits"]
            elif sp and next(iter(sp)) != r["split"]:
                s.loc[i, ["split", "reason"]] = [next(iter(sp)), "follows bridge home block"]
    # 4. other sources never sit inside a test block
    test_blocks = set(s.loc[s["split"] == "test", "block_z14"])
    m = (s["split"].isin(["train", "val"])) & s["block_z14"].isin(test_blocks)
    s.loc[m, ["split", "reason"]] = ["buffer", "inside a test block"]
    # 5. buffer ring (any source): lower priority side drops
    prio = {"test": 3, "train": 2, "val": 1}
    g = s[s["split"].isin(prio)]
    xy, sp = g[["x", "y"]].to_numpy(int), g["split"].to_numpy()
    pr = np.array([prio[v] for v in sp])
    drop = []
    order = np.argsort(xy[:, 0])
    xs = xy[order, 0]
    for k in range(len(g)):                        # sorted sweep: only compare chips within buf columns
        lo, hi = np.searchsorted(xs, xy[k, 0] - buf), np.searchsorted(xs, xy[k, 0] + buf, side="right")
        cand = order[lo:hi]
        near = cand[(np.abs(xy[cand, 1] - xy[k, 1]) <= buf) & (sp[cand] != sp[k]) & (pr[cand] > pr[k])]
        if len(near):
            drop.append(g.index[k])
    s.loc[drop, ["split", "reason"]] = ["buffer", f"within {buf} chip(s) of another split"]
    # 6. near duplicates across splits (pHash), compared within each 4-hex-digit prefix bucket to stay fast
    g = s[s["split"].isin(prio)]
    for _, grp in g.groupby(g["phash"].str[:4]):
        if grp["split"].nunique() < 2:
            continue
        hs = grp[["phash", "split"]].to_numpy()
        idx = grp.index.to_numpy()
        for a in range(len(hs)):
            for c in range(a + 1, len(hs)):
                if hs[a][1] != hs[c][1] and _ham(hs[a][0], hs[c][0]) <= ph_max:
                    victim = idx[c] if hs[a][1] == "train" else idx[a]
                    s.loc[victim, ["split", "reason"]] = ["buffer", "near-duplicate across splits"]
    return s


def leakage_tests(s, cfg):
    buf = cfg.get("split", {}).get("buffer_chips", 1)
    tests = {(t["source"], t["region"]) for t in cfg.get("split", {}).get("test_regions", [])}
    k = s[s["split"].isin(["train", "val", "test"])]
    out = []

    def add(name, bad):
        bad = list(bad)
        out.append({"test": name, "passed": len(bad) == 0, "violations": len(bad), "example": str(bad[:3])})

    add("every chip in one split", k["chip_id"][k["chip_id"].duplicated()])
    ex = k[["chip_id", "split", "bridge_ids"]].explode("bridge_ids").dropna(subset=["bridge_ids"])
    nb = ex.groupby("bridge_ids")["split"].nunique()
    add("every bridge in one split", nb[nb > 1].index)
    xy, sp = k[["x", "y"]].to_numpy(int), k["split"].to_numpy()
    order = np.argsort(xy[:, 0]); xs = xy[order, 0]; bad = []
    for i in range(len(k)):
        lo, hi = np.searchsorted(xs, xy[i, 0] - buf), np.searchsorted(xs, xy[i, 0] + buf, side="right")
        cand = order[lo:hi]
        if ((np.abs(xy[cand, 1] - xy[i, 1]) <= buf) & (sp[cand] != sp[i])).any():
            bad.append(k["chip_id"].iloc[i])
    add(f"no chips of different splits within {buf} chip(s)", bad)
    add("test holds only the test regions", k[(k["split"] == "test") & ~k.apply(lambda r: (r["source"], r["region"]) in tests, axis=1)]["chip_id"])
    test_imgs = set(k.loc[k["split"] == "test", "image_id"])
    add("test images not used in train/val", k[(k["split"] != "test") & k["image_id"].isin(test_imgs)]["chip_id"])
    return pd.DataFrame(out)


def _md_table(df):
    if df is None or not len(df):
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(("" if (isinstance(v, float) and v != v) else (f"{v:.2f}" if isinstance(v, float) else str(v))).replace("|", "/") for v in r) + " |")
    return "\n".join(lines) + "\n"


def dataset_card(cfg, ch, s, br, leak, state, hub):
    counts = s[s["split"].isin(["train", "val", "test"])].pivot_table(index="source", columns="split", values="chip_id",
                                                                      aggfunc="count", fill_value=0).reset_index()
    kinds = s[s["split"].isin(["train", "val", "test"])].pivot_table(index="split", columns="kind", values="chip_id",
                                                                     aggfunc="count", fill_value=0).reset_index()
    name = cfg.get("hub", {}).get("pretty_name", "fAIr Bridges training chips")
    return f"""---
license: odbl
pretty_name: {name}
task_categories:
- object-detection
tags:
- remote-sensing
- aerial-imagery
- bridges
- openstreetmap
- openaerialmap
- oriented-bounding-boxes
- hotosm
- fair
size_categories:
- {('1K<n<10K' if len(ch) < 10000 else '10K<n<100K')}
configs:
- config_name: default
  data_files: "data/*.parquet"
---

# {name}

256 × 256 px aerial image chips (Web Mercator zoom 18, about 0.6 m per pixel, the way fAIr cuts chips) with
**rotated bridge boxes** for the fAIr Bridges base model (YOLO26s-OBB, one class: `bridge`).

## What is inside

- `data/shard-*.parquet`: one row per chip. Columns include `image` (PNG), `obb_yolo` (YOLO-OBB lines
  `0 x1 y1 x2 y2 x3 y3 x4 y4`, normalised 0–1), `bridge_ids`, the tile `x, y, z`, `bounds_3857`, the source
  image (`image_url`, `licence_img`, `provider`), and `kind` (`positive`, `hard_negative`, `random_negative`).
- `index/chips-*.parquet`: the same rows without images (fast analysis).
- `index/bridges-*.parquet`: one row per bridge, with its **geometry as delivered** (`source_wkt`: OSM way or
  outline, BD TOPO feature), the deck polygon used for the boxes (`deck_wkt`), size, direction and tags.
- `splits/split.parquet`: `chip_id` → `train` / `val` / `test` / `buffer` / `drop`, with the reason.
- `MANIFEST.csv` (sha256 of every file) and `dataset_record.md` (sources, licences, counts, tests).

## Chips per split

{_md_table(counts)}
{_md_table(kinds)}

## How the split avoids leakage

Test = whole OAM regions ({', '.join(f"{t['source']}/{t['region']}" for t in cfg.get('split', {}).get('test_regions', [])) or 'none set'}).
Train / validation = ~2.4 km blocks (zoom-14 tiles) chosen by a stable hash; every chip of a bridge stays in one
split; chips within {cfg.get('split', {}).get('buffer_chips', 1)} chip of another split are dropped; near-duplicate
chips across splits are dropped. Leakage tests: {'all passed' if leak['passed'].all() else 'FAILED - see dataset_record.md'}.

## Licences and attribution

- **Labels:** © OpenStreetMap contributors, **ODbL 1.0** (OAM + OSM chips); IGN BD TOPO, **Etalab 2.0** (France chips).
  Because the database includes OSM-derived data, the dataset as a whole is released under the **ODbL**.
- **Imagery:** OpenAerialMap images keep their own open licence (listed per chip in `licence_img`, CC BY 4.0
  family; non-commercial images are excluded) and must be credited to their providers (`provider`, `image_url`);
  IGN BD ORTHO, **Etalab 2.0**.

## Load

```python
from datasets import load_dataset
ds = load_dataset("{getattr(hub, 'repo_id', '<repo>')}", split="train", streaming=True)   # all shards; use splits/split.parquet to filter
```

To export a YOLO-OBB folder for training: `python -m bridgebuild to-yolo --config <config>` (see the repo README).
"""


def dataset_record(cfg, ch, s, br, leak, state, manifest):
    live = s["split"].isin(["train", "val", "test"])
    imgs = ch.drop_duplicates("image_id")[["source", "image_id", "provider", "licence_img", "acq_date", "image_url"]]
    per_src = ch.groupby("source").agg(chips=("chip_id", "count"), with_bridges=("n_boxes", lambda v: int((v > 0).sum())),
                                       boxes=("n_boxes", "sum"), images=("image_id", "nunique"), regions=("region", "nunique")).reset_index()
    br_s = br.drop_duplicates("bridge_id").groupby("source").agg(bridges=("bridge_id", "count"), median_length_m=("length_m", "median"),
                                                                 median_width_m=("width_m", "median"),
                                                                 over_water=("crossing", lambda v: round(float((v == "water").mean()), 2))).reset_index() if len(br) else pd.DataFrame()
    return f"""# Dataset record: fAIr Bridges training chips

Built {state.get('created', '')} → finalized {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
config hash `{config_hash(cfg)}` · {len(manifest)} files in `MANIFEST.csv` (sha256 each).

## 1. Sources and licences

| Source | Imagery | Labels |
|---|---|---|
| oam_osm | OpenAerialMap, per-image open licence (CC BY 4.0 family; non-commercial excluded) | © OpenStreetMap contributors, ODbL 1.0 |
| france | IGN BD ORTHO 20 cm, Etalab 2.0 | IGN BD TOPO, Etalab 2.0 |

**Dataset licence:** ODbL 1.0 (contains OSM-derived data). Imagery attribution is kept per chip.

## 2. Volume

{_md_table(per_src)}
{_md_table(br_s)}

## 3. Split

{_md_table(s.groupby(['split', 'reason']).size().reset_index(name='chips'))}

## 4. Leakage tests

{_md_table(leak)}

## 5. Images used ({len(imgs)})

{_md_table(imgs.head(500))}
{'(first 500 shown)' if len(imgs) > 500 else ''}

## 6. Build log

- cells finished: {len(state.get('done_cells', []))}; cells failed: {len(state.get('failed_cells', {}))}
{chr(10).join(f'  - {k}: {v}' for k, v in list(state.get('failed_cells', {}).items())[:50])}
"""


def finalize(cfg):
    hub = open_hub(cfg)
    work = Path(cfg.get("work_dir", "bb_work")) / "finalize"
    work.mkdir(parents=True, exist_ok=True)
    st_p = hub.get(STATE_PATH)
    if not st_p:
        raise SystemExit("no state on the hub: run the build first")
    state = json.loads(Path(st_p).read_text())
    ch, br = load_index(hub, work)
    if not len(ch):
        raise SystemExit("no chips yet")
    s = assign_split(ch, br, cfg)
    leak = leakage_tests(s, cfg)
    sp = work / "split.parquet"
    s[["chip_id", "shard", "split", "reason", "kind", "n_boxes", "source", "region"]].to_parquet(sp, index=False)
    lp = work / "leakage_tests.csv"
    leak.to_csv(lp, index=False)
    man = pd.DataFrame([{"file": f, **v} for f, v in state["files"].items()])
    man = pd.concat([man, pd.DataFrame([{"file": "splits/split.parquet", "sha256": sha256_file(sp), "bytes": os.path.getsize(sp),
                                         "rows": len(s)}])], ignore_index=True)
    mp = work / "MANIFEST.csv"
    man.to_csv(mp, index=False)
    card = work / "README.md"
    card.write_text(dataset_card(cfg, ch, s, br, leak, state, hub))
    rec = work / "dataset_record.md"
    rec.write_text(dataset_record(cfg, ch, s, br, leak, state, man))
    hub.commit({"splits/split.parquet": sp, "reports/leakage_tests.csv": lp, "MANIFEST.csv": mp, "README.md": card,
                "dataset_record.md": rec}, "finalize: split, leakage tests, manifest, dataset card, record")
    print(s.groupby("split").size().to_string())
    print(leak.to_string(index=False))
    print(f"[finalize] written to {hub.url()}")
    return s, leak

"""Export the Hub dataset to a YOLO-OBB folder for training, one shard at a time.

RAM stays flat: each shard is downloaded, read in small row batches (pyarrow), written to disk as
images + label .txt files, then the downloaded shard is deleted before the next one.

Output:
  <out>/images/{train,val,test}/OAM-x-y-z.png
  <out>/labels/{train,val,test}/OAM-x-y-z.txt      (empty file = negative chip)
  <out>/data.yaml                                  (path, train, val, test, names {0: bridge})
"""
import io
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .hub import open_hub


def to_yolo(cfg, out, splits=("train", "val", "test"), neg_keep=1.0, fmt="png", max_shards=None, batch_rows=64):
    hub = open_hub(cfg)
    out = Path(out)
    dl = Path(cfg.get("work_dir", "bb_work")) / "to_yolo_dl"
    dl.mkdir(parents=True, exist_ok=True)
    sp_path = hub.get("splits/split.parquet")
    if not sp_path:
        raise SystemExit("splits/split.parquet not found: run `finalize` first")
    split = pd.read_parquet(sp_path)
    split = split[split["split"].isin(splits)]
    want = dict(zip(split["chip_id"], split["split"]))
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    counts = {s: {"positive": 0, "negative": 0} for s in splits}
    shards = sorted(split["shard"].unique())[: max_shards or None]
    for k, shard in enumerate(shards):
        local = hub.get(shard, cache_dir=str(dl)) if hasattr(hub, "api") else hub.get(shard)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["chip_id", "file_name", "obb_yolo", "n_boxes", "image"]):
            for r in batch.to_pylist():
                s = want.get(r["chip_id"])
                if s is None:
                    continue
                neg = r["n_boxes"] == 0
                if neg and rng.random() > neg_keep:
                    continue
                stem = Path(r["file_name"]).stem
                if r["chip_id"].split("/")[0] != "oam_osm":           # keep names unique across sources
                    stem = f"{r['chip_id'].split('/')[0]}_{stem}"
                data = r["image"]["bytes"]
                if fmt == "png":
                    (out / "images" / s / f"{stem}.png").write_bytes(data)
                else:
                    from PIL import Image
                    Image.open(io.BytesIO(data)).convert("RGB").save(out / "images" / s / f"{stem}.jpg", quality=95)
                (out / "labels" / s / f"{stem}.txt").write_text((r["obb_yolo"] or "") + ("\n" if r["obb_yolo"] else ""))
                counts[s]["negative" if neg else "positive"] += 1
        del pf
        if hasattr(hub, "api"):                                        # drop the downloaded copy (HF cache)
            Path(local).unlink(missing_ok=True)
            shutil.rmtree(dl / ".cache", ignore_errors=True)
        print(f"[to-yolo] shard {k + 1}/{len(shards)} {shard} | " + ", ".join(f"{s}: {c['positive']}+{c['negative']}" for s, c in counts.items()))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n" + "".join(f"{s}: images/{s}\n" for s in splits) + "names:\n  0: bridge\n")
    print(f"[to-yolo] written {out / 'data.yaml'}")
    return counts

"""Analysis of the built dataset from the small index tables only (no images downloaded).

Writes CSV tables and PNG figures to <work_dir>/reports and, with push=True, to reports/ on the Hub.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .finalize import load_index
from .hub import open_hub


def stats(cfg, push=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hub = open_hub(cfg)
    rep = Path(cfg.get("work_dir", "bb_work")) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    ch, br = load_index(hub, rep / "_idx")
    if not len(ch):
        raise SystemExit("no chips yet")
    sp = hub.get("splits/split.parquet")
    if sp:
        ch = ch.merge(pd.read_parquet(sp)[["chip_id", "split"]].drop_duplicates("chip_id"), on="chip_id", how="left")
    else:
        ch["split"] = "not finalized"
    br = br.drop_duplicates("bridge_id") if len(br) else br
    t = {}
    t["chips_by_source_region"] = ch.groupby(["source", "region", "kind"]).size().unstack(fill_value=0).reset_index()
    t["chips_by_split"] = ch.groupby(["split", "source", "kind"]).size().unstack(fill_value=0).reset_index()
    t["boxes_per_positive_chip"] = ch[ch.n_boxes > 0]["n_boxes"].describe().to_frame().T
    t["images_used"] = ch.groupby(["source", "image_id", "provider", "licence_img"]).agg(
        chips=("chip_id", "count"), gsd_m=("gsd_native_m", "first"), date=("acq_date", "first")).reset_index()
    t["licences"] = ch.groupby(["licence_img", "licence_lbl"]).size().reset_index(name="chips")
    if len(br):
        t["bridges_by_source"] = br.groupby(["source", "cls", "crossing"]).agg(
            n=("bridge_id", "count"), length_m_median=("length_m", "median"), width_m_median=("width_m", "median")).reset_index()
        used = set(x for ids in ch["bridge_ids"] for x in ids)
        br["in_chips"] = br["bridge_id"].isin(used)
        t["bridge_coverage"] = br.groupby("source")["in_chips"].agg(["sum", "count", "mean"]).reset_index().rename(
            columns={"sum": "bridges_in_chips", "count": "bridges_found", "mean": "share"})
    for k, v in t.items():
        v.to_csv(rep / f"{k}.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ch.groupby(["source", "kind"]).size().unstack(fill_value=0).plot.bar(stacked=True, ax=ax[0, 0], title="Chips by source and kind")
    ax[0, 1].hist(ch.loc[ch.n_boxes > 0, "n_boxes"], bins=range(1, 12), color="#3a7")
    ax[0, 1].set_title("Boxes per positive chip")
    if len(br):
        ax[1, 0].hist(np.clip(br["length_m"], 0, 400), bins=40, color="#37a")
        ax[1, 0].set_title("Bridge length (m, clipped at 400)")
        ax[1, 1].hist(br["bearing_deg"], bins=36, color="#a73")
        ax[1, 1].set_title("Bridge direction (deg) - should be spread out")
    fig.tight_layout()
    fig.savefig(rep / "overview.png", dpi=110)
    plt.close(fig)
    # box size in pixels: are bridges big enough at 256 px?
    sides = []
    for s in ch.loc[ch.n_boxes > 0, "boxes_px"]:
        for c in json.loads(s):
            p = np.array(c).reshape(4, 2)
            e = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(2)]
            sides.append((max(e), min(e)))
    if sides:
        sd = np.array(sides)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sd[:, 0], sd[:, 1], s=4, alpha=.4)
        ax.set_xlabel("long side (px)"); ax.set_ylabel("short side (px)"); ax.set_title("Box sizes in 256 px chips")
        fig.tight_layout(); fig.savefig(rep / "box_sizes.png", dpi=110); plt.close(fig)
        pd.DataFrame(sd, columns=["long_px", "short_px"]).describe().to_csv(rep / "box_sizes.csv")
    print({k: len(v) for k, v in t.items()})
    print(t["chips_by_split"].to_string(index=False))
    if push:
        hub.commit({f"reports/{p.name}": p for p in rep.iterdir() if p.is_file()}, "reports: dataset statistics")
    return t, rep


def overlays(cfg, n=24, out=None, seed=0):
    """Overlay sample: draw the YOLO-OBB boxes on random positive chips (one shard downloaded). For the ticket's
    'overlay samples' and for a quick visual check of label alignment."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    from PIL import Image
    hub = open_hub(cfg)
    out = Path(out or Path(cfg.get("work_dir", "bb_work")) / "reports" / "overlay_samples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = hub.list("data/")
    rng = np.random.default_rng(seed)
    rows = []
    for sh in rng.permutation(shards)[:3]:
        t = pq.read_table(hub.get(sh), columns=["chip_id", "kind", "boxes_px", "image"]).to_pylist()
        rows += [r for r in t if r["kind"] == "positive"]
        if len(rows) >= n:
            break
    rows = [rows[i] for i in rng.permutation(len(rows))[:n]]
    cols = 6
    fig, axs = plt.subplots(int(np.ceil(len(rows) / cols)), cols, figsize=(cols * 2.6, np.ceil(len(rows) / cols) * 2.8), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, r in zip(axs.flat, rows):
        ax.imshow(Image.open(io.BytesIO(r["image"]["bytes"])))
        for c in json.loads(r["boxes_px"]):
            p = np.array(c).reshape(4, 2)
            ax.plot(*np.vstack([p, p[:1]]).T, color="yellow", lw=1.4)
        ax.set_title(r["chip_id"].split("/", 1)[1], fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[overlays] {out}")
    return out

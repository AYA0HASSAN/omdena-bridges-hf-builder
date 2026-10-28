"""Build step: cells -> chips -> Parquet shards on the Hub, in small batches, resumable.

Memory stays flat because:
- one cell (~4–5 km) is processed at a time, and imagery is read one 256 px chip at a time;
- chips are kept as PNG bytes and flushed to a Parquet shard every `shard_size` chips (default 500, ~50–100 MB);
- each shard is committed to the Hub together with its index and the updated state, then deleted locally.

Resuming: state/state.json on the Hub lists finished cells. Re-running the same command skips them, so a
Colab/Kaggle disconnect costs at most one shard of work.
"""
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .chips import cut_cell, bridge_rows
from .hub import open_hub
from .sources import SOURCES, cells_for_region

STATE_PATH = "state/state.json"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("hub", "work_dir")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_state(hub, cfg):
    p = hub.get(STATE_PATH)
    if p:
        s = json.loads(Path(p).read_text())
        if s.get("config_hash") != config_hash(cfg):
            print(f"[state] note: config changed since the last run ({s.get('config_hash')} -> {config_hash(cfg)}); "
                  "finished cells are still skipped")
        return s
    return {"version": 1, "created": _now(), "config_hash": config_hash(cfg), "done_cells": [], "failed_cells": {},
            "next_shard": 0, "files": {}, "totals": {"chips": 0, "positive": 0, "boxes": 0, "bridges": 0, "cells": 0}}


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _features():
    from datasets import Features, Value, Sequence, Image
    s, f, i = Value("string"), Value("float64"), Value("int64")
    return Features({
        "chip_id": s, "file_name": s, "source": s, "region": s, "country": s, "global_south": Value("bool"),
        "cell_id": s, "image_id": s, "image_url": s, "image_title": s, "provider": s, "acq_date": s,
        "licence_img": s, "licence_lbl": s, "label_source": s, "x": i, "y": i, "z": i, "lon": f, "lat": f,
        "bounds_3857": Sequence(f), "block_z14": s, "gsd_native_m": f, "gsd_chip_m": f, "kind": s, "n_boxes": i,
        "obb_yolo": s, "boxes_px": s, "bridge_ids": Sequence(s), "valid_frac": f, "phash": s, "img_sha256": s,
        "image": Image()})


class ShardWriter:
    def __init__(self, hub, state, cfg):
        self.hub, self.state, self.cfg = hub, state, cfg
        self.size = cfg.get("build", {}).get("shard_size", 500)
        self.work = Path(cfg.get("work_dir", "bb_work"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.rows, self.bridges, self.pending = [], [], []
        self.empty_since_commit = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.size:
            self.flush()

    def add_bridges(self, rows):
        self.bridges += rows

    def cell_done(self, key):
        self.pending.append(key)
        self.empty_since_commit += 1
        if not self.rows and self.empty_since_commit >= self.cfg.get("build", {}).get("state_every_cells", 25):
            self.flush(force=True)

    def flush(self, force=False):
        if not self.rows and not force:
            return
        st = self.state
        files = {}
        if self.rows:
            from datasets import Dataset
            n = st["next_shard"]
            shard = self.work / f"shard-{n:05d}.parquet"
            Dataset.from_list(self.rows, features=_features()).to_parquet(str(shard))
            meta = pd.DataFrame([{k: v for k, v in r.items() if k != "image"} for r in self.rows])
            meta["shard"] = f"data/shard-{n:05d}.parquet"
            meta["bridge_ids"] = meta["bridge_ids"].apply(json.dumps)
            meta["bounds_3857"] = meta["bounds_3857"].apply(json.dumps)
            idx = self.work / f"chips-{n:05d}.parquet"
            meta.to_parquet(idx, index=False)
            files[f"data/shard-{n:05d}.parquet"] = shard
            files[f"index/chips-{n:05d}.parquet"] = idx
            if self.bridges:
                br = self.work / f"bridges-{n:05d}.parquet"
                pd.DataFrame(self.bridges).to_parquet(br, index=False)
                files[f"index/bridges-{n:05d}.parquet"] = br
            for dst, src in files.items():
                st["files"][dst] = {"sha256": sha256_file(src), "bytes": os.path.getsize(src),
                                    "rows": len(self.rows) if dst.startswith(("data/", "index/chips")) else len(self.bridges)}
            t = st["totals"]
            t["chips"] += len(self.rows)
            t["positive"] += sum(r["kind"] == "positive" for r in self.rows)
            t["boxes"] += sum(r["n_boxes"] for r in self.rows)
            t["bridges"] += len(self.bridges)
            st["next_shard"] = n + 1
        elif self.bridges:                    # bridges from cells that produced no chips (no imagery): keep them too
            k = st.get("next_bridge_part", 0)
            br = self.work / f"bridges-extra-{k:05d}.parquet"
            pd.DataFrame(self.bridges).to_parquet(br, index=False)
            files[f"index/bridges-extra-{k:05d}.parquet"] = br
            st["files"][f"index/bridges-extra-{k:05d}.parquet"] = {"sha256": sha256_file(br), "bytes": os.path.getsize(br),
                                                                    "rows": len(self.bridges)}
            st["totals"]["bridges"] += len(self.bridges)
            st["next_bridge_part"] = k + 1
        st["done_cells"] += self.pending
        st["totals"]["cells"] = len(st["done_cells"])
        st["updated"] = _now()
        sp = self.work / "state.json"
        sp.write_text(json.dumps(st, indent=1))
        files[STATE_PATH] = sp
        msg = f"shard {st['next_shard'] - 1:05d}: {len(self.rows)} chips" if self.rows else f"state: {len(self.pending)} cells"
        self.hub.commit(files, msg)
        for dst, src in files.items():
            if dst != STATE_PATH:
                Path(src).unlink(missing_ok=True)
        self.rows, self.bridges, self.pending, self.empty_since_commit = [], [], [], 0
        gc.collect()


def _expand(b, margin_m):
    dy = margin_m / 111320
    dx = margin_m / (111320 * max(0.2, math.cos(math.radians((b[1] + b[3]) / 2))))
    return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)


def _ram_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def build(cfg, only=None, max_cells=None, max_hours=None, retry_failed=False):
    hub = open_hub(cfg)
    state = load_state(hub, cfg)
    w = ShardWriter(hub, state, cfg)
    done = set(state["done_cells"])
    if retry_failed:
        state["failed_cells"] = {}
    failed = set(state["failed_cells"])
    b = cfg.get("build", {})
    max_hours = max_hours or b.get("max_hours", 11.0)
    t0, n_cells = time.time(), 0
    print(f"[build] target: {hub.url()} | already done: {len(done)} cells, {state['totals']['chips']} chips")
    try:
        for src_name, scfg in cfg["sources"].items():
            if not scfg.get("enabled", True) or (only and src_name not in only):
                continue
            for region in scfg["regions"]:
                for cell_id, core in cells_for_region(region, scfg.get("cell_deg", b.get("cell_deg", 0.04))):
                    key = f"{src_name}:{cell_id}"
                    if key in done or key in failed:
                        continue
                    if max_cells is not None and n_cells >= max_cells:
                        raise StopIteration
                    if (time.time() - t0) / 3600 > max_hours:
                        print(f"[build] time budget of {max_hours} h reached: stopping cleanly (re-run to continue)")
                        raise StopIteration
                    tc = time.time()
                    try:
                        cell = SOURCES[src_name](region, cell_id, _expand(core, b.get("label_margin_m", 300)), scfg)
                        n_chip = n_pos = 0
                        for row in cut_cell(cell, core, cfg):
                            w.add(row)
                            n_chip += 1
                            n_pos += row["kind"] == "positive"
                        brs = [r for r in bridge_rows(cell) if core[0] <= r["lon"] < core[2] and core[1] <= r["lat"] < core[3]]
                        w.add_bridges(brs)
                        w.cell_done(key)
                        print(f"[build] {key}: {len(brs)} bridges, {len(cell.imagery)} image(s), {n_chip} chips "
                              f"({n_pos} with bridges) | {time.time() - tc:.0f}s | RAM {_ram_mb():.0f} MB"
                              + (f" | {'; '.join(cell.notes)[:120]}" if cell.notes else ""))
                    except Exception as e:
                        state["failed_cells"][key] = f"{type(e).__name__}: {str(e)[:200]}"
                        print(f"[build] {key}: FAILED ({type(e).__name__}: {str(e)[:150]}) - will be skipped; use --retry-failed")
                    finally:
                        cell = None
                        gc.collect()
                    n_cells += 1
    except StopIteration:
        pass
    w.flush(force=True)
    t = state["totals"]
    print(f"[build] done for now: {t['cells']} cells, {t['chips']} chips ({t['positive']} with bridges, {t['boxes']} boxes), "
          f"{len(state['failed_cells'])} failed cells, {state['next_shard']} shards")
    return state

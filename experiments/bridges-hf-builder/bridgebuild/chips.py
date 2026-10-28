"""Cut one cell into chips: positives (tiles with bridges) first, then a few negatives.

Only tiles whose centre lies inside the cell's *core* bbox are cut here. Labels are queried on a slightly
larger bbox, so a chip near the cell edge still sees every bridge that touches it, and no tile is cut twice
by neighbouring cells.
"""
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile
from shapely.geometry import box

from .geo import (CHIP, Z, tile_bounds, tiles_touching, obb_lines_for_tile, ground_m_per_px, rect_measures,
                  phash, valid_fraction, png_bytes)


def _centre_inside(t, core):
    b = mercantile.bounds(t)
    cx, cy = (b.west + b.east) / 2, (b.south + b.north) / 2
    return core[0] <= cx < core[2] and core[1] <= cy < core[3]


def _covered(t, fp, min_valid):
    tb = box(*tile_bounds(t))
    return fp is not None and tb.intersection(fp).area >= min_valid * tb.area


def bridge_rows(cell):
    """One row per bridge: geometry as delivered (source), deck, size, direction, crossing."""
    b = cell.bridges
    if not len(b):
        return []
    src = cell.sources.set_index("src_id") if len(cell.sources) and "src_id" in cell.sources else None
    water_u = None
    if len(cell.water):
        from shapely.ops import unary_union
        water_u = unary_union(cell.water.geometry)
    deck_ll = gpd.GeoSeries(b.geometry, crs=3857).to_crs(4326)
    rows = []
    for (i, r), dll in zip(b.iterrows(), deck_ll):
        c = mercantile.lnglat(*r.geometry.centroid.coords[0])
        L, W, brg = rect_measures(r.geometry)
        k = math.cos(math.radians(c.lat))
        crossing = "water" if (water_u is not None and r.geometry.intersects(water_u)) else \
            ("road/rail" if len(cell.roads) and cell.roads.intersects(r.geometry).any() else "unknown")
        s_geom, tags = None, {}
        if src is not None and r.src_id in src.index:
            sr = src.loc[r.src_id]
            sr = sr.iloc[0] if isinstance(sr, pd.DataFrame) else sr
            s_geom = sr.geometry
            tags = {kk: (None if (isinstance(v, float) and v != v) else str(v)) for kk, v in sr.items()
                    if kk != "geometry" and v is not None and not (isinstance(v, float) and v != v)}
        rows.append({"bridge_id": f"{cell.source}:{r.src_id}", "source": cell.source, "region": cell.region["name"],
                     "src_id": r.src_id, "cls": r.cls, "width_m_used": None if r.width_m != r.width_m else float(r.width_m),
                     "length_m": L * k, "width_m": W * k, "bearing_deg": brg, "lon": c.lng, "lat": c.lat,
                     "crossing": crossing, "deck_wkt": dll.wkt, "source_wkt": s_geom.wkt if s_geom is not None else None,
                     "tags_json": json.dumps(tags, ensure_ascii=False)[:4000]})
    return rows


def cut_cell(cell, core_bbox, cfg, seed=0):
    """Yield chip rows (dict) for one cell. Images are read one chip at a time and kept as PNG bytes."""
    rng = random.Random(f"{cell.cell_id}-{seed}")
    c = cfg.get("chips", {})
    min_valid, neg_ratio = c.get("min_valid", 0.9), c.get("neg_ratio", 0.5)
    max_pos, hard_share = c.get("max_pos_per_cell", 400), c.get("hard_negative_share", 0.5)
    b = cell.bridges
    if not len(b) or not cell.imagery:
        return
    b = b.reset_index(drop=True)
    b["bridge_id"] = [f"{cell.source}:{s}" for s in b["src_id"]]
    bridge_tiles = [t for t in tiles_touching(b) if _centre_inside(t, core_bbox)]
    done = set()
    made_pos = 0
    for im in cell.imagery:
        pos = [t for t in bridge_tiles if (t.x, t.y) not in done and _covered(t, im.footprint, min_valid)]
        rng.shuffle(pos)
        pos = pos[: max(0, max_pos - made_pos)]
        n_pos_here = 0
        for t in pos:
            row = _chip_row(cell, im, t, b, "positive", min_valid)
            if row is not None:
                done.add((t.x, t.y)); n_pos_here += 1; made_pos += 1
                yield row
        # negatives for this image
        n_neg = int(round(neg_ratio * n_pos_here))
        if n_neg <= 0:
            continue
        bset = set((t.x, t.y) for t in tiles_touching(b))
        hard = []
        ctx = [g for g in (cell.water, cell.roads) if g is not None and len(g)]
        if ctx:
            cg = gpd.GeoDataFrame(pd.concat([g[["geometry"]] for g in ctx]), crs=3857)
            cg = cg[cg.intersects(im.footprint)]
            if len(cg) > 2000:
                cg = cg.sample(2000, random_state=0)
            hard = [t for t in tiles_touching(cg) if (t.x, t.y) not in bset and (t.x, t.y) not in done
                    and _centre_inside(t, core_bbox) and _covered(t, im.footprint, min_valid)]
            rng.shuffle(hard)
        n_hard = min(len(hard), int(round(n_neg * hard_share)))
        w, s, e, n = core_bbox
        taken = set((t.x, t.y) for t in hard[:n_hard])     # a random negative must not repeat a hard one
        allt = []
        for _ in range(n_neg * 20):
            if len(allt) >= n_neg - n_hard:
                break
            t = mercantile.tile(rng.uniform(w, e), rng.uniform(s, n), Z)
            if (t.x, t.y) not in bset and (t.x, t.y) not in done and (t.x, t.y) not in taken and t not in allt and _covered(t, im.footprint, min_valid):
                allt.append(t)
        for kind, tiles in (("hard_negative", hard[:n_hard]), ("random_negative", allt)):
            for t in tiles:
                row = _chip_row(cell, im, t, b, kind, min_valid)
                if row is not None:
                    done.add((t.x, t.y))
                    yield row


def _chip_row(cell, im, t, b, kind, min_valid):
    bb = tile_bounds(t)
    try:
        img = im.reader(bb, (CHIP, CHIP))
    except Exception as e:
        print(f"  read failed {t}: {str(e)[:100]}")
        return None
    vf = valid_fraction(img)
    if vf < min_valid:
        return None
    lines, corners, idx = obb_lines_for_tile(b, t) if kind == "positive" else ([], [], [])
    if kind == "positive" and not lines:
        return None                                      # bridge pieces too small for this tile
    png = png_bytes(img)
    ll = mercantile.ul(t)
    p = mercantile.parent(t, zoom=14)
    name = f"OAM-{t.x}-{t.y}-{t.z}"
    return {
        "chip_id": f"{cell.source}/{name}", "file_name": f"{name}.png", "source": cell.source,
        "region": cell.region["name"], "country": cell.region.get("country"), "global_south": cell.region.get("global_south"),
        "cell_id": cell.cell_id, "image_id": im.image_id, "image_url": im.url, "image_title": im.title,
        "provider": im.provider, "acq_date": im.date, "licence_img": im.licence, "licence_lbl": cell.licence_lbl,
        "label_source": cell.label_source, "x": t.x, "y": t.y, "z": t.z, "lon": ll.lng, "lat": ll.lat,
        "bounds_3857": [float(v) for v in bb], "block_z14": f"{p.x}-{p.y}",
        "gsd_native_m": float(im.gsd_m), "gsd_chip_m": ground_m_per_px(ll.lat), "kind": kind,
        "n_boxes": len(lines), "obb_yolo": "\n".join(lines),
        "boxes_px": json.dumps([[round(v, 2) for p_ in cr for v in p_] for cr in corners]),
        "bridge_ids": [b.loc[i, "bridge_id"] for i in idx], "valid_frac": round(vf, 4),
        "phash": phash(img), "img_sha256": hashlib.sha256(png).hexdigest(),
        "image": {"bytes": png, "path": f"{name}.png"},
    }

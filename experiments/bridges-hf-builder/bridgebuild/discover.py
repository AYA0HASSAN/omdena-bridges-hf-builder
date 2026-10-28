"""Find where to build: OAM images with an allowed licence and enough resolution, ranked by how many
OSM bridges sit inside each one (Overpass `out count`, cheap). Use the result to pick `regions:` so
the build spends its time on imagery that actually has bridges.
"""
import pandas as pd
from shapely.geometry import box

from . import net
from .sources import oam_images


def count_osm_bridges(bbox):
    w, s, e, n = bbox
    q = f'[out:json][timeout:60];(way["bridge"]["bridge"!="no"]["highway"]({s},{w},{n},{e});' \
        f'way["bridge"]["bridge"!="no"]["railway"]({s},{w},{n},{e});way["man_made"="bridge"]({s},{w},{n},{e}););out count;'
    js = net.overpass(q, timeout=60).json()
    try:
        return int(js["elements"][0]["tags"]["total"])
    except Exception:
        return 0


def discover(search_bbox, max_gsd_m=0.6, allowed=("CC-BY 4.0", "CC-BY-SA 4.0", "CC0"), max_images=60, out_csv=None):
    imgs = oam_images(search_bbox, max_gsd_m, list(allowed), max_pages=10)
    rows = []
    for im in imgs[:max_images]:
        fp = im["footprint_ll"].intersection(box(*search_bbox))
        if fp.is_empty:
            continue
        b = fp.bounds
        area_km2 = (b[2] - b[0]) * (b[3] - b[1]) * 111 * 111
        if area_km2 > 2500:                               # very large mosaics: count would be slow; skip
            n = None
        else:
            try:
                n = count_osm_bridges(b)
            except Exception as e:
                print("  count failed:", str(e)[:80]); n = None
        rows.append({"image_id": im["image_id"], "title": im["title"], "provider": im["provider"], "licence": im["licence"],
                     "gsd_m": im["gsd"], "date": im["date"], "bbox": [round(v, 5) for v in b], "area_km2": round(area_km2, 1),
                     "osm_bridges": n, "url": im["url"]})
        print(f"  {im['image_id']}: {n} bridges, {area_km2:.0f} km2, {im['gsd']:.2f} m, {im['licence']}")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("osm_bridges", ascending=False, na_position="last")
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def as_regions(df, prefix, top=10, min_bridges=5, country=None, global_south=None):
    """Turn the best rows into `regions:` entries for the config (YAML-ready dicts)."""
    d = df[df["osm_bridges"].fillna(0) >= min_bridges].head(top)
    return [{"name": f"{prefix}_{i:02d}", "bbox": list(r.bbox), "country": country, "global_south": global_south,
             "note": f"OAM {r.image_id} ({r.provider}, {r.gsd_m:.2f} m), ~{int(r.osm_bridges)} OSM bridges"}
            for i, r in enumerate(d.itertuples())]

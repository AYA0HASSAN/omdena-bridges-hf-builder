"""Georeferenced helpers: imagery readers, OSM bridges, the bridge filter (catalogue §3.4), boxes on the z18 grid."""
import io
import math
import os
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import box, LineString, Polygon
from shapely.ops import substring, polygonize, unary_union

from . import net

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff,.TIFF")
os.environ.setdefault("GDAL_CACHEMAX", "256")            # MB: keeps GDAL's block cache small on free notebooks
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
Z = 18
CHIP = 256


def bbox_3857(bbox_ll):
    x0, y0 = _TO_3857.transform(bbox_ll[0], bbox_ll[1])
    x1, y1 = _TO_3857.transform(bbox_ll[2], bbox_ll[3])
    return (x0, y0, x1, y1)


def tile_bounds(t):
    b = mercantile.xy_bounds(t)
    return (b.left, b.bottom, b.right, b.top)


def ground_m_per_px(lat, z=Z):
    return 156543.03392 / 2 ** z * math.cos(math.radians(lat))


# ---------------------------------------------------------------- imagery
def to_uint8_rgb(arr, limits=None, p=(2, 98)):
    """3-band uint8: keep bands 1-3, stretch non-uint8 data (one `limits` per source image)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a[..., :3] if a.shape[-1] >= 3 else np.repeat(a[..., :1], 3, -1)
    if a.dtype == np.uint8 and limits is None:
        return a
    out = np.zeros(a.shape, np.uint8)
    for i in range(3):
        b = a[..., i].astype("float32")
        if limits is not None:
            lo, hi = limits[i]
        else:
            v = b[b > 0]
            lo, hi = np.percentile(v, p) if v.size else (0.0, 1.0)
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def wms_getmap(url, layer, bounds, width, height, fmt="image/jpeg", version="1.3.0", crs="EPSG:3857"):
    key = "CRS" if version == "1.3.0" else "SRS"
    params = {"SERVICE": "WMS", "VERSION": version, "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "",
              key: crs, "BBOX": ",".join(f"{v:.3f}" for v in bounds),
              "WIDTH": int(width), "HEIGHT": int(height), "FORMAT": fmt}
    r = net.get(url, params=params)
    if "image" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"WMS did not return an image: {r.text[:300]}")
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))


def wms_layers(url, version="1.3.0", contains=""):
    r = net.get(url, params={"SERVICE": "WMS", "REQUEST": "GetCapabilities", "VERSION": version})
    return [n for n in re.findall(r"<Name>([^<]+)</Name>", r.text) if contains.lower() in n.lower()]


def cog_read(url, bounds3857, out_size, bands=(1, 2, 3), limits=None):
    """Warp a (C)OG to EPSG:3857 onto exactly these bounds/size -> RGB uint8 (outside = 0)."""
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    w, h = int(out_size[0]), int(out_size[1])
    with rasterio.open(url) as src, WarpedVRT(src, crs="EPSG:3857", transform=from_bounds(*bounds3857, w, h),
                                             width=w, height=h, resampling=Resampling.bilinear) as vrt:
        arr = np.moveaxis(vrt.read(list(bands)), 0, -1)
    return arr if (arr.dtype == np.uint8 and limits is None) else to_uint8_rgb(arr, limits=limits)


def stretch_limits(url, bands=(1, 2, 3), p=(2, 98), max_px=1024):
    with rasterio.open(url) as src:
        k = max(1, math.ceil(max(src.width, src.height) / max_px))
        a = src.read(list(bands), out_shape=(len(bands), max(1, src.height // k), max(1, src.width // k)))
    return [tuple(np.percentile(b[b > 0], p)) if (b > 0).any() else (0, 1) for b in a]


def cog_footprint_3857(url):
    from rasterio.warp import transform_bounds
    with rasterio.open(url) as src:
        return box(*transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21))


# ---------------------------------------------------------------- OSM
OVERPASS = "https://overpass-api.de/api/interpreter"


def overpass_bridge_context(bbox_ll, timeout=180):
    s, w, n, e = bbox_ll[1], bbox_ll[0], bbox_ll[3], bbox_ll[2]
    bb = f"({s},{w},{n},{e})"
    q = f"""[out:json][timeout:{timeout}];
(
  way["man_made"="bridge"]{bb};
  relation["man_made"="bridge"]{bb};
  way["bridge"]{bb};
  way["waterway"]{bb};
  way["natural"="water"]{bb};
  way["highway"]{bb};
  way["railway"]{bb};
);
out tags geom;"""
    r = net.overpass(q, timeout=timeout)
    rows = []
    for el in r.json()["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            closed = len(pts) > 3 and pts[0] == pts[-1]
            areal = closed and (tags.get("man_made") == "bridge" or tags.get("natural") == "water" or tags.get("area") == "yes")
            geom = Polygon(pts) if areal else LineString(pts)
        elif el["type"] == "relation":
            outers = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                      for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
            polys = list(polygonize(unary_union(outers))) if outers else []
            if not polys:
                continue
            geom = unary_union(polys)
        else:
            continue
        rows.append({"src_id": f"osm/{el['type']}/{el['id']}", "geometry": geom, **tags})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(
        {"src_id": []}, geometry=[], crs=4326)
    for col in ["man_made", "bridge", "waterway", "natural", "highway", "railway", "layer", "width", "tunnel", "name"]:
        if col not in gdf:
            gdf[col] = None
    return gdf


def split_osm(gdf):
    outlines = gdf[(gdf["man_made"] == "bridge") & gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    bridge_ways = gdf[gdf["bridge"].notna() & (gdf["bridge"] != "no") & (gdf.geom_type == "LineString")]
    water = gdf[gdf["waterway"].notna() | (gdf["natural"] == "water")]
    ground = gdf[(gdf["highway"].notna() | gdf["railway"].notna()) & (gdf["bridge"].isna() | (gdf["bridge"] == "no"))
                 & (gdf.geom_type == "LineString")]
    return outlines.copy(), bridge_ways.copy(), water.copy(), ground.copy()


BRIDGE_OK = {"yes", "viaduct", "aqueduct", "movable", "cantilever", "covered", "trestle"}
DEFAULT_WIDTH_M = {"motorway": 20, "trunk": 15, "primary": 12, "secondary": 10, "tertiary": 8,
                   "residential": 7, "unclassified": 6, "living_street": 6, "service": 5, "track": 4,
                   "footway": 3, "path": 2.5, "cycleway": 3, "pedestrian": 4, "steps": 3, "bridleway": 3}
RAIL_WIDTH_M = 6


def way_width(row):
    try:
        val = float(str(row.get("width")).replace("m", "").strip())
        if math.isfinite(val) and val > 0:
            return val
    except (TypeError, ValueError):
        pass
    if row.get("railway"):
        return RAIL_WIDTH_M
    return DEFAULT_WIDTH_M.get(row.get("highway"), 6)


def _layer(row):
    try:
        return float(row.get("layer"))
    except (TypeError, ValueError):
        return 0.0


def filter_bridge_lines(bridge_ways, outlines, water, ground, value_col="bridge", ok_values=BRIDGE_OK, min_len_m=6.0):
    """Catalogue §3.4 rules -> (kept, rejected) in a metric CRS with a 'reason' column."""
    if bridge_ways.empty:
        e = bridge_ways.assign(reason=pd.Series(dtype=str), width_m=pd.Series(dtype=float))
        return e, e
    crs_m = bridge_ways.estimate_utm_crs()
    bw = bridge_ways.to_crs(crs_m).copy()
    wat = unary_union(water.to_crs(crs_m).geometry) if len(water) else None
    gnd = ground.to_crs(crs_m) if len(ground) else ground
    outl = unary_union(outlines.to_crs(crs_m).geometry) if len(outlines) else None
    reasons = []
    for _, row in bw.iterrows():
        g = row.geometry
        val = str(row.get(value_col)).lower()
        if val not in ok_values:
            reasons.append(f"value '{val}' not a bridge"); continue
        if str(row.get("tunnel")) == "culvert":
            reasons.append("culvert"); continue
        if g.length < min_len_m:
            reasons.append(f"shorter than {min_len_m:g} m"); continue
        if outl is not None and g.intersects(outl):
            reasons.append("covered by outline (use outline)"); continue
        inner = substring(g, 1.0, max(1.0, g.length - 1.0)) if g.length > 3 else g
        crosses_water = wat is not None and inner.intersects(wat)
        crosses_way = len(gnd) > 0 and gnd.intersects(inner).any()
        if crosses_water or crosses_way:
            reasons.append("ok: crosses water" if crosses_water else "ok: crosses road/rail"); continue
        if _layer(row) >= 1:
            reasons.append("ok: layer>=1 only (review)"); continue
        reasons.append("crosses nothing")
    bw["reason"] = reasons
    bw["width_m"] = [way_width(r) for _, r in bw.iterrows()]
    keep = bw["reason"].str.startswith("ok")
    return bw[keep].copy(), bw[~keep].copy()


def lines_to_decks(lines_m, width_col="width_m"):
    out = lines_m.copy()
    out["geometry"] = [g.buffer(w / 2, cap_style=2) for g, w in zip(out.geometry, out[width_col])]
    return out


def osm_bridges(bbox_ll):
    """OSM -> (bridges 3857 deck polygons, source labels 4326, water 3857, roads 3857, filter report)."""
    osm = overpass_bridge_context(bbox_ll)
    outlines, bridge_ways, water, ground = split_osm(osm)
    kept, rejected = filter_bridge_lines(bridge_ways, outlines, water, ground)
    decks = lines_to_decks(kept) if len(kept) else kept
    parts = []
    if len(outlines):
        o = outlines.to_crs(3857)[["src_id", "geometry"]].copy()
        o["cls"], o["width_m"] = "S", np.nan
        parts.append(o)
    if len(decks):
        d = decks.to_crs(3857)[["src_id", "width_m", "geometry"]].copy()
        d["cls"] = "F"
        parts.append(d)
    bridges = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=3857) if parts else empty_bridges()
    src = gpd.GeoDataFrame(pd.concat([outlines, bridge_ways]), crs=4326) if (len(outlines) or len(bridge_ways)) \
        else gpd.GeoDataFrame({"src_id": []}, geometry=[], crs=4326)
    report = pd.concat([kept, rejected])[["src_id", "reason"]] if (len(kept) or len(rejected)) else pd.DataFrame()
    roads = ground.to_crs(3857).copy()
    if len(roads):
        roads["width_m"] = [way_width(r) for _, r in ground.iterrows()]
    return bridges, src, water.to_crs(3857), roads, report


def empty_bridges():
    return gpd.GeoDataFrame({"src_id": [], "cls": [], "width_m": []}, geometry=[], crs=3857)


# ---------------------------------------------------------------- chips and boxes
def obb_lines_for_tile(polys_3857, tile, size=CHIP, min_px=12):
    """Clip polygons to a z18 tile -> (YOLO-OBB lines, pixel corners, index of the polygon for each line)."""
    b = mercantile.xy_bounds(tile)
    tb = box(b.left, b.bottom, b.right, b.top)
    sx = size / (b.right - b.left)
    lines, corners, idx = [], [], []
    for i, g in zip(polys_3857.index, polys_3857.geometry):
        c = g.intersection(tb)
        if c.is_empty or c.area == 0:
            continue
        pts = list(c.minimum_rotated_rectangle.exterior.coords)[:4]
        px = [((x - b.left) * sx, (b.top - y) * sx) for x, y in pts]
        sides = [math.dist(px[i_], px[(i_ + 1) % 4]) for i_ in range(4)]
        if max(sides) < min_px or min(sides) < 2:
            continue
        corners.append(px)
        idx.append(i)
        lines.append("0 " + " ".join(f"{min(max(v / size, 0), 1):.6f}" for p in px for v in p))
    return lines, corners, idx


def tiles_touching(polys_3857, z=Z):
    tiles = set()
    for g in polys_3857.to_crs(4326).geometry:
        w, s, e, n = g.bounds
        tiles.update(mercantile.tiles(w, s, e, n, z))
    return sorted(tiles, key=lambda t: (t.y, t.x))


def save_geotiff_chip(img, tile, path):
    b = tile_bounds(tile)
    with rasterio.open(path, "w", driver="GTiff", width=img.shape[1], height=img.shape[0], count=3, dtype="uint8",
                       crs="EPSG:3857", transform=from_bounds(*b, img.shape[1], img.shape[0]), compress="deflate") as d:
        d.write(np.moveaxis(img[:, :, :3], -1, 0))


def rect_measures(geom_m):
    r = geom_m.minimum_rotated_rectangle
    c = list(r.exterior.coords)[:4]
    e = [math.dist(c[i], c[(i + 1) % 4]) for i in range(2)]
    i = int(np.argmax(e))
    dx, dy = c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]
    return max(e), min(e), (math.degrees(math.atan2(dx, dy)) + 180) % 180


def phash(img, n=8):
    """64-bit perceptual hash (DCT of a 32 x 32 grey thumbnail), hex string. Used to find near-duplicates."""
    from scipy.fft import dctn
    g = np.asarray(Image.fromarray(img).convert("L").resize((32, 32), Image.BILINEAR), dtype=float)
    d = dctn(g, norm="ortho")[:n, :n]
    bits = (d > np.median(d[1:, 1:])).flatten()
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)


def valid_fraction(img):
    return float((img.sum(-1) > 0).mean())


def png_bytes(img):
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return b.getvalue()

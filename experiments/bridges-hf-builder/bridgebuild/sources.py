"""Sources, cut into small cells so every step stays small in memory.

A region (bbox) is split into cells of `cell_deg` degrees (~4–5 km). For each cell a source returns
a CellData: the bridges (deck polygons, EPSG:3857), their source geometry as delivered (EPSG:4326),
water and roads for negatives, and one or more imagery readers with their footprints.

Sources implemented (phase 1):
  oam_osm  OpenAerialMap imagery (CC BY 4.0 family) + OSM bridges (ODbL)
  france   BD ORTHO 20 cm (WMS) + BD TOPO bridges (WFS), Etalab 2.0
"""
import io
import json
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shp_wkt
from shapely.geometry import box, shape
from shapely.ops import unary_union

from . import net
from .geo import (bbox_3857, cog_read, wms_getmap, osm_bridges, overpass_bridge_context, split_osm, way_width,
                  lines_to_decks, empty_bridges)

_TO_LL = None


@dataclass
class Imagery:
    image_id: str
    reader: Callable                     # reader(bounds3857, (w, h)) -> RGB uint8
    footprint: object                    # shapely polygon, EPSG:3857
    gsd_m: float
    licence: str
    provider: str = ""
    date: str = ""
    url: str = ""
    title: str = ""


@dataclass
class CellData:
    source: str
    region: dict
    cell_id: str
    bbox: tuple
    bridges: gpd.GeoDataFrame            # EPSG:3857: bridge_id, src_id, cls, width_m, geometry (deck)
    sources: gpd.GeoDataFrame            # EPSG:4326: src_id + tags, geometry as delivered
    water: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    imagery: List[Imagery] = field(default_factory=list)
    licence_lbl: str = ""
    label_source: str = ""
    notes: list = field(default_factory=list)


def cells_for_region(region, cell_deg):
    """Split a region bbox into a grid of cells; ids are stable across runs (for resuming)."""
    x0, y0, x1, y1 = region["bbox"]
    nx, ny = max(1, math.ceil((x1 - x0) / cell_deg)), max(1, math.ceil((y1 - y0) / cell_deg))
    for j in range(ny):
        for i in range(nx):
            b = (x0 + i * cell_deg, y0 + j * cell_deg, min(x1, x0 + (i + 1) * cell_deg), min(y1, y0 + (j + 1) * cell_deg))
            yield f"{region['name']}_{i:03d}_{j:03d}", b


def _norm_licence(s):
    return str(s or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def licence_allowed(lic, allowed):
    """True if the image licence matches one of the allowed ones (e.g. 'CC-BY 4.0'); non-commercial is always refused."""
    n = _norm_licence(lic)
    if not n or "bync" in n or "noncommercial" in n:
        return False
    return any(_norm_licence(a) in n for a in allowed)


# =================================================================== OAM + OSM
def oam_images(bbox, max_gsd_m, allowed_licences, limit=100, max_pages=5):
    rows, page = [], 1
    while page <= max_pages:
        r = net.get("https://api.openaerialmap.org/meta",
                    params={"bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": limit, "page": page})
        js = r.json()
        res = js.get("results", [])
        for x in res:
            url = str(x.get("uuid") or "")
            if not url.lower().endswith((".tif", ".tiff")):
                continue
            gsd = x.get("gsd")
            if gsd is None or gsd > max_gsd_m:
                continue
            lic = (x.get("properties") or {}).get("license") or x.get("license") or ""
            if not licence_allowed(lic, allowed_licences):
                continue
            fp = None
            if x.get("footprint"):
                try:
                    fp = shp_wkt.loads(x["footprint"]) if isinstance(x["footprint"], str) else shape(x["footprint"])
                except Exception:
                    fp = None
            if fp is None and x.get("bbox"):
                fp = box(*x["bbox"])
            if fp is None:
                continue
            rows.append({"image_id": str(x.get("_id") or url.rsplit("/", 1)[-1]), "url": url, "gsd": float(gsd),
                         "licence": lic, "provider": x.get("provider", ""), "title": x.get("title", ""),
                         "date": str(x.get("acquisition_start", ""))[:10], "footprint_ll": fp})
        found = (js.get("meta") or {}).get("found", len(res))
        if len(res) < limit or page * limit >= found:
            break
        page += 1
    return rows


def oam_osm_cell(region, cell_id, bbox, opts):
    bridges, src, water, roads, report = osm_bridges(bbox)
    cell = CellData("oam_osm", region, cell_id, bbox, bridges, src, water, roads, licence_lbl="ODbL 1.0",
                    label_source="OpenStreetMap")
    if report is not None and len(report):
        cell.notes.append("filter: " + json.dumps(report["reason"].value_counts().to_dict()))
    if not len(bridges):
        return cell                                       # nothing to chip: skip the imagery query
    imgs = oam_images(bbox, opts.get("max_gsd_m", 0.6), opts.get("allowed_licences", ["CC-BY 4.0", "CC0"]))
    if not imgs:
        cell.notes.append("no OAM image with an allowed licence and GSD")
        return cell
    b_ll = gpd.GeoSeries(bridges.geometry, crs=3857).to_crs(4326)
    for im in imgs:                                       # score: bridges inside the footprint, then finer GSD
        im["n_bridges"] = int(b_ll.intersects(im["footprint_ll"]).sum())
    imgs = [im for im in imgs if im["n_bridges"] > 0]
    imgs.sort(key=lambda im: (-im["n_bridges"], im["gsd"]))
    for im in imgs[: opts.get("images_per_cell", 2)]:
        fp3857 = gpd.GeoSeries([im["footprint_ll"]], crs=4326).to_crs(3857).iloc[0].intersection(box(*bbox_3857(bbox)))
        cell.imagery.append(Imagery(im["image_id"], (lambda bb, sz, u=im["url"]: cog_read(u, bb, sz)), fp3857, im["gsd"],
                                    im["licence"], im["provider"], im["date"], im["url"], im["title"]))
    if not cell.imagery:
        cell.notes.append("OAM images here do not cover any bridge")
    return cell


# =================================================================== France
def france_cell(region, cell_id, bbox, opts):
    wfs = opts.get("wfs", "https://data.geopf.fr/wfs/ows")
    wms = opts.get("wms", "https://data.geopf.fr/wms-r/wms")
    layer = opts.get("wms_layer", "HR.ORTHOIMAGERY.ORTHOPHOTOS")
    b3857 = bbox_3857(bbox)

    def wfs_get(t):
        params = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": t, "SRSNAME": "EPSG:3857",
                  "BBOX": ",".join(f"{v:.2f}" for v in b3857) + ",EPSG:3857", "OUTPUTFORMAT": "application/json", "COUNT": 5000}
        g = gpd.read_file(io.BytesIO(net.get(wfs, params=params, timeout=180).content))
        return g.set_crs(3857, allow_override=True) if len(g) else gpd.GeoDataFrame({"nature": []}, geometry=[], crs=3857)

    surf, line = wfs_get("BDTOPO_V3:construction_surfacique"), wfs_get("BDTOPO_V3:construction_lineaire")
    surf_b = surf[surf["nature"].astype(str).str.lower() == "pont"].copy() if len(surf) else surf
    line_b = line[line["nature"].astype(str).str.lower().str.contains(r"^(?:pont|viaduc|passerelle)", regex=True)].copy() if len(line) else line
    ids = lambda g, p: [f"bdtopo/{p}/{c}" for c in (g["cleabs"] if "cleabs" in g else range(len(g)))]
    surf_b["src_id"] = ids(surf_b, "surf")
    line_b["src_id"] = ids(line_b, "line")
    parts = []
    if len(surf_b):
        parts.append(surf_b.assign(cls="S", width_m=np.nan)[["src_id", "cls", "width_m", "geometry"]])
    if len(line_b):
        crs_m = gpd.GeoSeries([box(*bbox)], crs=4326).estimate_utm_crs()
        lb = line_b.to_crs(crs_m).copy()
        lb["width_m"] = [3.0 if str(n).lower().startswith("passerelle") else opts.get("line_width_m", 6.0) for n in lb["nature"]]
        decks = lines_to_decks(lb).to_crs(3857)
        if len(surf_b):                                    # prefer the surface where both exist
            decks = decks[~decks.intersects(unary_union(surf_b.geometry))]
        if len(decks):
            parts.append(decks.assign(cls="F")[["src_id", "cls", "width_m", "geometry"]])
    bridges = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=3857) if parts else empty_bridges()
    src = gpd.GeoDataFrame(pd.concat([surf_b, line_b], ignore_index=True), crs=3857).to_crs(4326) if (len(surf_b) or len(line_b)) \
        else gpd.GeoDataFrame({"src_id": []}, geometry=[], crs=4326)
    water = roads = gpd.GeoDataFrame(geometry=[], crs=3857)
    if len(bridges) and opts.get("osm_context", True):      # water / roads for hard negatives
        try:
            osm = overpass_bridge_context(bbox)
            _, _, w, g = split_osm(osm)
            water, roads = w.to_crs(3857), g.to_crs(3857)
        except Exception as e:
            print("  OSM context failed:", str(e)[:100])
    cell = CellData("france", region, cell_id, bbox, bridges, src, water, roads, licence_lbl="Etalab 2.0 (IGN BD TOPO)",
                    label_source="IGN BD TOPO")
    if len(bridges):
        cell.imagery.append(Imagery(f"bdortho/{cell_id}", lambda bb, sz: wms_getmap(wms, layer, bb, sz[0], sz[1]),
                                    box(*b3857), opts.get("native_res_m", 0.20), "Etalab 2.0 (IGN BD ORTHO)", "IGN", "",
                                    wms, "BD ORTHO 20 cm"))
    return cell


SOURCES = {"oam_osm": oam_osm_cell, "france": france_cell}

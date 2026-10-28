"""A small synthetic world per bbox: a river, roads, two bridges, buildings, and distractors.

Imagery is rendered from the TRUE geometry; labels are served shifted by a known error (dx, dy metres),
so the offset stage can be checked end to end without any real service.
"""
import math

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon, box, Point
from shapely import affinity


class World:
    def __init__(self, bbox, seed=0, label_shift_m=(0.0, 0.0), palette="aerial"):
        self.bbox = bbox
        self.rng = np.random.default_rng(seed)
        self.crs_m = gpd.GeoSeries([box(*bbox)], crs=4326).estimate_utm_crs()
        b = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(self.crs_m).total_bounds
        self.bm = b
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        r = self.rng
        a = r.uniform(20, 70)
        u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        L = 4000
        self.river_axis = LineString([(cx - u[0] * L, cy - u[1] * L), (cx + u[0] * L, cy + u[1] * L)])
        self.river = self.river_axis.buffer(45, cap_style=2)
        n = np.array([-u[1], u[0]])
        roads = []
        for k, off in enumerate((-250, 0, 260)):           # 3 roads crossing the river (one per bridge + one more)
            p = np.array([cx, cy]) + u * off
            ang = r.uniform(-25, 25)
            v = np.array([math.cos(math.radians(ang)) * n[0] - math.sin(math.radians(ang)) * n[1],
                          math.sin(math.radians(ang)) * n[0] + math.cos(math.radians(ang)) * n[1]])
            roads.append(("primary" if k != 1 else "secondary", 10.0 if k != 1 else 8.0, LineString([tuple(p - v * L), tuple(p + v * L)])))
        for k in range(3):                                  # roads along the banks and a grid street
            off = (-1) ** k * r.uniform(90, 300)
            p = np.array([cx, cy]) + n * off
            roads.append(("residential", 7.0, LineString([tuple(p - u * L), tuple(p + u * L)])))
        self.roads = roads
        self.bridges = []                                   # (kind, deck polygon, centre line, width)
        for i, (hw, w, line) in enumerate(roads[:3]):
            seg = line.intersection(self.river.buffer(12))
            seg = max(getattr(seg, "geoms", [seg]), key=lambda g: g.length)
            self.bridges.append(("outline" if i == 1 else "line", seg.buffer(w / 2 + 1, cap_style=2), seg, w + 2))
        self.buildings = []
        for _ in range(250):
            p = Point(r.uniform(b[0], b[2]), r.uniform(b[1], b[3]))
            bx = affinity.rotate(box(p.x - r.uniform(6, 15), p.y - r.uniform(6, 15), p.x + r.uniform(6, 15), p.y + r.uniform(6, 15)), r.uniform(0, 90))
            if not bx.intersects(self.river.buffer(10)) and not any(bx.intersects(l.buffer(w)) for _, w, l in roads):
                self.buildings.append(bx)
        self.dx, self.dy = label_shift_m
        pal = {"aerial": ((96, 110, 70), (40, 62, 70), (150, 150, 145), (205, 200, 190), (170, 95, 80)),
               "satellite": ((120, 115, 95), (55, 70, 85), (160, 158, 150), (215, 210, 200), (150, 140, 130)),
               "dry": ((165, 140, 105), (70, 90, 95), (175, 170, 160), (225, 220, 210), (190, 160, 130))}[palette]
        self.pal = pal

    # ---------------------------------------------------------------- rendering
    def render(self, bounds, crs, w, h, dtype="uint8", bands=3):
        t = from_bounds(*bounds, w, h)
        to = lambda geoms: gpd.GeoSeries(geoms, crs=self.crs_m).to_crs(crs).tolist()
        bg, water, road, deck, roof = self.pal
        img = np.zeros((h, w, 3), np.float32) + np.array(bg, np.float32)
        noise = self.rng.normal(0, 10, (h, w, 1))
        layers = [(to([self.river]), water), (to([l.buffer(wd / 2, cap_style=2) for _, wd, l in self.roads]), road),
                  (to(self.buildings), roof), (to([d for _, d, _, _ in self.bridges]), deck)]
        for geoms, col in layers:
            m = rasterize([(g, 1) for g in geoms if not g.is_empty], out_shape=(h, w), transform=t, fill=0, all_touched=False).astype(bool)
            img[m] = col
        img = np.clip(img + noise, 0, 255)
        if dtype == "uint16":
            img = (img * 8).astype(np.uint16)
        else:
            img = img.astype(np.uint8)
        if bands == 4:
            img = np.concatenate([img, img[..., 1:2]], -1)
        return img

    def write_cog(self, path, crs, res_m, dtype="uint8", bands=3):
        b = gpd.GeoSeries([box(*self.bbox)], crs=4326).to_crs(crs).total_bounds
        if gpd.GeoSeries([box(*self.bbox)], crs=4326).to_crs(crs).crs.is_geographic:
            res = res_m / 111320
        else:
            res = res_m
        w, h = int((b[2] - b[0]) / res), int((b[3] - b[1]) / res)
        img = self.render(tuple(b), crs, w, h, dtype, bands)
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=img.shape[2], dtype=img.dtype, crs=crs,
                           transform=from_bounds(*b, w, h), tiled=True, compress="deflate") as d:
            d.write(np.moveaxis(img, -1, 0))
            d.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
        return str(path)

    # ---------------------------------------------------------------- labels (shifted by the known error)
    def shifted(self, g):
        return affinity.translate(g, self.dx, self.dy)

    def to4326(self, g):
        return gpd.GeoSeries([self.shifted(g)], crs=self.crs_m).to_crs(4326).iloc[0]

    def overpass_elements(self):
        els, k = [], 1

        def way(geom, **tags):
            nonlocal k
            k += 1
            return {"type": "way", "id": k, "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in geom.coords]}
        river_poly = self.to4326(self.river)
        els.append(way(river_poly.exterior, natural="water"))
        els.append(way(self.to4326(self.river_axis.intersection(box(*self.bm))), waterway="river"))
        for i, (hw, w, line) in enumerate(self.roads):
            line_c = line.intersection(box(*self.bm))
            if i < 3:
                _, deck, seg, _ = self.bridges[i]
                pa, pb = seg.coords[0], seg.coords[-1]
                d0, d1 = sorted([line_c.project(Point(pa)), line_c.project(Point(pb))])
                from shapely.ops import substring
                for part, tags in ((substring(line_c, 0, d0), {}), (substring(line_c, d0, d1), {"bridge": "yes", "layer": "1"}),
                                   (substring(line_c, d1, line_c.length), {})):
                    els.append(way(self.to4326(part), highway=hw, **tags))
                if self.bridges[i][0] == "outline":
                    els.append(way(self.to4326(deck).exterior, man_made="bridge"))
            else:
                els.append(way(self.to4326(line_c), highway=hw))
        # distractors: a boardwalk and a culvert
        c = self.river.centroid
        els.append(way(self.to4326(LineString([(c.x + 400, c.y + 400), (c.x + 430, c.y + 400)])), highway="footway", bridge="boardwalk"))
        els.append(way(self.to4326(LineString([(c.x - 500, c.y + 300), (c.x - 480, c.y + 300)])), highway="residential", bridge="yes", tunnel="culvert"))
        return els

    def deck_polys_4326(self):
        return [self.to4326(d) for _, d, _, _ in self.bridges]

    def bridge_lines_4326(self):
        return [(self.to4326(s), w) for _, _, s, w in self.bridges]

    def road_lines_4326(self):
        return [(hw, w, self.to4326(l.intersection(box(*self.bm)))) for hw, w, l in self.roads]

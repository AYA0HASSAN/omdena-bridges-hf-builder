"""OAM tile geometry: slippy-map x/y/z -> EPSG:3857 bounds -> EPSG:4326.

Chips served here are PNGs named `OAM-x-y-z.{png,jpg}` (bridgebuild's `to-yolo` export), not
georeferenced GeoTIFFs, so geo bounds come from the tile coordinates in the filename rather than an
embedded affine transform (contrast with `models/yolo26s_obb_bridges/pipeline.py`, which serves
arbitrary georeferenced GeoTIFF chips for the fAIr platform and reads the transform from each file).
"""
import re

OAM_NAME_RE = re.compile(r"OAM-(\d+)-(\d+)-(\d+)")

_WGS84_TRANSFORMER = None


def parse_oam_name(name: str) -> tuple[int, int, int] | None:
    m = OAM_NAME_RE.search(name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in EPSG:3857."""
    import mercantile

    b = mercantile.xy_bounds(x, y, z)
    return b.left, b.bottom, b.right, b.top


def pixel_to_3857(
    px: float, py: float, bounds_3857: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float]:
    """Chip pixel (px, py), origin top-left -> EPSG:3857 (x, y)."""
    left, bottom, right, top = bounds_3857
    x = left + px * (right - left) / width
    y = top - py * (top - bottom) / height
    return x, y


def _wgs84_transformer():
    global _WGS84_TRANSFORMER
    if _WGS84_TRANSFORMER is None:
        from pyproj import Transformer

        _WGS84_TRANSFORMER = Transformer.from_crs(3857, 4326, always_xy=True)
    return _WGS84_TRANSFORMER


def chip_corners_to_lonlat(
    corners_px, x: int, y: int, z: int, width: int, height: int
) -> list[tuple[float, float]]:
    """Pixel corners of one `OAM-x-y-z` chip -> lon/lat (EPSG:4326)."""
    bounds = tile_bounds_3857(x, y, z)
    transformer = _wgs84_transformer()
    pts_3857 = [pixel_to_3857(px, py, bounds, width, height) for px, py in corners_px]
    return [transformer.transform(px, py) for px, py in pts_3857]

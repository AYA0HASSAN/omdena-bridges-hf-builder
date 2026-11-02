import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_COUNT = 6
CHIP_SIZE = 96
BASE_LON, BASE_LAT, STEP = 39.25, -6.81, 0.002
X_TILE_STEP = 2  # spreads chips across multiple OAM tile blocks (default block_size=4)

_WEST, _SOUTH = BASE_LON, BASE_LAT
_EAST, _NORTH = BASE_LON + CHIP_COUNT * STEP, BASE_LAT + STEP
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[_WEST, _SOUTH], [_EAST, _SOUTH], [_EAST, _NORTH], [_WEST, _NORTH], [_WEST, _SOUTH]]],
}
_BBOX = [_WEST, _SOUTH, _EAST, _NORTH]


def _rotated_square(cx: float, cy: float, half: float, degrees: float) -> list[list[float]]:
    """A square label rotated off-axis, so the OBB path (minimum_rotated_rectangle) is actually exercised."""
    angle = math.radians(degrees)
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    ring = [
        [cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle)]
        for dx, dy in corners
    ]
    return [*ring, ring[0]]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()
    labels_dir = root / "labels"
    labels_dir.mkdir()

    features: list[dict[str, Any]] = []
    for i in range(CHIP_COUNT):
        x_tile = 100 + i * X_TILE_STEP
        lon = BASE_LON + i * STEP
        lat = BASE_LAT
        name = f"OAM-{x_tile}-200-18.tif"
        transform = from_bounds(lon, lat, lon + STEP, lat + STEP, CHIP_SIZE, CHIP_SIZE)
        with rasterio.open(
            chips_dir / name,
            "w",
            driver="GTiff",
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(np.random.randint(0, 255, (3, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8))

        cx, cy = lon + STEP / 2, lat + STEP / 2
        half = STEP / 5
        features.append(
            {
                "type": "Feature",
                "properties": {"bridge": "yes"},
                "geometry": {"type": "Polygon", "coordinates": [_rotated_square(cx, cy, half, degrees=35)]},
            }
        )

    labels_path = labels_dir / "labels.geojson"
    labels_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_bridge_detection"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-bridge-detection",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "description": "Toy bridge detection dataset",
            "label:type": "vector",
            "label:tasks": ["object-detection"],
            "label:classes": [{"name": "bridge", "classes": ["bridge"]}],
            "label:description": "Detection labels",
            "keywords": ["bridge"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }

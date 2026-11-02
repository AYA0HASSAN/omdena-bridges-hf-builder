"""YOLO26s-OBB bridge detection: rotated-box object detection for bridges on OAM imagery.

Self-contained pack (no external git-installed model library): dataset prep, training, export and the
ZenML pipelines/steps live in this file, built on Ultralytics YOLO (task="obb") directly. Heavy imports
(ultralytics, rasterio, geopandas, shapely, pyproj) stay inside function bodies so other tooling can read
the dataset-prep helpers without pulling in training-only dependencies. `predict()` itself lives in
`.serving`, which this module only needs at the `run_inference` step - the inference container's
MODEL_MODULE points at `.serving` directly, never at this file, since this file needs `zenml` importable
at module load for the `@step`/`@pipeline` decorators (ticket [BRIDGES-XX], agent feedback #858).
"""

import random
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

from .obb_utils import CLASS_NAME, CLASS_VALUE, MODEL_IMGSZ
from .serving import predict

DEFAULT_BLOCK_SIZE = 4  # OAM-x-y-z tile blocks for the spatial split


def _get_device() -> str:
    """Use CUDA when the training image can see a GPU; otherwise CPU."""
    import torch

    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def _restore_checkpoint(trained_model: Any) -> Any:
    from ultralytics import YOLO

    if isinstance(trained_model, YOLO):
        return trained_model
    if isinstance(trained_model, bytes):
        checkpoint = Path(tempfile.mkdtemp()) / "best.pt"
        checkpoint.write_bytes(trained_model)
        return YOLO(str(checkpoint), task="obb")
    return YOLO(trained_model, task="obb")


# ------------------------------------------------------------------ dataset prep (GeoJSON -> YOLO-OBB)
DEFAULT_SPLIT_REF_ZOOM = 14  # ~2.4 km blocks; used when matched tiles span more than one zoom
MAX_REF_ZOOM_SPAN = 6  # cap on (split_ref_zoom - tile zoom): the block fan-out grows as 4**span
MIN_TILE_ZOOM, MAX_TILE_ZOOM = 0, 24

_OAM_XYZ_RE = re.compile(r"^OAM-([+-]?\d+)-([+-]?\d+)-([+-]?\d+)")


def _parse_oam_xyz(name: str) -> tuple[int, int, int] | None:
    """Parse the leading `OAM-x-y-z` slippy-map tile address out of a chip file name.

    Returns `None` for names that don't start with `OAM-` at all - those aren't using this convention
    and fall back to a plain random split. Raises `ValueError` (naming the file) for names that start
    with `OAM-` but have a malformed, missing or out-of-range x/y/z, since those are upstream data bugs
    rather than an intentionally different naming scheme (ticket [BRIDGES-XX], agent feedback #858 -
    mixed-zoom split leakage).
    """
    if not name.startswith("OAM-"):
        return None
    m = _OAM_XYZ_RE.match(name)
    if m is None:
        raise ValueError(f"{name}: expected OAM-x-y-z with three integers, e.g. OAM-100-200-18.tif")
    x, y, z = (int(v) for v in m.groups())
    if not (MIN_TILE_ZOOM <= z <= MAX_TILE_ZOOM):
        raise ValueError(f"{name}: zoom {z} is outside the valid range [{MIN_TILE_ZOOM}, {MAX_TILE_ZOOM}]")
    if not (0 <= x < 2**z) or not (0 <= y < 2**z):
        raise ValueError(f"{name}: tile ({x}, {y}) is out of range for zoom {z} (0 <= x, y < {2**z})")
    return x, y, z


def _ref_zoom_blocks(x: int, y: int, z: int, ref_zoom: int) -> list[tuple[int, int]]:
    """The reference-zoom (`ref_zoom`) block(s) one OAM-x-y-z tile covers.

    A tile at or finer than `ref_zoom` covers exactly one coarser block (shift right); a tile coarser
    than `ref_zoom` covers every finer block inside its own footprint (shift left, then enumerate the
    square). `MAX_REF_ZOOM_SPAN` bounds how much coarser a tile may be than `ref_zoom`, since the block
    count grows as `4 ** (ref_zoom - z)`.
    """
    if z >= ref_zoom:
        d = z - ref_zoom
        return [(x >> d, y >> d)]
    d = ref_zoom - z
    if d > MAX_REF_ZOOM_SPAN:
        raise ValueError(
            f"tile zoom {z} is {d} levels coarser than split_ref_zoom={ref_zoom} (max "
            f"{MAX_REF_ZOOM_SPAN}); raise block_size/split_ref_zoom or exclude this tile"
        )
    bx0, by0, span = x << d, y << d, 1 << d
    return [(bx0 + dx, by0 + dy) for dx in range(span) for dy in range(span)]


class _UnionFind:
    """Minimal union-find over reference-zoom block keys, used to group OAM tiles whose footprints
    overlap into one spatial component - so a tile and any of its ancestors/descendants always end up
    in the same group, and therefore the same split."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, key: tuple[int, int]) -> tuple[int, int]:
        self._parent.setdefault(key, key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if root_b < root_a:  # deterministic regardless of call order: the smaller root always wins
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a


def _resolve_ref_zoom(explicit_ref_zoom: int | None, block_size: int, zooms: set[int]) -> int:
    """Pick the reference zoom (REF_Z) spatial grouping is done at.

    An explicit `split_ref_zoom` always wins. Otherwise, if every matched tile shares one zoom z0 (the
    common case for a single dataset), `REF_Z = z0 - log2(block_size)` reproduces the legacy
    `(x // block_size, y // block_size)` grouping exactly - e.g. block_size=16 at z18 gives REF_Z=14.
    Mixed-zoom tile sets (the actual bug this guards against) have no single z0 to derive from, so they
    default to `DEFAULT_SPLIT_REF_ZOOM`.
    """
    if explicit_ref_zoom is not None:
        return explicit_ref_zoom
    if len(zooms) == 1:
        if block_size <= 0 or block_size & (block_size - 1) != 0:
            raise ValueError(f"block_size must be a positive power of 2 to derive split_ref_zoom, got {block_size}")
        (z0,) = zooms
        return z0 - (block_size.bit_length() - 1)
    return DEFAULT_SPLIT_REF_ZOOM


def _assert_no_cross_split_overlap(names: list[str], split_of_name: dict[str, str], ref_zoom: int) -> None:
    """Re-derive every matched tile's reference-zoom blocks from scratch and fail loudly if any block
    ends up claimed by more than one split - the exact failure mode of the original bug, where a parent
    and child tile of the same place could land in different splits. Independent of
    `_spatial_split_names`'s own bookkeeping, so a future refactor can't silently break the guarantee.
    """
    block_owner: dict[tuple[int, int], tuple[str, str]] = {}
    for name in names:
        xyz = _parse_oam_xyz(name)
        if xyz is None:
            continue
        x, y, z = xyz
        split = split_of_name[name]
        for block in _ref_zoom_blocks(x, y, z, ref_zoom):
            prior = block_owner.get(block)
            if prior is not None and prior[1] != split:
                raise AssertionError(
                    f"split leakage: REF_Z block {block} (zoom {ref_zoom}) is in both splits - "
                    f"'{prior[0]}' ({prior[1]}) and '{name}' ({split})"
                )
            block_owner.setdefault(block, (name, split))


def _spatial_split_names(
    chip_names: list[str],
    val_ratio: float,
    seed: int,
    block_size: int,
    ref_zoom: int | None = None,
) -> tuple[list[str], list[str], int]:
    """Split OAM-x-y-z named chips into train/val by whole spatial groups, never splitting a group
    across train and val.

    Every tile is mapped to the reference-zoom (REF_Z) block(s) it covers (`_ref_zoom_blocks`); two
    tiles whose footprints share a REF_Z block - e.g. a parent tile and any of its descendants at a
    finer zoom - are unioned into one group via `_UnionFind`, and the whole group goes to train or val
    together. The pre-fix behaviour grouped tiles by their own raw `(x // block_size, y // block_size)`,
    ignoring zoom entirely, so a parent and child tile of the same place could land in different splits
    whenever their un-scaled x/y happened to fall in different raw blocks - leaking val into train
    (ticket [BRIDGES-XX], agent feedback #858).

    Group-to-split assignment is a seeded shuffle of the groups' sorted keys (each group's own minimum
    block, so the result never depends on input order); this exactly mirrors the legacy per-block
    shuffle when REF_Z is derived to match `block_size` for a single-zoom dataset (`_resolve_ref_zoom`).

    Names that don't start with `OAM-` fall back to their own seeded random split, as before. Returns
    `(train_names, val_names, resolved_ref_zoom)` so callers can log/reuse the zoom actually used.
    """
    matched: dict[str, tuple[int, int, int]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        xyz = _parse_oam_xyz(name)
        if xyz is None:
            unmatched.append(name)
        else:
            matched[name] = xyz

    train_names: list[str] = []
    val_names: list[str] = []
    resolved_ref_zoom = ref_zoom if ref_zoom is not None else DEFAULT_SPLIT_REF_ZOOM

    if matched:
        resolved_ref_zoom = _resolve_ref_zoom(ref_zoom, block_size, {z for _, _, z in matched.values()})

        uf = _UnionFind()
        name_blocks: dict[str, list[tuple[int, int]]] = {}
        for name, (x, y, z) in matched.items():
            blocks = _ref_zoom_blocks(x, y, z, resolved_ref_zoom)
            name_blocks[name] = blocks
            first = blocks[0]
            uf.find(first)
            for block in blocks[1:]:
                uf.union(first, block)

        groups: dict[tuple[int, int], list[str]] = {}
        for name, blocks in name_blocks.items():
            groups.setdefault(uf.find(blocks[0]), []).append(name)

        group_keys = sorted(groups)
        shuffled_groups = list(group_keys)
        random.Random(seed).shuffle(shuffled_groups)
        val_group_count = max(1, round(len(shuffled_groups) * val_ratio)) if shuffled_groups and val_ratio > 0 else 0
        val_groups = set(shuffled_groups[:val_group_count])

        train_names = [n for key, names_ in groups.items() if key not in val_groups for n in names_]
        val_names = [n for key, names_ in groups.items() if key in val_groups for n in names_]

        split_of_name = {n: "train" for n in train_names} | {n: "val" for n in val_names}
        _assert_no_cross_split_overlap(list(matched), split_of_name, resolved_ref_zoom)

    shuffled_unmatched = sorted(unmatched)
    random.Random(seed).shuffle(shuffled_unmatched)
    val_unmatched_count = round(len(shuffled_unmatched) * val_ratio)
    val_names = val_names + shuffled_unmatched[:val_unmatched_count]
    train_names = train_names + shuffled_unmatched[val_unmatched_count:]
    return sorted(train_names), sorted(val_names), resolved_ref_zoom


def _obb_label_lines(chip_path: Path, labels_gdf: Any) -> list[str]:
    """Label polygons intersecting this chip -> YOLO-OBB lines `0 x1 y1 x2 y2 x3 y3 x4 y4` (normalised),
    via each clipped polygon's minimum rotated rectangle - same idea as bridgebuild/geo.py's
    obb_lines_for_tile, generalised to any chip CRS via the chip's own affine transform."""
    import rasterio
    from shapely.geometry import box as shapely_box

    with rasterio.open(chip_path) as src:
        transform, width, height, crs = src.transform, src.width, src.height, src.crs

    local = labels_gdf.to_crs(crs) if labels_gdf.crs != crs else labels_gdf
    chip_box = shapely_box(*rasterio.transform.array_bounds(height, width, transform))
    inv = ~transform

    lines = []
    for geom in local.geometry:
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(chip_box)
        if clipped.is_empty:
            continue
        polys = list(clipped.geoms) if clipped.geom_type.startswith("Multi") else [clipped]
        for poly in polys:
            if poly.area <= 0:
                continue
            rect = poly.minimum_rotated_rectangle
            if rect.geom_type != "Polygon":  # a degenerate (line-like) clip
                continue
            corners = list(rect.exterior.coords)[:4]
            px = [inv * (x, y) for x, y in corners]
            norm = [(min(1.0, max(0.0, px_ / width)), min(1.0, max(0.0, py_ / height))) for px_, py_ in px]
            flat = " ".join(f"{v:.6f}" for pt in norm for v in pt)
            lines.append(f"{CLASS_VALUE} {flat}")
    return lines


def _chip_to_png(chip_path: Path, out_path: Path) -> None:
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(chip_path) as src:
        arr = src.read([1, 2, 3])
    rgb = np.ascontiguousarray(np.transpose(arr, (1, 2, 0))).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def _prepare_yolo_obb_dataset(
    chips_dir: str | Path,
    labels_dir: str | Path,
    val_ratio: float,
    seed: int,
    block_size: int,
    ref_zoom: int | None = None,
    out_dir: str | Path | None = None,
) -> tuple[Path, dict[str, dict[str, int]], int]:
    import geopandas as gpd
    import pandas as pd

    chip_paths = sorted(p for pat in ("*.tif", "*.tiff") for p in Path(chips_dir).glob(pat))
    if not chip_paths:
        raise FileNotFoundError(f"No georeferenced (.tif/.tiff) chips found in {chips_dir}")

    label_files = sorted(Path(labels_dir).glob("*.geojson"))
    if not label_files:
        raise FileNotFoundError(f"No .geojson label files found in {labels_dir}")
    labels_gdf = (
        gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in label_files], ignore_index=True))
        if len(label_files) > 1
        else gpd.read_file(label_files[0])
    )
    if labels_gdf.crs is None:
        labels_gdf = labels_gdf.set_crs("EPSG:4326")

    by_name = {p.name: p for p in chip_paths}
    train_names, val_names, resolved_ref_zoom = _spatial_split_names(
        list(by_name), val_ratio, seed, block_size, ref_zoom
    )

    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="yolo_obb_dataset_"))
    counts: dict[str, dict[str, int]] = {}
    for split, split_names in (("train", train_names), ("val", val_names)):
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        positive = 0
        for name in split_names:
            chip_path = by_name[name]
            lines = _obb_label_lines(chip_path, labels_gdf)
            stem = Path(name).stem
            _chip_to_png(chip_path, img_dir / f"{stem}.png")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            positive += bool(lines)
        counts[split] = {"chips": len(split_names), "positive": positive}

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n"
    )
    return out, counts, resolved_ref_zoom


def _resolve_yolo_dir_for_step(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> Path:
    """Resolve the generated dataset, rebuilding it if this step runs in a fresh container."""
    yolo_dir = Path(split_info["_yolo_dir"])
    if (yolo_dir / "data.yaml").exists():
        return yolo_dir

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    labels_dir = resolve_directory(dataset_labels, "*.geojson")
    yolo_dir, _, _ = _prepare_yolo_obb_dataset(
        chips_dir,
        labels_dir,
        split_info["val_ratio"],
        split_info["seed"],
        split_info.get("block_size", DEFAULT_BLOCK_SIZE),
        split_info.get("split_ref_zoom"),
        out_dir=yolo_dir,
    )
    return yolo_dir


# ------------------------------------------------------------------ ZenML steps + pipelines
@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", DEFAULT_BLOCK_SIZE)
    ref_zoom = hyperparameters.get("split_ref_zoom")

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    labels_dir = resolve_directory(dataset_labels, "*.geojson")
    yolo_dir, counts, resolved_ref_zoom = _prepare_yolo_obb_dataset(
        chips_dir, labels_dir, val_ratio, seed, block_size, ref_zoom
    )

    # Independent re-check against the actual files written to disk (not _spatial_split_names's own
    # bookkeeping), so a future refactor of the dataset-prep path can't silently reintroduce split
    # leakage without CI catching it (ticket [BRIDGES-XX], agent feedback #858).
    split_of_name = {
        img.name: split for split in ("train", "val") for img in (yolo_dir / "images" / split).glob("*.png")
    }
    _assert_no_cross_split_overlap(list(split_of_name), split_of_name, resolved_ref_zoom)

    split_info = {
        "strategy": "spatial_block_oam_xyz",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "split_ref_zoom": resolved_ref_zoom,
        "train_count": counts["train"]["chips"],
        "val_count": counts["val"]["chips"],
        "train_positive": counts["train"]["positive"],
        "val_positive": counts["val"]["positive"],
        "description": (
            "Chips named OAM-x-y-z are grouped by the reference-zoom (split_ref_zoom) block(s) they "
            "cover; a tile and any ancestor/descendant tile of the same place share a block and so "
            "always land in the same split. Whole groups go to train or val by a seeded shuffle. Names "
            "that don't start with OAM- fall back to a seeded random split."
        ),
        "_yolo_dir": str(yolo_dir),
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 1,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    del num_classes
    from ultralytics import YOLO

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 16)
    imgsz = hyperparameters.get("chip_size", MODEL_IMGSZ)
    learning_rate = hyperparameters.get("learning_rate", 0.01)
    patience = hyperparameters.get("patience", 50)

    yolo_dir = _resolve_yolo_dir_for_step(dataset_chips, dataset_labels, hyperparameters, split_info)
    local_weights = _download_checkpoint(base_model_weights)
    device = _get_device()

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        model = YOLO(str(local_weights), task="obb")
        results = model.train(
            data=str(yolo_dir / "data.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            lr0=learning_rate,
            patience=patience,
            degrees=180,
            flipud=0.5,
            fliplr=0.5,
            scale=0.3,
            hsv_h=0.02,
            hsv_v=0.3,
            verbose=False,
        )
        if results and hasattr(results, "results_dict"):
            log_metadata(metadata={k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))})

    best_path = getattr(getattr(model, "trainer", None), "best", None)
    if best_path and Path(best_path).exists():
        return Path(best_path).read_bytes()
    saved_path = Path(tempfile.mkdtemp()) / "best.pt"
    model.save(str(saved_path))
    return saved_path.read_bytes()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    del class_names
    imgsz = hyperparameters.get("chip_size", MODEL_IMGSZ)
    yolo_dir = _resolve_yolo_dir_for_step(dataset_chips, dataset_labels, hyperparameters, split_info)

    model = _restore_checkpoint(trained_model)
    results = model.val(data=str(yolo_dir / "data.yaml"), imgsz=imgsz, split="val", verbose=False)
    if not hasattr(results, "results_dict") or not results.results_dict:
        raise RuntimeError("YOLO OBB validation produced no results")

    rd = results.results_dict
    metrics = {
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
        "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
    }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx

    model = _restore_checkpoint(trained_model)
    onnx_path = model.export(format="onnx", imgsz=MODEL_IMGSZ, opset=17, simplify=True)
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)
    try:
        return Path(onnx_path).read_bytes()
    finally:
        Path(onnx_path).unlink(missing_ok=True)


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any] | None = None,
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params or {})


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )

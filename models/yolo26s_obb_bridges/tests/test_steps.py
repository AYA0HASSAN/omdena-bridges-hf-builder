"""Step tests for YOLO26s-OBB bridge detection."""

from pathlib import Path
from typing import Any

import pytest

_PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt"


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    from upath import UPath

    cache = tmp_path_factory.mktemp("yolo26s_obb_bridges_weights") / "model.pt"
    cache.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(cache)


def _fast_hyperparameters(base_hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Small chip size and batch for a quick CPU smoke run; still exercises the real train/val/export path."""
    return {**base_hyperparameters, "chip_size": 64, "batch_size": 2, "patience": 1}


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo26s_obb_bridges.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "spatial_block_oam_xyz"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    yolo_dir = Path(info["_yolo_dir"])
    assert (yolo_dir / "data.yaml").exists()
    assert (yolo_dir / "images" / "train").is_dir()
    assert (yolo_dir / "images" / "val").is_dir()
    assert any((yolo_dir / "labels" / "train").glob("*.txt"))


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo26s_obb_bridges.pipeline import split_dataset, train_model

    hyperparameters = _fast_hyperparameters(base_hyperparameters)
    dataset_chips = {"image": str(toy_chips)}
    split_info = split_dataset.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=1,
    )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo26s_obb_bridges.pipeline import evaluate_model, split_dataset, train_model

    hyperparameters = _fast_hyperparameters(base_hyperparameters)
    dataset_chips = {"image": str(toy_chips)}
    split_info = split_dataset.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=1,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
        split_info=split_info,
    )

    assert set(metrics) == {"precision", "recall", "map50", "map50_95"}


def test_export_onnx(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    import onnx

    from models.yolo26s_obb_bridges.pipeline import export_onnx, split_dataset, train_model

    hyperparameters = _fast_hyperparameters(base_hyperparameters)
    dataset_chips = {"image": str(toy_chips)}
    split_info = split_dataset.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=dataset_chips,
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=1,
    )
    onnx_bytes = export_onnx.entrypoint(trained_model=model_bytes)

    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) >= 1
    assert len(loaded.graph.output) >= 1

# yolo26s-obb-bridges

## Overview

Single-class bridge **object detection** for VHR RGB aerial imagery on the fAIr platform, using **oriented
(rotated) bounding boxes** so a bridge deck is represented by its true footprint rather than an axis-aligned
box. This is a self-contained model pack: dataset preparation, training, ONNX export and serving all live in
`pipeline.py`, with no external model library dependency.

The model returns a GeoJSON `FeatureCollection` where each detection is an EPSG:4326 `Polygon` (the four
corners of the rotated box) with a confidence score and an `angle` property.

## Architecture

- **Model**: Ultralytics YOLO26s, OBB task, single class `bridge` (`value: 0`).
- **Input**: 3-band RGB GeoTIFF chips; ONNX input size is **512×512** (declared in `stac-item.json`
  `mlm:input`). fAIr's native chip is 256×256 and is resized up in `preprocess`/`predict`.
- **Output**: GeoJSON rotated-rectangle polygons, with per-detection confidence and angle.
- **Serving**: ONNX Runtime session (CPU only); pre/postprocessing, rotated NMS and georeferencing are
  implemented directly in `pipeline.py` with numpy/rasterio/pyproj/shapely (no torch, no Ultralytics in the
  inference image).
- **Accelerator**: the training/runtime image ships CUDA Torch and uses a GPU when one is visible, otherwise
  CPU. Live inference stays on CPU ONNX Runtime.
- **Dataset prep**: chips named `OAM-x-y-z` are split into train/val by whole `(x // block_size, y // block_size)`
  tile blocks (a seeded shuffle); other filenames fall back to a seeded random split. Label polygons are
  clipped to each chip and converted to YOLO-OBB lines via their minimum rotated rectangle.

## Pretrained source

Pretrained weights come from Ultralytics' DOTA-pretrained checkpoint:
[yolo26s-obb.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt)
([Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)).

The finetuning dataset this model was developed against is OAM imagery + OpenStreetMap bridge tags across
several Global South cities (Dar es Salaam, Freetown, Manila, Beira, Port-au-Prince, Monrovia, Blantyre),
plus a France BD ORTHO/BD TOPO set used for domain adaptation, built by the `bridges-hf-builder` pipeline
(`experiments/bridges-hf-builder/` in `OmdenaAI/HOTOSM-fAIr-Bridges`).

## Limitations

- **Unverified ONNX output layout.** `postprocess` assumes the standard Ultralytics OBB export: one tensor
  shaped `(1, 6, N)` for this one-class model (box already decoded to real pixel coordinates, class score
  post-sigmoid, angle in radians). This has **not** been confirmed against a real exported `yolo26s-obb`
  ONNX graph. Before trusting production predictions, export a trained model, inspect
  `session.get_outputs()[0].shape`, and adjust `postprocess`/`predict` if the layout differs (angle
  convention, channel order, or a built-in NMS head).
- **Severe class imbalance against France.** The France domain-adaptation chips substantially outnumber OAM
  chips in early builds of the dataset; recall on OAM/target-domain imagery was measured far below France
  recall in early training runs. Oversampling OAM chips in training and growing OAM coverage (more
  `discover`-verified cities) both help; see the dataset repo's own notes for the current ratio.
- **Small-bridge recall.** Detection recall drops sharply for bridges under ~16-32 px at the trained input
  resolution; this is a data-volume problem more than a model-capacity one.
- **Label noise.** OSM bridge tags include culverts, footbridges and slivers that are not reliably visible
  in VHR RGB imagery; the upstream dataset pipeline flags and can exclude these, but any upstream label noise
  that was not excluded will degrade reported metrics.
- **IoU 0.5 vs 0.3 gap.** Earlier evaluation runs showed detection boxes often land close to but not tightly
  on the true bridge footprint (an OSM geometry offset, not purely a detection failure) - recall roughly
  doubles between IoU 0.5 and IoU 0.3 matching.

## Usage

- **Live inference**: run the inference container and call `POST /predict` with a `model_uri` (ONNX), an
  `image_uri`/`imagery` TMS source, a bbox, a zoom, and inference params (`confidence_threshold`,
  `iou_threshold`). Defaults and preview fields are defined in `stac-item.json`.
- **Fine-tuning**: run `training_pipeline` with a dataset that provides `chips` (RGB GeoTIFF directory, named
  `OAM-x-y-z` where possible) and `labels` (directory containing one or more GeoJSON files of bridge deck
  polygons). The pipeline trains on the train split only, evaluates on the val split only, and exports ONNX.
- **Contributor workflow**: follow `docs/getting-started.md` and `docs/contributing/model.md`
  (validate → build → example run → serve smoke).

## Citation

- Architecture: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).
- Dataset pipeline: `OmdenaAI/HOTOSM-fAIr-Bridges`, `experiments/bridges-hf-builder/`.
- Labels: © OpenStreetMap contributors (ODbL 1.0); IGN BD TOPO (Etalab 2.0) for the France domain-adaptation
  set.

## License

AGPL-3.0-only (this model pack and the Ultralytics YOLO26 architecture; must match `properties.license` in
`stac-item.json`). OpenStreetMap-derived labels are under ODbL 1.0; France labels are under Etalab 2.0.

# 08 — Baseline Model, Serving Path and ONNX: How to Finish the Ticket

*Ticket: "As the Bridges engineer, I have base model code in the Bridges repo that goes from OAM chips to GeoJSON, with an ONNX export and a recorded baseline score."*

---

## In short

- **Most of the hard work is done.** The model is chosen (YOLO26s-OBB, doc 04/05). It is trained, and a first baseline is measured on a leak-free spatial split. The run is `exp2_clean_512` in `train_experiments.ipynb`, and the weights are on Hugging Face.
- **What's left is code and paperwork:**
  1. the serving path (chips → predictions → GeoJSON);
  2. ONNX export and parity tests;
  3. an evaluation harness for the validation split;
  4. CI;
  5. the baseline note.
- **Yes, upload the training notebook.** It goes in `experiments/` and is the "baseline evaluation notebook" the ticket asks for.
- **Baseline to record:** `exp2_clean_512 / stage2`, at 512 px with confidence threshold 0.10. On clean OAM test it scores **F1 0.22 at IoU 0.5** and **0.41 at IoU 0.3**, with **mAP50 0.23**.

---

## 1. What the training run tells us

### 1.1 Results

The confidence threshold for each model was chosen on OAM validation and then applied to test.

| Model | OAM val F1 @0.5 | OAM val mAP50 | France val mAP50 | **Clean test F1 @0.5** (P / R) | **Clean test F1 @0.3** (P / R) | Clean test mAP50 | Raw test mAP50 | Threshold |
|---|---|---|---|---|---|---|---|---|
| baseline_256 (first-run settings) | 0.169 | 0.194 | **0.626** | 0.109 (0.23 / 0.07) | 0.341 (0.73 / 0.22) | **0.247** | 0.153 | 0.25 |
| stage1 (512 px, clean, OAM ×3) | 0.162 | 0.218 | 0.577 | 0.148 (0.28 / 0.10) | 0.370 (0.69 / 0.25) | 0.242 | 0.149 | 0.25 |
| **stage2 (+ OAM fine-tune)** | **0.191** | 0.206 | 0.565 | **0.217** (0.25 / 0.19) | **0.411** (0.47 / 0.36) | 0.231 | **0.157** | 0.10 |

**About the data behind these numbers:**
- Test is Dar es Salaam and Freetown: 148 clean chips with 99 bridges. The raw test set has 175 bridges.
- OAM validation is small: 156 chips with 59 bridges.
- All 5 leakage tests passed.

### 1.2 What it means

1. **Stage 2 is the best model to ship as the baseline.** It has the best OAM F1 at both IoU levels.
2. **Most of that gain comes from the threshold, not a better model.** mAP50, which doesn't depend on the threshold, is flat at 0.23–0.25 for all three models. Stage 2 mostly gains recall, from 0.07 to 0.19, at a lower threshold. With 59 validation and 99 test bridges, differences under about ±0.03 are noise.
3. **Location is a big part of the error.** F1 roughly doubles from IoU 0.5 to IoU 0.3 (0.22 → 0.41). The model finds bridges but its boxes are shifted or the wrong shape against OSM. That fits the OSM offset we measured earlier.
4. **The clean-up was justified.** At IoU 0.3, recall is 0.33 on clean labels, 0.11 on culverts, and 0 on slivers, tiny boxes and footbridges. In the rule-check picture, the "footbridges" in Monrovia are long boxes over flooded houses, which are bad OSM data.
5. **Small bridges are the weak spot.** Recall is 0.05 for boxes under 16 px and 0.16 for 16–32 px, against 0.41 for 32–64 px.
6. **The real bottleneck is data.** The OAM-only training set has **98 clean bridge chips**. France performs well (mAP50 0.57–0.63) with about 550, more than five times as many. The OAM fine-tune costs some French accuracy (0.63 → 0.57), which is fine for a model aimed at OAM.

> **Before committing the notebook**, replace section 10 ("Reading the results"). I wrote those lines before the run, as expectations, and some don't match the results: for example, mAP50 did *not* improve. Paste §1.2 in its place.

---

## 2. Do I need to upload the training notebook?

**Yes.** The ticket asks for "the baseline evaluation notebook in the `experiments/` folder".

| What | Where in the repo | Note |
|---|---|---|
| `train_experiments.ipynb` (with its outputs) | `experiments/baseline/exp2_clean_512.ipynb` | Outputs are the evidence, so keep them. It is about 8 MB; if you want it smaller, clear only the long training-log cells. There's no secret in it, because the token comes from Kaggle secrets |
| `exp2_clean_512_metrics.csv`, `exp2_clean_512_settings.json`, `exp2_clean_512_test_predictions.png` | `experiments/baseline/` | Download them from the HF model repo folder `exp2_clean_512/` |
| `experiment.py`, `finalize.py` | already in `experiments/bridges-hf-builder/bridgebuild/` | – |
| Weights (`best.pt`, `.onnx`) | **not in git.** Keep them in HF `AyaHassan/fair-bridges-yolo` | Large files don't belong in git |

---

## 3. What to build, in the `pipeline/` folder

```
pipeline/
  __init__.py
  detector.py        # model-agnostic interface + backends (ONNX, Ultralytics)
  postprocess.py     # decode raw output, rotated NMS, px -> polygons
  geo.py             # OAM-x-y-z -> tile bounds, px -> EPSG:3857 -> EPSG:4326
  serve.py           # chips folder -> predictions -> GeoJSON  (CLI)
  export_onnx.py     # .pt -> .onnx + metadata  (CLI)
  evaluate.py        # evaluation harness on a split  (CLI)
  metrics.py         # rotated IoU, matching, P/R/F1 sweep (moved from bridgebuild/experiment.py)
  requirements.txt   # numpy, shapely, mercantile, pillow, onnxruntime  (no torch, no ultralytics)
  requirements-train.txt  # + ultralytics, torch, onnx  (export and tests only)
  tests/
    test_geo.py
    test_postprocess.py
    test_geojson.py
    test_onnx_parity.py
.github/workflows/pipeline-ci.yml
```

### 3.1 Keep it model-agnostic (the AGPL edge case)

Ultralytics YOLO is **AGPL-3.0**, and HOT hasn't answered the licence question yet. So:

- **Serving runs on ONNX Runtime only.** `pipeline/requirements.txt` has no `ultralytics` and no `torch`. Pre-processing, decoding and NMS are our own code.
- `detector.py` defines one small interface. Every model, now or later (RTMDet-R, a non-YOLO swap), plugs in behind it:

```python
class Detector(Protocol):
    imgsz: int
    def predict(self, chips: np.ndarray) -> list["Detections"]: ...
    # chips: N x 256 x 256 x 3 uint8 (RGB)
    # Detections: corners (K x 4 x 2, px in the 256 chip), scores (K,)
```

- Backends:
  - `OnnxDetector`: onnxruntime, CPU by default.
  - `UltralyticsDetector`: only imported inside a function, used for training checks and the parity test.
- If the AGPL answer is "no", only `export_onnx.py` and the training notebook touch Ultralytics. You swap the model, and the serving path stays the same.

### 3.2 Serving path: chips → predictions → GeoJSON (`serve.py`)

```bash
python -m pipeline.serve --model bridges-v0.onnx --chips /path/chips --out bridges.geojson --conf 0.10 --batch 16
```

1. **Read chips** named `OAM-x-y-z.png` (or `.tif`), in batches of 16 so RAM stays flat.
2. **Pre-process:** RGB, resize 256 → 512 (bilinear), scale to 0–1, NCHW float32. Training used 512 px input, so serving must too.
3. **Run the detector.**
4. **Post-process** (`postprocess.py`):
   - The exported model gives **raw** output that still needs NMS. The DOTA YOLO26 OBB export is `(1, 20, N)`: 4 box + 15 classes + 1 angle. **For our 1-class model at 512 px, expect `(1, 6, 5376)`, holding `cx, cy, w, h, score, angle (radians)` in 512-px coordinates.** Check this first with `session.get_outputs()[0].shape`, and adjust the decoder if the export comes out different.
   - Keep `score ≥ conf`, convert `(cx, cy, w, h, angle)` to 4 corners, run **rotated NMS** (IoU 0.7, shapely polygons are fine at this size), and scale corners by 256/512.
5. **Georeference** (`geo.py`):
   - `x, y, z` come from the file name.
   - `mercantile.xy_bounds(x, y, z)` gives the tile bounds in EPSG:3857.
   - Pixel to metres is `X = left + px·(right−left)/256` and `Y = top − py·(top−bottom)/256`. Transform to EPSG:4326, because GeoJSON must be lon/lat (RFC 7946).
6. **Write GeoJSON:** one rotated-rectangle **Polygon** per bridge, with these properties:
   - `confidence`, `length_m`, `width_m`, `bearing_deg` (required by doc 04 §7, computed in metres from the 3857 polygon and scaled by cos(lat));
   - `chip_id`, `model`, `model_version`.

   The `FeatureCollection` also records the model name and threshold.
7. **Optional, behind a flag:** `--merge-across-chips` joins pieces of one bridge cut by a chip edge. They must touch or overlap, with a direction difference under 15°.

The fAIr contract (doc 04) asks for **polygon** output with the `object-detection` keyword, and the rotated rectangle is that polygon.

### 3.3 ONNX export (`export_onnx.py`)

```bash
python -m pipeline.export_onnx --weights best.pt --imgsz 512 --opset 17 --out bridges-v0.onnx
```

- Export with Ultralytics (`model.export(format="onnx", imgsz=512, opset=17, simplify=True, dynamic=True)`). A dynamic batch size lets serving batch chips.
- Write metadata into the ONNX file: `imgsz`, `chip_size=256`, `conf_default=0.10`, `classes=["bridge"]`, `train_run=exp2_clean_512/stage2`, `split=splits/exp2_clean_512.parquet` and the weights sha256.
- Upload `bridges-v0.onnx` to `AyaHassan/fair-bridges-yolo/v0/` next to `best.pt`.

### 3.4 Parity tests: does ONNX match PyTorch? (`tests/test_onnx_parity.py`)

Check at three levels; all must pass:

| Level | What is compared | Pass if |
|---|---|---|
| 1 Raw tensor | `model.model(x)` in PyTorch vs `session.run` in ONNX, for the same input | max abs diff < 1e-3 (fp32) |
| 2 Detections | Our post-processing on both raw outputs, for 20 chips | Same number of detections; each matched pair has rotated IoU ≥ 0.99 and score diff < 1e-3 |
| 3 GeoJSON | `serve.py` with each backend | Same feature count; matched polygons IoU ≥ 0.99 |

**In CI there are no secrets and no big weights.** The test builds a tiny random-weight model from config (`YOLO("yolo26n-obb.yaml")`, which downloads nothing) with `nc=1`, exports it and compares. Parity is about the export, not about accuracy, so random weights are a fair test.

Add a second test, `@pytest.mark.slow`, that runs only when `HF_TOKEN` is set. It downloads the real `best.pt` and 20 real validation chips and checks levels 1–3.

### 3.5 Evaluation harness on the validation split (`evaluate.py`)

```bash
python -m pipeline.evaluate --model bridges-v0.onnx --split-file exp2_clean_512_split.parquet --set val --chips <pool>/images --labels <pool>/labels
```

- Move the rotated-IoU matching, the P/R/F1 sweep and the recall-by-category code from `bridgebuild/experiment.py` into `pipeline/metrics.py`. The notebook and the harness then share one implementation.
- It reads the split saved in the dataset repo (`splits/exp2_clean_512.parquet`), so the same chips are used every time.
- It runs through the **Detector interface**, so the same command scores the ONNX model and the PyTorch model. The two scores should match, which is a fourth parity check.
- Output:
  - `metrics.json`: P/R/F1 at IoU 0.3 and 0.5, the chosen threshold, number of boxes;
  - a confidence sweep CSV;
  - recall by size and category.
- Default: CPU, batch 16, so it runs on a laptop or a free Kaggle CPU session.

### 3.6 No Colab Pro: no GPU assumptions

- Everything in `pipeline/` defaults to CPU (`onnxruntime` `CPUExecutionProvider`). A GPU is used only with `--device cuda`.
- Don't call `.cuda()`, assume AMP or hard-code a batch size for GPU memory anywhere in `pipeline/`.
- Training stays in the notebook on free Kaggle GPU (the ticket's "run small"). The validation set is 156 chips, so it takes about a minute on CPU.

### 3.7 CI (`.github/workflows/pipeline-ci.yml`)

```yaml
name: pipeline-ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r pipeline/requirements.txt -r pipeline/requirements-train.txt pytest
      - run: pytest pipeline/tests -m "not slow" -q
```

Install the CPU-only torch wheel in CI to keep it fast:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**What the tests cover:**
- `test_geo`: a known tile maps to known lon/lat corners, and pixel (0, 0) is the tile's top-left.
- `test_postprocess`: a synthetic raw output gives the expected boxes after NMS, and the angle convention round-trips.
- `test_geojson`: valid GeoJSON, polygons closed, lon/lat order, required properties present.
- `test_onnx_parity`: levels 1–3 on the tiny model.

---

## 4. Step-by-step plan

| # | Step | Where | Done when |
|---|---|---|---|
| 1 | Commit the training notebook (section 10 fixed) and its result files | `experiments/baseline/` | The notebook opens on GitHub with its outputs |
| 2 | `geo.py`, `postprocess.py`, `detector.py` plus their unit tests | `pipeline/` | `pytest` passes locally on CPU |
| 3 | `export_onnx.py`; export `stage2/best.pt` to `bridges-v0.onnx` and upload it to HF `v0/` | `pipeline/` + HF | The ONNX output shape is confirmed and noted in the README |
| 4 | `test_onnx_parity.py`, tiny model plus the slow real-weights test | `pipeline/tests/` | All 3 levels pass, including the slow test once by hand |
| 5 | `serve.py`: run it on the 148 clean test chips and open the GeoJSON in QGIS on top of OAM | `pipeline/` | Polygons sit on the bridges in QGIS, with all properties present |
| 6 | `metrics.py` + `evaluate.py`: score ONNX and PyTorch on `val` | `pipeline/` | Both backends give the same numbers, and they match §1.1 (stage2 val F1 0.19) |
| 7 | CI workflow | `.github/workflows/` | Green on the PR |
| 8 | Baseline note: upload `09-baseline-note.md` as a Google Doc in the project Drive and put the link on the ticket | Drive + ticket | The link is on the ticket |
| 9 | Close: tick the acceptance criteria below | ticket | – |

**Commits:** put the ticket ID in every commit message, for example `[BRIDGES-XX] pipeline: ONNX export + parity tests`. Small commits, one per step.

---

## 5. Acceptance checklist

| Criterion | How it is met |
|---|---|
| CI tests pass | `pipeline-ci` is green (geo, post-processing, GeoJSON, parity) |
| ONNX matches PyTorch | Parity levels 1–3 pass, and `evaluate.py` gives the same metrics for both backends |
| Serving path, ONNX export and parity tests in `pipeline/` | §3 layout |
| Baseline evaluation notebook in `experiments/` | `experiments/baseline/exp2_clean_512.ipynb` |
| Every commit linked to the ticket | Ticket ID in each commit message |
| Short baseline note as a Google Doc, linked on the ticket | `09-baseline-note.md` → Drive |
| ⚑ AGPL unanswered → model-agnostic | Serving uses ONNX Runtime only; Ultralytics appears only in export and tests, behind the `Detector` interface |
| ⚑ No Colab Pro → run small, no GPU assumptions | CPU defaults everywhere in `pipeline/`; training on free Kaggle GPU |

---

## 6. After the ticket

1. **More OAM data.** 98 clean OAM training chips is the main limit (§1.2.6). Run `discover` and `build` for more cities, then re-run the training notebook with a new `EXP`; it only downloads new chips.
2. **Offset correction.** The IoU 0.3 / 0.5 gap says it's worth doing next.
3. **Bigger test set.** Add 1–2 more OAM test cities, so that a 0.03 difference means something.

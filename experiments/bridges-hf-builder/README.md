# bridges-hf-builder

Builds the fAIr Bridges training dataset and stores it on the Hugging Face Hub, ready for training and analysis.

- **Labels:** OSM bridges (`bridge=*` ways + `man_made=bridge` outlines) and, for France, IGN BD TOPO bridges.
- **Imagery:** OpenAerialMap (open licences only; non-commercial is refused) and IGN BD ORTHO.
- **Chips:** 256 × 256 px, Web Mercator zoom 18, named `OAM-x-y-z`, cut the way fAIr cuts chips.
- **Labels written:** YOLO-OBB rotated boxes, with the source geometry kept per bridge.

Only places with bridges are cut. Negatives come in a set ratio and include tricky ones (water, roads).

Put this folder in the repo as `experiments/bridges-hf-builder/` (ticket requirement). It holds code only: `.gitignore` blocks imagery and Parquet files.

## One command per step

```bash
pip install -r requirements.txt
export HF_TOKEN=...                  # write token; on Kaggle/Colab use the secrets store (see notebook)
export BB_REPO_ID=<user-or-org>/fair-bridges-chips

python -m bridgebuild discover --bbox 85.20 27.60 85.45 27.80     # which OAM images have bridges
python -m bridgebuild all      --config configs/phase1_oam_france.yaml   # build -> finalize (split + leakage tests) -> stats
python -m bridgebuild to-yolo  --config configs/phase1_oam_france.yaml --out yolo_bridges
```

`all` is the "one command builds chips, labels and the split, and the leakage test passes" acceptance criterion. It exits with code 2 if a leakage test fails.

To try everything without the Hub, add `--local-dir some_folder`.

Running on Kaggle or Colab: `notebooks/run_bridgebuild.ipynb`.

## What ends up on the Hub (private dataset repo)

| Path | Content |
|---|---|
| `data/shard-*.parquet` | One row per chip. `image` is the PNG (HF Image feature). Other fields: `obb_yolo`, `boxes_px`, `bridge_ids`, tile x/y/z, bounds, image URL/provider/licence, `kind` (positive / hard_negative / random_negative) |
| `index/chips-*.parquet` | The same rows without images, for fast analysis |
| `index/bridges-*.parquet` | One row per bridge: `source_wkt` (geometry as delivered), `deck_wkt`, length, width, bearing, crossing, tags |
| `splits/split.parquet` | chip → train / val / test / buffer / drop, with the reason |
| `reports/` | Leakage tests, statistics tables, figures, `overlay_samples.png` |
| `MANIFEST.csv` | sha256, size and rows of every file |
| `dataset_record.md` | Sources, licences, counts, split, leakage tests, every image used |
| `README.md` | Dataset card (licence ODbL, attribution, how to load) |
| `state/state.json` | Build progress (finished and failed cells): this is what makes resuming work |

## Why it does not run out of RAM

1. A region is cut into cells of about 4 km (`cell_deg: 0.04`), and only one cell is in memory at a time.
2. Imagery is read one 256 px window at a time from the COG or WMS. A full image is never downloaded.
3. Chips are kept as PNG bytes and written to a Parquet shard every `shard_size` (500) chips, about 40–80 MB.
4. Each shard is uploaded in one commit together with its index and the state file, then deleted locally.
5. If the session dies, run the same command again: finished cells are skipped. `--max-hours 11` stops cleanly before Kaggle's 12-hour limit.
6. `finalize` and `stats` read only the small index tables. `to-yolo` streams one shard at a time in 64-row batches.

The offline test showed RAM stayed at about 200–300 MB.

## The split (finalize)

- **Test:** whole regions from `split.test_regions` (OAM).
- **Train and validation:** zoom-14 blocks (~2.4 km), chosen by a stable hash.
- **Grouping:** every chip of a bridge follows the bridge's home block.
- **Buffer:** chips within 1 chip of another split are dropped.
- **Near-duplicates:** pHash near-duplicates across splits are dropped.

Five leakage tests run:

1. Each chip is in exactly one split.
2. Each bridge is in exactly one split.
3. No chips from different splits sit within the buffer.
4. Test contains only the test regions.
5. Test images are never used in train or validation.

## Tests

```bash
python tests/test_units.py              # licence filter, OAM metadata parsing
python tests/test_offline_pipeline.py   # fake source -> build (stop + resume, failing cells) -> finalize -> stats -> to-yolo
```

The offline test also checks that the leakage tests catch a split that is deliberately made leaky.

## Adding a source later (phase 2: Netherlands, Switzerland, NAIP, Maxar)

1. Write a function `(region, cell_id, bbox, opts) -> CellData` in `sources.py`.
2. Register it in `SOURCES`.
3. Add a block under `sources:` in the config.

Re-running `build` only processes the new cells, and `finalize` re-splits everything. Keep the test regions fixed.

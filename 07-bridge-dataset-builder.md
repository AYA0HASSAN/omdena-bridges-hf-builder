# 07 — Bridge Dataset Builder

*How we turn OSM and BD TOPO bridges plus open imagery into a training dataset on Hugging Face, and how to run it on Kaggle or Colab.*

---

## In short

- **What it does.** The builder finds bridges and cuts 256 × 256 chips over them, the same way fAIr cuts chips. It labels each bridge with a rotated box, splits the chips into train, validation and test without leakage, and stores everything on Hugging Face.
- **Where it runs.**
  - Building: Kaggle CPU. You get 32 GB RAM and 12-hour sessions.
  - Training: Kaggle GPU.
  - Colab's free tier also works.
- **Why it won't crash.**
  - It works on one small area at a time and reads one chip at a time.
  - It saves every 500 chips and uploads them.
  - If a session dies, you run the same command again and it continues.
- **One command.** `python -m bridgebuild all` builds the chips, labels and split, and runs the leakage tests. It stops with an error if a test fails.
- **Code lives in** `experiments/bridges-hf-builder/`. Imagery and data never go into git.

---

## 1. What problem it solves

The ticket asks for bridge training data built from OSM bridges over openly licensed imagery, cut the way fAIr cuts chips. Doing that by hand, or in one notebook, breaks in three places:

1. **Memory.** A whole OAM image or a whole French department doesn't fit in Colab's ~12 GB.
2. **Sessions.** Kaggle stops after 12 hours and Colab disconnects sooner. Losing hours of work is not acceptable.
3. **Wasted effort.** Most of any image has no bridges. Chipping it all and then throwing most of it away is slow.

The builder handles each one:

1. It works in small areas, one at a time.
2. It saves progress to Hugging Face after every batch.
3. It looks for bridges *first* and only touches imagery where there are some.

---

## 2. The flow

```
 discover  ──►  build  ──►  finalize  ──►  stats  ──►  to-yolo  ──►  train
 (where?)      (chips)     (split +       (analysis    (YOLO-OBB     (YOLO26s-OBB)
                            leakage        + overlay    folder)
                            tests)         samples)
```

| Step | What happens | Reads | Writes to Hugging Face |
|---|---|---|---|
| **discover** | Lists open-licence OAM images in a search box and counts the OSM bridges inside each one | OAM API, Overpass (`out count`) | – (local CSV) |
| **build** | Cuts chips and boxes, area by area. Resumable | OSM / BD TOPO, OAM / BD ORTHO | `data/`, `index/`, `state/` |
| **finalize** | Removes duplicates, assigns the split, runs leakage tests, writes the manifest, dataset card and record | `index/` only | `splits/`, `reports/leakage_tests.csv`, `MANIFEST.csv`, `README.md`, `dataset_record.md` |
| **stats** | Counts, licences, bridge sizes and directions, box sizes, overlay samples | `index/` + 1–3 shards | `reports/` |
| **to-yolo** | Writes `images/`, `labels/` and `data.yaml` for training, one shard at a time | `data/`, `splits/` | – (local folder) |

---

## 3. How a region becomes chips

### 3.1 Regions and cells

You list **regions** (bounding boxes) in the config. Each region is cut into a grid of **cells** of 0.04° (about 4.4 km). A cell is the unit of work: it is fetched, chipped, saved and marked done. Cell names never change (`kathmandu_003_001`), which is what makes resuming possible.

### 3.2 Bridges first

For each cell:

1. **Get the bridges.**
   - **OAM + OSM:** an Overpass query returns bridge ways, `man_made=bridge` outlines, water and roads. Bridge lines are filtered as before: they must cross water or a road, be at least 6 m long, and not be culverts or boardwalks. Each kept line is widened into a deck using its width tag or a default for the road class.
   - **France:** BD TOPO bridge surfaces come from the WFS. Bridge lines get a 6 m deck (3 m for footbridges) when no surface exists.
2. **No bridges → stop.** The cell is marked done and no imagery is requested.
3. **Pick the imagery.**
   - **OAM:** only images with an allowed licence and a resolution of 0.6 m or finer are considered. They are ranked by how many bridges fall inside them, then by sharpness, and the best two are kept. This fixes the Kathmandu problem, where the chosen image missed the bridges.
   - **France:** BD ORTHO through WMS.

Labels are fetched **300 m beyond the cell edge**, so a chip at the border still sees the whole bridge. A chip belongs to the cell that contains its **centre**, so two cells never cut the same tile.

### 3.3 Chips

| Chip type | How it is chosen | Label file |
|---|---|---|
| **positive** | Every z18 tile a bridge deck touches, up to 400 per cell | One YOLO-OBB line per bridge piece |
| **hard negative** | Tiles on water or roads with no bridge | Empty |
| **random negative** | Random tiles in the cell with no bridge | Empty |

There is about **one negative per positive**, and half the negatives are hard ones (doc 04 §6). A chip is kept only if at least 90 % of it is real imagery and not nodata.

Each chip is read **on its own** as a 256 × 256 window from the cloud-optimised GeoTIFF or the WMS. The full image is never downloaded.

### 3.4 Boxes and source geometry

Each bridge deck is clipped to the chip and turned into its smallest rotated rectangle. That gives the YOLO-OBB line `0 x1 y1 x2 y2 x3 y3 x4 y4`, normalised 0–1. Pieces shorter than 12 px or thinner than 2 px are skipped, and a "positive" chip left with no box is dropped.

The **original geometry** (the OSM way or outline, or the BD TOPO feature) is kept for every bridge in `index/bridges-*.parquet`, next to the deck that was used. That covers the ticket's edge case: *keep source geometry while the box type is undecided*. If we later switch to upright boxes or lines, or apply an offset correction, the labels can be rebuilt without fetching OSM again.

---

## 4. Why it doesn't run out of memory

| What could blow up | What the builder does instead |
|---|---|
| A whole image or region in memory | One ~4 km cell at a time. The cell is cleared before the next one starts |
| Downloading big GeoTIFFs | Reads 256 px windows straight from the remote COG or WMS |
| Thousands of chips in a list | Chips are kept as compressed PNG bytes and written out every **500 chips** (~40–80 MB) |
| Local disk filling up | Each 500-chip file is uploaded and then **deleted** locally |
| Big tables in analysis | `finalize` and `stats` read the image-free index tables. `to-yolo` reads one shard at a time in batches of 64 rows |

Each progress line prints the RAM in use, for example:

```
[build] oam_osm:kathmandu_002_001: 14 bridges, 2 image(s), 37 chips (21 with bridges) | 48s | RAM 310 MB
```

In the offline test, RAM stayed at about **200–300 MB** for the whole run.

### Resuming after a disconnect

Every upload is **one commit** that holds the chip file, its index, its bridges and `state/state.json`. The state file lists every finished cell. When you run `build` again it reads the state, skips the finished cells and continues. The most you can lose is the chips since the last upload (at most 500).

- `--max-hours 11` stops cleanly before Kaggle's 12-hour limit.
- A cell that fails, for example because a server returned errors, is **recorded and skipped** so the run keeps going. Run with `--retry-failed` later to try those cells again.
- Network calls retry with back-off, and Overpass falls back to three mirrors.

---

## 5. What's on Hugging Face

The dataset is a **private** dataset repo (default `<user>/fair-bridges-chips`).

```
data/shard-00000.parquet ...        chips with images (HF Image feature) + labels + metadata
index/chips-00000.parquet ...       same rows, no images (fast analysis)
index/bridges-00000.parquet ...     one row per bridge, with source geometry
splits/split.parquet                chip_id -> train / val / test / buffer / drop + reason
reports/                            leakage tests, statistics tables, figures, overlay_samples.png
MANIFEST.csv                        sha256, size, row count of every file
dataset_record.md                   sources, licences, counts, split, tests, every image used
README.md                           dataset card (licence, attribution, how to load)
state/state.json                    build progress (what makes resuming work)
```

### Main columns per chip

| Column | Meaning |
|---|---|
| `chip_id`, `file_name` | `oam_osm/OAM-x-y-z`, `OAM-x-y-z.png` |
| `image` | The 256 × 256 PNG |
| `kind`, `n_boxes` | positive / hard_negative / random_negative, and the number of boxes |
| `obb_yolo` | Label text, ready for YOLO |
| `boxes_px` | The same boxes as pixel corners (for drawing) |
| `bridge_ids` | Which bridges are in the chip (join to `index/bridges`) |
| `x, y, z, bounds_3857, lon, lat, block_z14` | Where the chip is |
| `image_id, image_url, provider, acq_date, licence_img` | Which image it came from, who to credit |
| `licence_lbl, label_source` | ODbL / OpenStreetMap or Etalab 2.0 / BD TOPO |
| `gsd_native_m, gsd_chip_m, valid_frac` | Native and chip resolution, and the share of real pixels |
| `phash, img_sha256` | Near-duplicate hash and exact hash |

### Main columns per bridge

`bridge_id`, `source`, `cls` (S = structure outline, F = filtered line), `length_m`, `width_m`, `bearing_deg`, `crossing` (water / road/rail / unknown), `deck_wkt`, `source_wkt`, `tags_json`.

### Loading it

```python
from datasets import load_dataset
ds = load_dataset("<user>/fair-bridges-chips", split="train", streaming=True)   # all shards
# filter by splits/split.parquet for train / val / test
```

---

## 6. The split and the leakage tests

`finalize` decides the split using only the index tables, in this order:

1. **Duplicates.** If the same tile was cut twice (from two images or overlapping regions), the copy from the sharper image is kept.
2. **Test.** Whole regions listed in `split.test_regions` go to test. These are OAM regions, chosen *before* training and never moved.
3. **Train and validation.** Zoom-14 blocks (about 2.4 km) are hashed in a stable way and about 20 % go to validation.
4. **Bridges stay together.** Every chip of a bridge follows the bridge's home block. A chip holding bridges from two splits becomes buffer.
5. **No other source inside a test block.**
6. **Buffer.** A chip within 1 chip (about 150 m) of a chip in another split is dropped from the lower-priority split. Priority is test, then train, then validation.
7. **Near-duplicates.** A chip whose pHash is within 4 bits of a chip in another split is dropped.

Then five leakage tests run. All must pass:

| Test | What it catches |
|---|---|
| Every chip in one split | The same chip in train and test |
| Every bridge in one split | Half a bridge in train, half in test |
| No chips of different splits within the buffer | Neighbouring chips that share context |
| Test holds only the test regions | Test chips coming from somewhere else |
| Test images not used in train or validation | The same image on both sides |

The offline test also builds a **deliberately leaky** split and checks that the tests fail on it.

---

## 7. Running it

### 7.1 Before the first run

1. **Hugging Face:** create a **write** token.
2. **Kaggle:**
   - Settings → Internet **on**.
   - Add-ons → Secrets → add `HF_TOKEN`.
   - Upload `bridges-hf-builder.zip` as a Kaggle Dataset, or set `REPO_URL` in the notebook.
3. **Colab:** 🔑 Secrets → add `HF_TOKEN` and switch on notebook access.

The token is read from the secret store and is never typed into the notebook.

### 7.2 The notebook (`notebooks/run_bridgebuild.ipynb`)

| Cell | What to do | Time |
|---|---|---|
| 1 Setup | Run as is. It detects Kaggle or Colab, installs requirements and runs the unit tests | 2–3 min |
| 2 Dry run | Builds 3 cells into a local folder. Check that every service answers and look at the overlay picture | 2–5 min |
| 3 Discover | Set a search box per city, copy the best regions into the config | minutes |
| 4 Build | `build --max-hours 11`. Re-run after any disconnect | hours |
| 5 Finalize | Split, tests, manifest, card | minutes |
| 6 Stats | Tables, figures, overlay samples, pushed to `reports/` | minutes |
| 7 Export and train | `to-yolo`, then YOLO26s-OBB with the doc 04 settings (GPU session) | – |

### 7.3 Command line

```bash
pip install -r requirements.txt
export HF_TOKEN=...   BB_REPO_ID=<user>/fair-bridges-chips
CFG=configs/phase1_oam_france.yaml

python -m bridgebuild discover --bbox 85.20 27.60 85.45 27.80       # W S E N
python -m bridgebuild build    --config $CFG --max-cells 3 --local-dir dry   # dry run, no upload
python -m bridgebuild all      --config $CFG                          # build + finalize + stats
python -m bridgebuild build    --config $CFG --only france            # one source per session
python -m bridgebuild build    --config $CFG --retry-failed           # retry cells that failed
python -m bridgebuild to-yolo  --config $CFG --out yolo_bridges --neg-keep 1.0
```

---

## 8. Config reference (`configs/phase1_oam_france.yaml`)

| Key | Default | What it changes |
|---|---|---|
| `hub.repo_id` / `BB_REPO_ID` | – | Target dataset repo (created private if missing) |
| `hub.local_dir` | off | Write to a folder instead of Hugging Face |
| `build.cell_deg` | 0.04 | Cell size. Smaller means less RAM per step but more requests |
| `build.label_margin_m` | 300 | How far beyond the cell labels are fetched |
| `build.shard_size` | 500 | Chips per file, which is also one commit |
| `build.max_hours` | 11 | Clean stop before the platform limit |
| `chips.min_valid` | 0.9 | Minimum share of real pixels in a chip |
| `chips.max_pos_per_cell` | 400 | Cap so one dense city doesn't dominate |
| `chips.neg_ratio` | 1.0 | Negatives per positive chip |
| `chips.hard_negative_share` | 0.5 | Share of negatives on water or roads |
| `sources.oam_osm.max_gsd_m` | 0.6 | Coarsest OAM image accepted |
| `sources.oam_osm.images_per_cell` | 2 | Best images kept per cell |
| `sources.oam_osm.allowed_licences` | CC BY family, CC0, PD | Non-commercial images are always refused |
| `sources.france.line_width_m` | 6 | Deck width for BD TOPO lines without a surface |
| `split.test_regions` | dar_es_salaam | Whole regions held out as test |
| `split.val_fraction` | 0.2 | Share of blocks in validation |
| `split.buffer_chips` | 1 | Buffer width in chips |
| `split.phash_max` | 4 | Near-duplicate threshold (bits) |

---

## 9. Size and limits

| Item | Number |
|---|---|
| One chip (PNG + metadata) | ~100–200 KB |
| 10,000 chips | ~1–2 GB |
| Hugging Face free private storage | 100 GB, roughly 500k–1M chips. Phase 1 needs far less |
| Commits | One per 500 chips plus one state commit per 25 empty cells. 100k chips is about 200 commits, well within limits |
| Kaggle CPU session | 12 h. The build stops at 11 h and you re-run to continue |
| Kaggle GPU | ~30 h per week |

**Don't run two build sessions on the same repo at the same time.** They would both write `state.json`. Run sources one after another, using `--only`.

---

## 10. Licences and attribution

- **OSM labels are ODbL** (share-alike), so the dataset as a whole is released under **ODbL 1.0**. The dataset card already says so.
- **OAM imagery** keeps its own licence per image, in `licence_img`. Every image used is listed in `dataset_record.md` with its provider and URL for attribution. Non-commercial images are never used.
- **IGN BD ORTHO and BD TOPO** are Etalab 2.0, credited to IGN.
- Keep the repo **private** until HOT has checked the licence wording. Then make it public or share it with the team.
- No imagery is committed to git. `.gitignore` blocks `.parquet`, `.tif`, `.png` and `.jpg` files and the work folders, as the ticket requires.

---

## 11. What the builder does *not* do (yet)

| Item | Status | Why or next step |
|---|---|---|
| **Offset correction** | Not applied | The offset sample was too noisy to correct scenes (notebook 02). The source geometry is kept, so labels can be rebuilt once a correction is trusted |
| **Phase 2 sources** (Netherlands, Switzerland, NAIP, Maxar) | Not added | Add a function to `sources.py`, register it in `SOURCES` and add a config block. Re-running `build` processes only the new cells |
| **Upright boxes or lines** | Not exported | Rotated boxes are the decision (doc 04). The source geometry is there if that changes |
| **Real-service test** | Not run by me | My test used a fake source. The notebook's dry run (cell 2) is the first real check |

---

## 12. Ticket checklist

| Ticket item | Covered by |
|---|---|
| Extract OSM bridges for chosen regions | `build`, using `discover` to choose the regions |
| Fetch OAM imagery, cut 256 × 256 chips, OAM-x-y-z names | `build` (`chips.py`) |
| Convert bridge ways to boxes, keep source geometry | `obb_yolo` / `boxes_px` plus `index/bridges` `source_wkt` |
| Spatial block split with buffer | `finalize` (§6) |
| Dataset record with licences, manifest with hashes | `dataset_record.md`, `MANIFEST.csv` |
| ⚑ Measure the offset on a sample | Done earlier (pipeline notebook 02). Not applied here (§11) |
| ⚑ Keep source geometry | `source_wkt` per bridge |
| ⚑ Don't commit large or restricted imagery | Data lives on Hugging Face only, `.gitignore`, non-commercial images refused |
| One command builds chips, labels and split; leakage test passes | `python -m bridgebuild all` (exits with code 2 on failure) |
| Builder, split, leakage test and overlay samples in `experiments/` | Commit `bridges-hf-builder/` and `reports/overlay_samples.png`, with the ticket ID in each commit |
| Dataset record as a Google Doc on Drive, linked on the ticket | Upload `dataset_record.md` after `finalize` |

---

## 13. When something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `0 chips` for every cell of a region | No OAM image there covers the bridges, or none has an allowed licence | Run `discover` and move the region |
| Many `FAILED` cells, HTTP 429/502/504 | Overpass or IGN is overloaded | Wait, then `build --retry-failed` |
| `config changed since the last run` | You edited the config between runs | Fine for adding regions. To change chip rules, use a new repo |
| `LEAKAGE TEST FAILED` | Overlapping regions or a test region next to a training region | Check `reports/leakage_tests.csv`. Move or shrink regions and re-run `finalize` |
| `401` / `403` from Hugging Face | Read-only token or the secret isn't set | Use a write token in `HF_TOKEN` |
| Kaggle can't download anything | Internet is off | Notebook settings → Internet on |
| Boxes look shifted in `overlay_samples.png` | OSM offset in that image | Note the image ID. If it is bad, add the region to a review list. Offset correction comes later (§11) |

# Poultry packaging segmentation & recognition (PDI — IFG)

- **T1 — segmentation:** locates and crops the label region on dataset images (no classification or OCR).
- **T2 — recognition:** labels each T1 crop with its product class by local-feature matching against per-class templates (see [§11](#11-product-recognition-t2)).

Assignment briefs: [requirements.md](./requirements.md) (T1) · [enunciadot2.pdf](./enunciadot2.pdf) (T2) · Glossary: [CONTEXT.md](./CONTEXT.md) · Changelog: [UPDATES.md](./UPDATES.md)

---

## 1. Input and output

```
dataset/                          result/
├── Peito_Congelado/              ├── Peito_Congelado/
│   ├── img001.jpg                │   ├── img001_segmented_1.png
│   └── ...                       │   └── ...
└── ...                           └── ...
```

- Folder names (`Peito_Congelado`, `Moela`, etc.) only organize output — the algorithm does not use them.
- Default data path: `data/Train_and_Validation/` — 18 classes × 50 images (1280×720, grayscale).
- `*.jpgZone.Identifier` sidecar files are ignored.

---

## 2. Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.12+
- [Docker](https://docs.docker.com/) + Docker Compose (for container delivery)

Course constraints: Part 1 techniques only (thresholding, morphology, filters, histograms, etc.). No ML.

---

## 3. Installation

```sh
make setup
```

Equivalent: `uv sync --extra dev`

---

## 4. Run segmentation (local)

```sh
make run
```

Writes `result/<Class>/<file>_segmented_<N>.png`.

Custom paths:

```sh
make run DATA=dataset OUT=result
```

---

## 5. Calibrate (overlays + boxes.json)

```sh
make calibrate
```

Writes `calibration/` with sample overlays, `boxes.json`, and `stats.csv`.

All frames in `boxes.json`:

```sh
make calibrate LIMIT=9999
```

---

## 6. Review in the browser

```sh
make review
```

Opens http://127.0.0.1:8765/ — shows source, overlay, and crops. Does not re-run the detector.

Other port:

```sh
make review PORT=9000
```

---

## 7. Docker (delivery)

### Production image from GitHub Packages

After the CI publishes the image, a professor can run without building locally and
without a `.env` file:

```sh
mkdir -p result calibration
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up pipeline
docker compose -f docker-compose.prod.yml --profile tools run --rm calibrate
```

The production Compose uses:

```text
ghcr.io/joserobertomi-itab/trabalho-pdi:latest
```

Put the dataset in `data/Train_and_Validation/`. Outputs are written to `result/` and
`calibration/`.

The GitHub Actions CI publishes this image on pushes to `main`, version tags like
`v1.0.0`, and manual workflow runs. After the first package is created, open the
package settings in GitHub Packages and set its visibility to **Public** so it can be
pulled without login.

### Local build

```sh
mkdir -p result calibration
cp .env.example .env
make docker-up
```

This runs all images with Docker settings from `.env`: 6 workers, a 6-CPU cap
(50% of a 12-logical-CPU notebook), and automatic GPU selection when NVIDIA Docker is
available.

Run calibration in Docker:

```sh
make docker-calibrate
```

Recommended full Docker flow:

```sh
make docker-up
make docker-calibrate
```

Do not run `docker-up` and `docker-calibrate` at the same time on a notebook.

If the notebook is still too busy, lower the CPU cap:

```sh
make docker-up DOCKER_CPUS=4.0 WORKERS=4
make docker-calibrate DOCKER_CPUS=4.0 WORKERS=4
```

Force CPU even on a machine with NVIDIA:

```sh
make docker-up DOCKER_GPU=off PDISEG_BACKEND=cpu
make docker-calibrate DOCKER_GPU=off PDISEG_BACKEND=cpu
```

If you want to run in chunks:

```sh
make docker-up MAX_IMAGES=100 OFFSET=0
make docker-up MAX_IMAGES=100 OFFSET=100
make docker-up MAX_IMAGES=100 OFFSET=200
```

Assignment-style dataset folder name:

```sh
make docker-up DATA=dataset OUT=result
```

Direct Compose equivalent:

```sh
DATA=./data/Train_and_Validation OUT=./result THREADS=1 WORKERS=6 PDISEG_BACKEND=auto DOCKER_CPUS=6.0 DOCKER_MEMORY=4g DOCKER_NICE=10 PROGRESS_EVERY=25 \
  docker compose up --build pipeline

DATA=./data/Train_and_Validation CALIB=./calibration LIMIT=9999 THREADS=1 WORKERS=6 PDISEG_BACKEND=auto DOCKER_CPUS=6.0 DOCKER_MEMORY=4g DOCKER_NICE=10 PROGRESS_EVERY=25 \
  docker compose --profile tools run --rm calibrate
```

Direct Compose with GPU override:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build pipeline
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile tools run --rm calibrate
```

Details: [docs/docker-compose.md](./docs/docker-compose.md).

If Docker review hits permission errors, in `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

Then: `make docker-export`

---

## 8. Make reference

Run `make` or `make help` to list targets.

### Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA` | `data/Train_and_Validation` | Input folder |
| `OUT` | `result` | Segmentation output |
| `CALIB` | `calibration` | Calibration output |
| `LIMIT` | `9999` | Per-class calibration cap; high value means all images |
| `MAX_IMAGES` | empty | Optional max images per run |
| `OFFSET` | `0` | Skip N sorted images before running |
| `PROGRESS_EVERY` | `25` | Print progress every N images |
| `THREADS` | `1` | Numeric library threads in Docker |
| `WORKERS` | `6` | Images processed concurrently |
| `PDISEG_BACKEND` | `auto` | Use GPU when CuPy/CUDA is available, otherwise CPU |
| `DOCKER_GPU` | `auto` | Makefile selects the GPU Compose override when NVIDIA Docker is available |
| `DOCKER_CPUS` | `6.0` | Hard Docker CPU cap; on this 12-logical-CPU notebook this is 50% |
| `DOCKER_MEMORY` | `4g` | Docker memory cap |
| `DOCKER_NICE` | `10` | Lower Docker process priority |
| `PORT` | `8765` | Review server port |

### Commands

| Command | Description |
|---------|-------------|
| `make run` | Segment → `result/` |
| `make calibrate` | Write `calibration/` |
| `make templates` | Bootstrap `templates/` from T1 (T2) |
| `make recognize` | Recognize products in `result/` segments (T2) |
| `make report` | Recognize + rebuild T2 report PDFs |
| `make review` | Web viewer |
| `make debug` | Full pipeline on 1 image/class → `debug_result/` |
| `make docker-up` | Pipeline in Docker |
| `make docker-calibrate` | Calibrate in Docker |
| `make docker-review` | Review in Docker |
| `make prod-up` | Pipeline from published GHCR image |
| `make prod-calibrate` | Calibrate from published GHCR image |
| `make prod-review` | Review from published GHCR image |
| `make docker-export` | Copy named volume → `./result` |
| `make test` | pytest |
| `make lint` | ruff check |
| `make format` | ruff format |
| `make typecheck` | mypy |
| `make check` | lint + mypy + tests |
| `make agent-check` | Validate agent harness |
| `make clean` | Remove `result/`, `calibration/`, caches |

### CLIs (without Make)

```sh
uv run pdiseg [INPUT] [OUTPUT]
uv run pdiseg-calibrate [INPUT] [OUTPUT_DIR] [--per-class-limit N]
uv run pdiseg-review --dataset DATA --calibration CALIB --result OUT [--port 8765]
uv run pdiseg-debug [INPUT] [OUTPUT] [--bundle-root DIR] [--per-class N]
uv run pdiseg-recognize [SEGMENTS_ROOT] [TEMPLATES_ROOT] [--min-match-frac F] [--sweep-csv PATH]
```

---

## 9. Grading

- 60% — performance on the provided dataset
- 40% — unseen images (frozen code)

---

## 10. Modular pipeline and debug

Detection lives under `src/pdiseg/`. **Full technical reference:** [docs/PIPELINE.md](./docs/PIPELINE.md) (I/O, preprocessing, masks, scoring, postprocess, config). **Doc index:** [docs/README.md](./docs/README.md).

| Function | Module |
|----------|--------|
| `load_image` | `io/dataset.py` |
| `preprocess_image` | `detection/preprocess.py` |
| `build_candidate_masks` | `detection/masks.py` |
| `find_candidate_boxes` | `detection/candidates.py` |
| `score_candidate` | `detection/scoring.py` |
| `postprocess_boxes` | `detection/postprocess.py` |
| `crop_and_save` / `process_dataset` | `runtime/pipeline.py` |
| `detect_name_labels` | `detection/detector.py` |

Full map: [`docs/src/ARCHITECTURE.md`](./docs/src/ARCHITECTURE.md).

Pipeline improvements: [docs/PIPELINE_IMPROVEMENTS.md](./docs/PIPELINE_IMPROVEMENTS.md).

### Debug notebook

```sh
make setup
make debug          # optional: run sample from terminal first
uv run jupyter notebook debug.ipynb
```

The notebook runs the **full production pipeline** on one image per class, writes to `debug_result/result/` and `debug_result/bundles/`, then visualizes masks, scores, and crops. Optional debug images never go into graded `result/`.

### AI harness

- Entry: [`AGENTS.md`](./AGENTS.md) and [`.agents/`](./.agents/)
- Skills: [`.cursor/skills/`](./.cursor/skills/) — e.g. `pdiseg-pipeline`, `pdiseg-debug-notebook`
- Rules: [`.cursor/rules/`](./.cursor/rules/)
- Guide: [`docs/agents/harness.md`](./docs/agents/harness.md)
- Validate: `make agent-check`

No spec-driven development — issues + `CONTEXT.md` + ADRs.

---

## 11. Product recognition (T2)

Classifies each T1 segment crop by matching **local descriptors** (skimage SIFT, ORB
optional) against one committed template per class. No ML — the decision is a single
knob: the fraction of template keypoints that find a geometrically-consistent match in
the segment. Below that threshold the crop is `unknown` (false positives cost more than
misses). Design rationale: [docs/adr/0006-local-feature-recognition.md](./docs/adr/0006-local-feature-recognition.md).

```sh
make templates      # 1. bootstrap templates/<Class>.png from the T1 detector (or hand-crop)
make run            # 2. produce T1 segments under result/
make recognize      # 3. write result/recognition.csv + calibration/recognition_sweep.csv
make report         # 4. (optional) rebuild docs/report/*.pdf
```

- **Templates** live in `templates/<Class>.png`, one per class, committed. Bootstrapped
  from the T1 label crop but replaceable by better manual crops at any time — the
  recognizer only reads the directory.
- **Pipeline:** `features.py` (keypoints/descriptors) → `matching.py` (Lowe ratio +
  cross-check + RANSAC affine verification) → `classify.py` / `batch.py` (per-segment
  scores, image-level max aggregation, `unknown` gate, threshold sweep).
- **Calibration:** `--sweep-csv` re-applies every `min_match_frac` over cached scores and
  reports accuracy, unknown count, and false-positive rate per threshold — pick the knob
  from that curve.
- **Report PDFs** (Brazilian Portuguese, graded deliverable):
  [docs/report/t2_report.pdf](./docs/report/t2_report.pdf) (full) and
  [t2_simplified.pdf](./docs/report/t2_simplified.pdf) (templates · parameters · results).

Details of everything added for T2: [UPDATES.md](./UPDATES.md).

---

## 12. Delivery

Colab on Moodle **or** Docker Compose.

Share the Colab link with: `alessandro.rodrigues@ifg.edu.br`

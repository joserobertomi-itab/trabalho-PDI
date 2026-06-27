# Poultry packaging segmentation (PDI — IFG, Practical Assignment 1)

Locates and crops the label region on dataset images. Segmentation only — no classification or OCR.

Assignment brief: [requirements.md](./requirements.md) · Glossary: [CONTEXT.md](./CONTEXT.md)

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

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

Or via Make:

```sh
make docker-up
```

Assignment-style dataset folder name:

```sh
DATA=./dataset docker compose up --build
```

Extra tools (`tools` profile):

```sh
make docker-calibrate
make docker-review
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
| `LIMIT` | `3` | Overlays per class in calibrate |
| `PORT` | `8765` | Review server port |

### Commands

| Command | Description |
|---------|-------------|
| `make run` | Segment → `result/` |
| `make calibrate` | Write `calibration/` |
| `make review` | Web viewer |
| `make debug` | Full pipeline on 1 image/class → `debug_result/` |
| `make docker-up` | Pipeline in Docker |
| `make docker-calibrate` | Calibrate in Docker |
| `make docker-review` | Review in Docker |
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

## 11. Delivery

Colab on Moodle **or** Docker Compose.

Share the Colab link with: `alessandro.rodrigues@ifg.edu.br`

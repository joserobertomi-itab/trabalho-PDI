# Poultry Packaging Segmentation

Automatically **locate and segment** the packaging of poultry products in images
captured in an industrial environment, using classical Digital Image Processing
(PDI — *Processamento Digital de Imagens*, IFG).

This is **Trabalho Prático 1**. Scope is **segmentation only** — no classification
and no OCR. See [requirements.md](./requirements.md) for the original brief and
[CONTEXT.md](./CONTEXT.md) for the domain glossary.

## What it does

The program walks every image in every folder under `dataset/`, detects the
package region(s) in each image, and writes one or more cropped images containing
only the segmented packaging to a matching folder under `resultado/`.

A crop is correct when it sits where the product name is expected — the text does
not need to be legible. A crop **without** the product name (or with an irrelevant
one, e.g. just "frango") counts as a false positive and is scored as an error.

## I/O layout

```
dataset/                          resultado/
├── Peito_Congelado/              ├── Peito_Congelado/
│   ├── img001.jpg                │   ├── img001_segmentada_1.png
│   └── ...                       │   ├── img001_segmentada_2.png
├── Moela/                        │   └── ...
└── ...                           ├── Moela/
                                  └── ...
```

Source folder names are product classes (e.g. `Peito_Congelado`, `Moela`); they
are used **only to organize the output**, never as input to the algorithm.

The provided base under `data/Train_and_Validation/` holds **18 classes × 50
grayscale 1280×720 frames = 900 images**. Each image ships with a Windows
`<name>.jpgZone.Identifier` sidecar file; these are metadata, not images, and are
ignored by the pipeline.

## Technique constraints

Only techniques from **Part 1** of the course are allowed:

- Color spaces · thresholding · segmentation · morphological operations
- Spatial filtering · geometric transformations · histograms

**Not allowed:** anything not yet covered in the course, and any library that
performs segmentation automatically via AI.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.12+
- [Docker](https://docs.docker.com/) + Docker Compose (optional, for containerized runs)

## Running locally (uv + Make)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
make help       # list all tasks
make setup      # uv sync --extra dev (locked deps from uv.lock)
make check      # lint + typecheck + tests
make run        # segment the base into resultado/
make calibrate  # overlays + stats.csv into calibration/
```

Override paths on the command line:

```sh
make run DATA=dataset OUT=resultado
make calibrate CALIB=calibration LIMIT=5
```

Entry points (also available after `uv sync`):

```sh
uv run pdiseg [INPUT_ROOT] [OUTPUT_ROOT]
uv run pdiseg-calibrate [INPUT_ROOT] [OUTPUT_DIR] [--per-class-limit N]
python -m pdiseg [INPUT_ROOT] [OUTPUT_ROOT]
python -m pdiseg.calibrate_cli [INPUT_ROOT] [OUTPUT_DIR]
```

## Running with Docker Compose

Build once, then run segmentation or calibration with mounted volumes:

```sh
cp .env.example .env   # optional: customize DATA / OUT / CALIB
make docker-build
make docker-run        # writes to ./resultado
make docker-calibrate  # writes to ./calibration
```

Or directly:

```sh
docker compose run --rm segment
docker compose run --rm calibrate
DATA=./dataset OUT=./resultado docker compose run --rm segment
```

The container runs as a non-root user (`pdiseg:1000`), with a read-only root
filesystem; only the mounted output volumes are writable. The dataset mount is
read-only.

## Development workflow

```sh
make setup
make format     # ruff format + auto-fix
make lint       # ruff check
make typecheck  # mypy (strict on pdiseg/)
make test       # pytest with ≥80% coverage gate
make ci         # full local gate (sync + check)
```

Optional pre-commit hooks:

```sh
uv run pre-commit install
uv run pre-commit run --all-files
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

1. `uv sync --frozen --extra dev`
2. Ruff lint + format check
3. Mypy
4. Pytest with coverage
5. Docker image build + entrypoint smoke test

## Evaluation

| Criterion | Weight |
|---|---|
| Performance on the provided base | 60% |
| Complementary evaluation on unseen images (code frozen) | 40% |

The complementary stage runs the frozen code on images not released in advance, to
measure generalization.

## Repository layout

```
.
├── pdiseg/             # pipeline + calibration CLI
├── tests/              # synthetic-dataset tests (public API only)
├── pyproject.toml      # project metadata + tool config (source of truth)
├── uv.lock             # locked dependency graph
├── Dockerfile          # multi-stage production image (uv + non-root)
├── docker-compose.yml  # segment + calibrate services
├── CONTEXT.md          # domain glossary
├── requirements.md     # original assignment brief (Portuguese)
└── docs/
    ├── adr/            # pipeline decisions
    └── agents/         # agent-skill configuration
```

## Deliverables

- Submit the **Colab link** on Moodle, **or** a **Docker Compose** with the solution.
- Share the Colab link with `alessandro.rodrigues@ifg.edu.br`.

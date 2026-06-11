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

## Running

### Locally (Make)

A `Makefile` wraps the common tasks (run `make help` to list them):

```sh
make setup      # create .venv and install runtime + dev dependencies
make test       # run the test suite
make run        # segment the base into resultado/<Class>/<source>_segmentada_<N>.png
make calibrate  # write per-class overlays + stats.csv into calibration/ for review
```

Override the paths on the command line, e.g. `make run DATA=dataset OUT=resultado`
or `make calibrate CALIB=calibration LIMIT=5`. Both `resultado/` and `calibration/`
are git-ignored. The underlying entry points are `python -m pdiseg [IN] [OUT]`
(segmentation) and `python -m pdiseg.calibrate_cli [IN] [OUT]` (calibration).

### Delivery (issue #7)

The frozen solution is delivered in one of two forms:

- **Google Colab** — open the notebook, mount/point it at `dataset/`, run all cells.
- **Docker Compose** — `docker compose up`, reading from `dataset/` and writing to
  `resultado/`.

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
├── CONTEXT.md          # domain glossary and language
├── requirements.md     # original assignment brief (Portuguese)
├── docs/
│   ├── adr/            # pipeline decisions (recorded during planning)
│   └── agents/         # agent-skill configuration
└── AGENTS.md           # contributor / agent instructions
```

## Deliverables

- Submit the **Colab link** on Moodle, **or** a **Docker Compose** with the solution.
- Share the Colab link with `alessandro.rodrigues@ifg.edu.br`.

# Context

Glossary for this repo. Class folder names stay in Portuguese because they match `dataset/`.

## What we're building

Segment poultry packaging in industrial camera frames (PDI class, IFG). Find a validated **label cluster** per frame and write a crop per detection. The crop must contain product-type label evidence and may include the adjacent brand badge. No OCR, no product classification.

## Terms

- **bandeja** / **selado** — tray vs sealed packaging in the dataset.
- **Name label** — product-type badge/region carrying the product name (e.g. "FILÉ DE PEITO"). Used as mandatory product evidence inside a final crop.
- **Brand badge** — "SUPER FRANGO" logo next to the name label; not the target alone but helps locate clusters.
- **Label cluster** — production crop target: name label plus adjacent brand/context when they form the same local package label. A brand-only crop is invalid.
- **Class** — one product folder under `dataset/` (e.g. `Moela`). Used only for output paths.
- **False positive** — crop without the right product name (or only generic "frango").
- **Overlay** — source frame with boxes drawn (red rejected candidates, yellow product anchors, green final label clusters).
- **Review viewer** — small web UI (`pdiseg-review`) to browse source, overlay and crops. Not part of graded pipeline.

## Dataset

18 classes, 50 images each, 1280×720 grayscale, FPS counter burned in top-left. Same rig for unseen eval.

Class folders (names as in data): Asas Resfriado Selado, Meio das Asas Congelado, Coxinhas das Asas Congelado, Coxinhas das Asas Congelado Selado, Meio das Asas Congelado Selado, Coxinhas das Asas Resfriado Selado, Filé de Peito Congelado, Filezinho Sassami Resfriado Selado, Filezinho Sassami Congelado, Filé de Peito Congelado Selado, Coração, Moela, Peito Congelado, Peito Resfriado, Filé de Coxas e Sobrecoxas com Pele Congelado Selado, Filé de Coxas e Sobrecoxas com Pele Congelado, Coxas e Sobrecoxas Congelado, Coxas e Sobrecoxas Resfriado Selado.

## I/O

```
dataset/<Class>/<image>.jpg  →  result/<Class>/<stem>_segmented_N.png
```

Full I/O and batch runner details: [docs/PIPELINE.md §3](docs/PIPELINE.md#3-io-layer-pdisegio-dataset).

## Allowed techniques

Part 1 only: color spaces, thresholding, segmentation, morphology, spatial filters, geometric transforms, histograms. No AI auto-segmentation libs.

## Evaluation

60% provided base, 40% unseen images with frozen code.

## Deliverables

Colab on Moodle or Docker Compose; email Colab link to alessandro.rodrigues@ifg.edu.br.

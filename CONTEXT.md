# Context

Glossary for this repo. Class folder names stay in Portuguese because they match `dataset/`.

## What we're building

Segment poultry packaging in industrial camera frames (PDI class, IFG). Find each **name label** (dark badge with product name) and write a crop per detection. No OCR, no product classification.

## Terms

- **bandeja** / **selado** — tray vs sealed packaging in the dataset.
- **Name label** — dark rounded rectangle with the product name (e.g. "FILÉ DE PEITO"). Target of segmentation.
- **Brand badge** — "SUPER FRANGO" logo next to the name label; not the target alone but helps locate clusters.
- **Label cluster** — brand badge + name label treated as one blob in stage 1; then refined to the name label.
- **Class** — one product folder under `dataset/` (e.g. `Moela`). Used only for output paths.
- **False positive** — crop without the right product name (or only generic "frango").
- **Overlay** — source frame with boxes drawn (red rejected, yellow kept, green final labels).
- **Review viewer** — small web UI (`pdiseg-review`) to browse source, overlay and crops. Not part of graded pipeline.

## Dataset

18 classes, 50 images each, 1280×720 grayscale, FPS counter burned in top-left. Same rig for unseen eval.

Class folders (names as in data): Asas Resfriado Selado, Meio das Asas Congelado, Coxinhas das Asas Congelado, Coxinhas das Asas Congelado Selado, Meio das Asas Congelado Selado, Coxinhas das Asas Resfriado Selado, Filé de Peito Congelado, Filezinho Sassami Resfriado Selado, Filezinho Sassami Congelado, Filé de Peito Congelado Selado, Coração, Moela, Peito Congelado, Peito Resfriado, Filé de Coxas e Sobrecoxas com Pele Congelado Selado, Filé de Coxas e Sobrecoxas com Pele Congelado, Coxas e Sobrecoxas Congelado, Coxas e Sobrecoxas Resfriado Selado.

## I/O

```
dataset/<Class>/<image>.jpg  →  result/<Class>/<stem>_segmented_N.png
```

## Allowed techniques

Part 1 only: color spaces, thresholding, segmentation, morphology, spatial filters, geometric transforms, histograms. No AI auto-segmentation libs.

## Evaluation

60% provided base, 40% unseen images with frozen code.

## Deliverables

Colab on Moodle or Docker Compose; email Colab link to alessandro.rodrigues@ifg.edu.br.

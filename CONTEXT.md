# Context — Poultry Packaging Segmentation (PDI)

> Glossary and domain language for this repo. Artifacts are written in English
> (see `AGENTS.md`), but **product/class names stay in Portuguese** because they
> are literal `dataset/` folder identifiers — translating them would break the
> link to the data.

## Purpose

Academic Digital Image Processing assignment (PDI — *Processamento Digital de
Imagens*, IFG). Build a program that automatically **locates and segments
packaging of poultry products** in images captured in an industrial environment.

This first stage is **segmentation only**. There is no classification and no OCR
(text reading). The product name may appear partially inside a segmented region,
but recognizing the text is out of scope.

## Glossary

- **Package / packaging** — the product container to be located and cropped out
  of the source image. Two physical forms appear in the dataset:
  - **bandeja** (tray) — open tray packaging.
  - **selado** (sealed) — sealed/vacuum packaging.
- **Name label** — the segmentation target: the dark, rounded-rectangle badge on a
  package that carries the product name in bright text (e.g. "FILÉ DE PEITO",
  "MOELA"). It is distinct from the **brand badge** ("SUPER FRANGO" logo) that sits
  beside it. Each name label found in a source image becomes one segmented image.
- **Brand badge** — the "SUPER FRANGO" logo region. It is *not* the target on its
  own, but it is a constant landmark: it appears next to every name label and looks
  the same across all classes.
- **Label cluster** — the **brand badge + adjacent name label** taken as one unit.
  It is the strong, consistent signature used to *locate* a product on the shelf;
  the name label is then isolated inside it. A label-cluster crop already contains
  the product name, so emitting it counts as a valid segmentation (not a false
  positive) when name-label isolation can't be refined.
- **Segmentation** — locating each name label in a source image and producing one
  cropped output image per label, containing (at least) that label region.
- **Class** — a product category, equal to a `dataset/` folder name (e.g.
  `Peito_Congelado`, `Moela`). Class names are used **only to organize results**,
  not as input to the algorithm.
- **Source image** — an input frame under `dataset/<Class>/`. Grayscale, 1280×720,
  with a burned-in "FPS: NN.NN" overlay in the top-left corner. Captured by a fixed
  industrial camera at a fixed position; the unseen evaluation set uses the same
  rig and the same resolution.
- **Segmented image** — an output crop under `resultado/<Class>/`, one per detected
  **name label** (`<source>_segmentada_<N>.png`).
- **False positive** — an output image that does **not** contain the product name,
  or contains an irrelevant name (e.g. just "frango"). False positives are scored
  as errors. A crop holding only part of the name (e.g. "asas" in one, "resfriada"
  in another) is acceptable; the text does not need to be legible, only correctly
  positioned.
- **Generalization** — performance on unseen images not provided in advance,
  measured at presentation time with the code frozen.

## Industrial-image challenges

The source images carry typical industrial-environment difficulties the pipeline
must tolerate:

- Reflections / specular highlights
- Package deformation
- Lighting variation
- Differences in product position and orientation
- Many stacked packages per frame (a packed crate): mutual occlusion and dense
  text clutter, so a frame normally holds several name labels at once
- Crinkled-bag classes glare into high-contrast filaments that mimic text and are
  the dominant source of false positives

## Dataset classes

Folder names present in the base (kept verbatim, in Portuguese):

Asas Resfriado Selado · Meio das Asas Congelado · Coxinhas das Asas Congelado ·
Coxinhas das Asas Congelado Selado · Meio das Asas Congelado Selado ·
Coxinhas das Asas Resfriado Selado · Filé de Peito Congelado ·
Filezinho Sassami Resfriado Selado · Filezinho Sassami Congelado ·
Filé de Peito Congelado Selado · Coração · Moela · Peito Congelado ·
Peito Resfriado · Filé de Coxas e Sobrecoxas com Pele Congelado Selado ·
Filé de Coxas e Sobrecoxas com Pele Congelado · Coxas e Sobrecoxas Congelado ·
Coxas e Sobrecoxas Resfriado Selado

## I/O structure

```
dataset/                          resultado/
├── Peito_Congelado/              ├── Peito_Congelado/
│   ├── img001.jpg                │   ├── img001_segmentada_1.png
│   └── ...                       │   ├── img001_segmentada_2.png
├── Moela/                        │   └── ...
└── ...                           ├── Moela/
                                  └── ...
```

The program walks **every image in every folder** of `dataset/`, detects the
package region(s), and writes one or more `*_segmentada_N.png` crops per source
image under the matching `resultado/<Class>/` folder.

## Constraints (technique scope)

**Allowed** — only techniques from **Part 1** of the course:

- Color spaces
- Thresholding
- Segmentation
- Morphological operations
- Spatial filtering
- Geometric transformations
- Histograms

**Forbidden** — anything not yet covered in the course, and any library that
performs segmentation automatically via AI.

## Evaluation

| Criterion | Weight |
|---|---|
| Performance on the provided base | 60% |
| Complementary evaluation (unseen images) | 40% |

The complementary evaluation runs the **frozen** code on images not released in
advance — no code changes allowed during evaluation. It measures generalization.

## Deliverables

- Submit the **Colab link** on Moodle, **or** a **Docker Compose** with the
  solution.
- Share the Colab link with `alessandro.rodrigues@ifg.edu.br`.

## See also

- [requirements.md](./requirements.md) — original assignment brief (Portuguese).
- `docs/adr/` — pipeline decisions, recorded as they are resolved during grilling.

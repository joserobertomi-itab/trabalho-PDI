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
- **Segmentation** — isolating the package region from the background and
  producing one or more cropped output images containing only that region.
- **Class** — a product category, equal to a `dataset/` folder name (e.g.
  `Peito_Congelado`, `Moela`). Class names are used **only to organize results**,
  not as input to the algorithm.
- **Source image** — an input photo under `dataset/<Class>/`.
- **Segmented image** — an output crop under `resultado/<Class>/`, one per detected
  package region.
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

# Practical Assignment 1 — Poultry Product Packaging Segmentation

## Objective

Build a solution that locates and segments poultry product packaging in images captured in an industrial environment.

In this first project stage there is **no** text classification or OCR. The exclusive focus is **packaging segmentation** in the images.

The dataset contains images of different products, including **tray** and **sealed** packaging types, spread across product classes. Folder names reflect the product but are used **only** to organize results.

---

## Classes in the Dataset

- Asas Resfriado Selado
- Meio das Asas Congelado
- Coxinhas das Asas Congelado
- Coxinhas das Asas Congelado Selado
- Meio das Asas Congelado Selado
- Coxinhas das Asas Resfriado Selado
- Filé de Peito Congelado
- Filezinho Sassami Resfriado Selado
- Filezinho Sassami Congelado
- Filé de Peito Congelado Selado
- Coração
- Moela
- Peito Congelado
- Peito Resfriado
- Filé de Coxas e Sobrecoxas com Pele Congelado Selado
- Filé de Coxas e Sobrecoxas com Pele Congelado
- Coxas e Sobrecoxas Congelado
- Coxas e Sobrecoxas Resfriado Selado

---

## Problem Description

Images show typical industrial challenges:

- Reflections
- Package deformation
- Lighting variation
- Different product position and orientation

The program must walk all images in all dataset folders, detect the region of the packaging that carries the product name, and write one or more crops containing only the segmented packaging area.

> **Note:** False positives count as errors — crops without the product name, or with irrelevant names (e.g. generic “chicken only”). Segmented images may contain only part of the name (e.g. “asas” in one crop and “resfriada” in another). Words need not be fully readable; the crop must cover the expected label region.

---

## File Layout

### Input

```
dataset/
├── Peito_Congelado/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Moela/
└── ...
```

### Output

> **Note:** This repository standardizes the output folder as `result/` and crop filenames as `{stem}_segmented_{N}.png` (English naming).

```
result/
├── Peito_Congelado/
│   ├── img001_segmented_1.png
│   ├── img001_segmented_2.png
│   ├── img002_segmented_1.png
│   ├── img003_segmented_1.png
│   └── ...
├── Moela/
└── ...
```

---

## Constraints and Allowed Techniques

**Not allowed:** topics not yet covered in the course, or libraries that perform segmentation automatically via AI.

**Allowed:** any technique from **Part 1** of the course only, including:

- Color spaces
- Thresholding
- Segmentation
- Morphological operations
- Spatial filtering
- Geometric transforms
- Histograms

Additional algorithms are allowed if they belong to Part 1 topics.

---

## Grading Criteria

| Criterion | Weight |
|-----------|--------|
| Performance on provided dataset | 60% |
| Complementary evaluation (unseen images) | 40% |

### Details

1. **Provided dataset (60%):** The algorithm runs on the released image set.

2. **Complementary evaluation (40%):** On presentation day the algorithm runs on unseen images. **No code changes** are allowed during evaluation. The goal is to measure **generalization**.

---

## Deliverables

- Submit the **Colab link** on Moodle, **or** a **Docker Compose** solution.
- Share the Colab link with: [alessandro.rodrigues@ifg.edu.br](mailto:alessandro.rodrigues@ifg.edu.br)

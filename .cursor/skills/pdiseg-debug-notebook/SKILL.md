---
name: pdiseg-debug-notebook
description: >-
  Debug segmentation end-to-end using debug.ipynb and pdiseg.debug_viz helpers.
  Use when investigating wrong crops, empty detections, glare failures, or
  visualizing masks and scores per frame or class.
---

# Debug segmentation (notebook)

## Setup

```sh
make setup
uv run jupyter notebook debug.ipynb
```

Paths: `data/Train_and_Validation`, `result/`, `debug_result/` (debug only).

## Per-frame inspection

```python
import pdiseg
img = pdiseg.load_image(path)
det, prep, masks = pdiseg.debug_frame(img)
pdiseg.scored_table(det.scored)  # via pandas
```

Visualize: `pdiseg.visualize_masks(masks)`, `pdiseg.draw_boxes`, `pdiseg.crop`.

## Save debug bundle

```python
pdiseg.save_debug_bundle(img, f"debug_result/{class_name}", stem)
```

Never write debug overlays into `result/`.

## Workflow

Follow `.agents/workflows/debug-segmentation.md` and role `.agents/roles/debugger.md`.

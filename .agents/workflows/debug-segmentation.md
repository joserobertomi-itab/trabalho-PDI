# Workflow: Debug segmentation

Use when crops are wrong, missing, or too numerous.

## Steps

1. Open `debug.ipynb` or create a one-off script using `pdiseg.debug_frame`.
2. Select class + image path from `data/Train_and_Validation/`.
3. Visualize: original, `prep.work`, mask layers, raw candidates, scored table, final labels.
4. Sort `scored_table` by score — inspect `text_density`, `dark_density`, `glare_fraction`.
5. If postprocess drops good boxes: trace `refine_to_name_label` shrink and `refine_score_floor`.
6. Save bundle: `pdiseg.save_debug_bundle(img, "debug_result/<class>", stem)`.
7. Fix in code; re-test same frame before batch run.

## Quick CLI

```python
import pdiseg
from pathlib import Path
p = next(Path("data/Train_and_Validation").rglob("*.jpg"))
img = pdiseg.load_image(p)
det = pdiseg.inspect_detection(img)
print(len(det.labels), det.labels)
```

# Calibration: locked recall-favoring thresholds for an inherently cluttered scene

The calibration harness (issue #6) ran the full pipeline over the base (18 classes ×
50 frames = 900 images), producing per-class overlays and funnel stats so a human
could judge detection quality and lock the fixed pixel-size thresholds. The camera and
resolution are fixed and no code changes are allowed during the official evaluation, so
the thresholds had to be frozen here.

## What the calibration showed

- The frames are **crates packed with many packages**, each carrying its own name
  label, so the segmentation target is **every visible name label** — multiple
  detections per frame are correct, not a bug.
- On **sealed-box classes** (flat printed faces, "...Selado") the text-density detector
  frames labels reasonably.
- On **crinkled-bag classes** (e.g. `Peito_Congelado`) the plastic reflects light and
  produces high-contrast filaments across the whole frame (~20% of pixels fire on the
  text mask). The real name-label text is buried in this glare and is **not separable by
  a single global signal**:
  - A wider morphological closing merges the entire crate into one blob (rejected by
    max-area) — so the closing must stay small.
  - A morphological opening sized to drop the thin glare filaments also destroys the
    letter strokes (the density mask captures letter *edges*, not solid interiors).
  - Raising the Stage-3 minimum area drops real labels faster than noise, because the
    labels are small in pixels and overlap the noise in area.

This is a limitation of classical thresholding + morphology on this clutter, not a
parameter bug. There is **no per-label ground truth** in the base, so calibration is by
visual judgement on the overlays, and **folder names are never used as algorithm input**
(they only group and label the reports).

## Decision

Lock the thresholds at **recall-favoring** values rather than chase precision:

- **Fix the FPS-overlay edge artifact**: fill the masked region with the frame median
  instead of `0`. A hard `0`-edge against bright neighbours read as a spurious top-left
  cluster on low-content frames; a neutral fill leaves a flat region with no
  text-density response.
- **Keep Stage-1 closing small** (`np.ones((7, 15))`) and clusters many-per-frame.
- **Keep Stage-3 geometry loose** (`_LABEL_MIN_AREA=3000`, `_LABEL_MAX_AREA=150000`,
  `_LABEL_MAX_ELONGATION=4.0`): reject only the unambiguous non-labels (SSA box /
  full-frame merges, barcodes / thin edges) and accept the resulting false positives.

## Consequences

- The delivered segmentation is **approximate**: good on sealed-box classes, noisier on
  crinkled-bag classes, with false positives the loose geometry deliberately tolerates
  to preserve recall against the "every visible name label" target.
- A future reader must not "tighten" `_LABEL_MIN_AREA` or enlarge the closing to clean
  up the overlays — both were measured to cost more real labels than noise (see above).
  Revisit only with a genuinely more discriminative signal (e.g. anchoring on the solid
  "SUPER FRANGO" brand badge), which is out of scope for the classical-techniques brief.
- The calibration artifacts (per-class overlays + `stats.csv`) are review-only and are
  written outside the repo; they are not part of the delivered `result/`.

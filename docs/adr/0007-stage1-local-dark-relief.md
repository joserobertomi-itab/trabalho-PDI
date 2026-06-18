---
status: proposed (under validation in debug.ipynb)
---

# Stage 1 keys on *local* darkness (top-hat relief), not a global dark percentile

The dark-badge Stage 1 (ADR 0006) thresholds the darkest *percentile* of the **whole frame**,
assuming the name label is among the globally darkest pixels. That assumption breaks on real
frames: many labels are **mid-grey**, not dark in absolute terms (e.g. the "peito" label sits at
roughly the same grey level as the surrounding fields). When a frame also carries genuinely darker
regions — shadows, crate gaps, dark plastic — those eat the darkest-*p*% budget and the grey label
never enters `dark_mask`. The label is then missed at Stage 1, which a recall net must not do.

The fix changes the discriminator from **absolute** darkness to **local** darkness. The label is
always darker than its *immediate surround*, whatever its absolute grey level. A **black top-hat**
(`dark_relief` = `grey_closing(image) − image`) surfaces exactly that: it lights up dark structures
*smaller than the structuring element* sitting on a brighter local background, and is ~0 on flat or
bright regions. So Stage 1 becomes: `dark_relief` (top-hat) → `relief_mask` (top *p*% of the relief)
→ `open_mask` (drop specks) → `close_mask` 9×9 (solidify the badge) → connected components → boxes.
On the two probe frames this recovers the grey labels the global percentile dropped and raises
recall (more candidates/kept) — over-detection is expected and is Stage 3's job to reject.

The structuring-element `size` must **exceed the badge's larger dimension**: a region larger than
the element is not filled by the closing, so its interior yields ~0 relief and is suppressed — which
is precisely why the top-hat ignores large shadows that fooled the global percentile. Too large and
it starts catching dark structures bigger than a label.

## Status / how it ships

- `dark_relief`, `relief_mask` and `detect_dark_relief_badges` are added **alongside** the ADR 0006
  `dark_mask` / `detect_dark_badges` (validation-first, same as 0006 was). `debug.ipynb` runs the
  *proposed* relief pipeline end to end with `TOPHAT_SIZE` + `STAGE1_PERCENTILE` knobs and a
  `TOPHAT_SIZE` sweep, rendering the old global-dark badges as red-dashed contrast so the recovered
  grey labels are visible.
- `_TOPHAT_SIZE = 51` and `_RELIEF_PERCENTILE = 20` are **provisional**. Lock them here once
  validated against the real frames (and ground-truth boxes, once captured) across classes.
- Promotion is a one-line change (point `detect_name_labels`/`inspect_frame` Stage 1 at
  `detect_dark_relief_badges`), at which point this ADR supersedes the Stage-1 portions of ADR 0006.

## Considered options

- **Local top-hat relief (chosen)** — pure morphology + threshold (within the Part-1 brief);
  invariant to absolute grey level, which is the property a grey label needs.
- **Local adaptive threshold** (`threshold_local`, deferred) — same global→local family and a viable
  alternative; pick between it and the top-hat by calibration if the top-hat under/over-detects.
- **CLAHE in `preprocess`** (deferred) — boosts local contrast before thresholding, but is local /
  non-monotonic, so it would break the percentile invariance ADR 0006 relied on; a separate decision.
- **Dark ∧ text conjunction** (deferred, inherited from ADR 0006) — a locally-dark region that
  *contains* bright strokes; more discriminative, more code. Revisit if relief over-detects.

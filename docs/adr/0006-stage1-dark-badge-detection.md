---
status: proposed (under validation in debug.ipynb) — Stage-1 portions being revised by ADR 0007
---

# Stage 1 pivots from text-density to dark-badge detection

> **Revised by ADR 0007:** the global dark *percentile* below misses mid-grey labels (it assumes
> the label is among the globally darkest pixels). ADR 0007 keeps the dark-badge intuition but
> switches the discriminator to *local* darkness via a black top-hat. This ADR's reasoning stands;
> only the global→local threshold is superseded.

The text-density Stage 1 (ADR 0001 reconciliation) fails on glare-heavy frames: the plastic
reflections fire the bright-text mask on ~17-20% of pixels, the morphological closing then merges
the whole crate into one blob, and the candidate boxes cover 89-96% of the frame — the real name
labels are engulfed rather than located. This is the failure ADR 0004 anticipated; no closing size
fixes it because the wrong signal (brightness) cannot separate label text from glare.

The discriminator is **darkness**: the name label is bright text on a **dark** rounded-rectangle
badge, while the glare is bright-on-bright. So Stage 1 now keys on the dark badge:
`dark_mask` (darkest *percentile* of pixels) → `open_mask` (drop glare specks) → `close_mask` 9x9
(solidify the badge) → connected components → boxes. On the two probe frames this turns "1 box at
96%" into 4 (easy) and ~4-7 (hard) compact, label-sized candidates.

The threshold is a **percentile**, which is invariant to the monotonic `equalize_hist` in
`preprocess` — so dark detection runs on the existing preprocessed frame with no second
preprocessing path, and equalize stays only for the Stage-2 Otsu split.

## Status / how it ships

- `detect_dark_badges` is added alongside the old `detect_clusters`; the graded
  `detect_name_labels` still uses the old path. `debug.ipynb` runs the *proposed* dark pipeline
  end to end with a `STAGE1_PERCENTILE` knob + sweep so the threshold is locked against real frames
  (and, once captured, ground-truth label boxes) across classes before promotion.
- `_DARK_PERCENTILE = 20` is **provisional**. Lock it here once validated.
- Promotion is a one-line change (point `detect_name_labels`/`inspect_frame` Stage 1 at
  `detect_dark_badges`), at which point this ADR supersedes the Stage-1 portions of ADR 0001 and
  ADR 0004, and the text-density `detect_clusters` is retired or kept only as a documented contrast.

## Considered options

- **Dark-badge detection (chosen)** — uses only thresholding + morphology (clearly within the
  Part-1 brief); darkness is the one signal that separates label from glare.
- **Brand-badge (SUPER FRANGO) anchoring** (deferred) — the most consistent landmark, but template
  correlation is of uncertain scope under "techniques covered so far".
- **Dark ∧ text conjunction** (deferred) — a dark region that *contains* bright strokes; more
  discriminative (rejects shadows/gaps) but more code. Revisit if dark-only over-detects.

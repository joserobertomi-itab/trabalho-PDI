# T2 product recognition via local feature matching (skimage SIFT, no OpenCV)

T2 (`enunciadot2.pdf`) requires identifying the product class of each T1 segment by
matching hand-engineered local descriptors against one manually curated template crop
per tray class. No AI/ML, decision based exclusively on local-descriptor analysis,
calibrated on a single knob: the percentage of valid matches between template and
segment keypoints. False positives are worse than missed matches.

## Decision

- New sub-package `pdiseg.recognition` (sibling of `detection`), CLI `pdiseg-recognize`.
- Descriptor: **`skimage.feature.SIFT`** (default), `ORB` selectable via config. No
  OpenCV dependency: scikit-image already ships SIFT/ORB and `match_descriptors`,
  and the brief only *allows* FLANN, it does not require it. Brute-force matching is
  fine at this scale (≤ ~6 k small crops × 18 templates).
- Matching: `match_descriptors(cross_check=True, max_ratio=…)` — Lowe ratio test plus
  cross-check, both precision-first filters.
- Score: `valid matches / template keypoints`. A segment is labelled with the
  best-scoring template **only if** the score clears `min_match_frac`; otherwise the
  result is `unknown`. One source image may have several T1 segments; the image-level
  prediction is the best segment score.
- Templates live in `templates/<Class>.png`, committed. They are bootstrapped from the
  T1 detector's label-cluster crop of one image per class and may be replaced by better
  manual crops at any time; the recognizer only reads the directory.

## Consequences

- `min_match_frac` (and `max_ratio`) are calibrated by a sweep that reports accuracy,
  false-positive rate, and per-class confusion; the chosen defaults are recorded in
  `RecognitionConfig`.
- `unknown` is a valid, expected output — preferring it over a wrong class is the
  design goal, matching the brief's false-positive requirement.
- skimage SIFT is slower than OpenCV's; acceptable because segments are small crops.
  If runtime ever matters, swapping the matcher for FLANN is a contained change inside
  `pdiseg.recognition.matching`.

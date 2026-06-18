# Two-stage label detection: locate the label cluster, then refine to the name label

The segmentation target is the **name label** (the dark badge carrying the product
name). The obvious approach — threshold the frame for dark regions and label the
blobs directly — was tried and fails: a global Otsu threshold merges each badge with
neighbouring shadows and inter-package gaps into full-height column blobs (see the
exploration in the planning session). So we detect in two stages instead: **Stage 1**
locates the **label cluster** (the constant "SUPER FRANGO" brand badge + its adjacent
name label) using text-density + morphology + connected components, because the
cluster is a far more distinctive and consistent signature than the bare badge;
**Stage 2** then isolates the dark name label *inside* each small cluster crop, where
only two regions exist (bright brand badge vs dark name label) and a local threshold
separates them trivially.

## Consequences

- If Stage 2 refinement fails on a hard case, we emit the **label-cluster crop** as a
  fallback. It still contains the product name, so it is a valid segmentation, not a
  false positive — a safety net that matters for the frozen-code evaluation on unseen
  images.
- The approach assumes every product carries the "SUPER FRANGO" brand badge. This held
  for every class inspected; a class without the badge would need a separate path.

## Considered options

- **Direct dark-badge thresholding** (rejected) — global Otsu merges badges with
  surrounding dark regions into columns; does not isolate individual labels.
- **Pure text-density detection without the cluster framing** (rejected as the primary
  signature) — localises labels but over-detects (nutrition tables, barcodes, the SSA
  logo, the FPS overlay), pushing all the burden onto false-positive rejection.

## Reconciliation (2026-06-17) — what Stage 1 actually is

The shipped `detect_clusters` does **not** anchor on the "SUPER FRANGO" brand badge.
It is **pure text-density**: `uniform_filter` local-mean threshold → `binary_closing` →
connected components → min-area drop. That is the option this ADR's "Considered options"
listed as *rejected* — the code and the original framing diverged, and the **code is the
source of truth**.

We accept this deliberately. Stage 1 is a **recall net**: its only job is that every
real name label is covered by at least one candidate box; over-detection (nutrition
tables, barcodes, glare) is expected and is **Stage 3's burden to reject**, exactly the
"all the burden on false-positive rejection" cost noted above. The
brand-badge-as-anchor idea remains the most promising *future* discriminator (see ADR
0004's "genuinely more discriminative signal"), but it was never built and is out of
scope for the classical-techniques brief.

- **Stage-1 PASS** = no real name label is missed (not contained in any candidate). A
  miss means the label merged into an over-large blob or fell below the min-area drop.
- Over-detection is **not** a Stage-1 failure.

**Stage 1 is being pivoted (see ADR 0006).** The text-density recall net described here is the
*currently graded* Stage 1, but it merges glare-heavy frames into a single blob. ADR 0006 proposes
replacing the text-density signal with **dark-badge detection** (`detect_dark_badges`); that path is
under validation in `debug.ipynb` and, once promoted, supersedes the Stage-1 portions of this ADR.

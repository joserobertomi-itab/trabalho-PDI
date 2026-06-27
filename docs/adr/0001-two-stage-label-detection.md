# Two-stage label detection: locate the label cluster, then refine to the name label

Status: partially superseded by [ADR-0005](0005-final-output-label-cluster.md) for final output geometry.

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

# Proposal: detect full product label clusters

**Status:** proposed  
**Created:** 2026-06-26  
**Workflow:** `.agents/workflows/pipeline-change.md`  
**Skills:** `.cursor/skills/pdiseg-pipeline/SKILL.md`,
`.cursor/skills/pdiseg-debug-notebook/SKILL.md`,
`.cursor/skills/grill-with-docs/SKILL.md`

## Requirement Change

The previous task targeted the dark **product type badge** alone. That is no longer the
right output.

The detector should now emit at least one correct bounding box per frame. A final box is
valid when it contains the product-type label region and may include the adjacent
`SUPER FRANGO` brand badge in the same crop. A crop containing only the brand badge is a
false positive. A crop containing only random package text, crate borders, glare, or
nutrition tables is also a false positive.

Practical target:

- Prefer the full **label cluster**: brand badge + product type badge.
- Require product-type evidence in every emitted crop.
- Include the adjacent brand badge when it is visibly part of the same local label
  cluster.
- Do not emit brand-only detections.
- Aim for at least one valid box per frame. Emit multiple boxes only when every emitted
  box passes the same strict cluster validation.

## Explore Summary

### Current `pdiseg` implementation

The current detector already has:

- broad candidate generation from text density, dark luma, black-hat, edge-density, and
  optional DoG masks;
- scoring features for dark density, bright-on-dark text, opened background, edge
  density, extent, bimodality, and body overlap;
- final product-badge gate;
- fragment grouping;
- precision-first fallback;
- ranked dark-plaque refinement in `refine_to_name_label`.

This is useful, but it is optimized for the old target. The final stage still shrinks
selected clusters toward the product type badge. With the new requirement, this produces
incomplete boxes: the type may be present, but the adjacent brand/logo part of the
label cluster is often left out.

Observed current behavior on representative frames:

| Frame | Current issue |
|---|---|
| `93000068_Moela_Congelada/...12_40_55...jpg` | Final crops include `MOELA` and sometimes `FRANGO + MOELA`, but not consistently the full local cluster. |
| `93000088_Peito_Congelado/...12_33_33...jpg` | Final crops are tight around `PEITO`, missing cluster context. |
| `93000096_File_de_Coxas...16_03_11...jpg` | Final crops merge some type fragments, but still output partial product labels and one border/partial cluster risk. |
| `93000003_Asas_Resfriado_Selado/...14_14_11...jpg` | Current final gate emits zero labels despite many raw candidates, which violates the new "at least one" requirement. |

### Colleague implementation

The referenced `food-segmentation-9718bf11870bb6440da5781691463535d49fa034/`
implementation uses one monolithic OpenCV script, but the useful algorithmic ideas are
classical and portable to this repo's `scipy`/`skimage` stack:

- **DoG-like bright text on dark background**:
  Gaussian background estimate, then `(gray - background) > contrast_threshold` within
  a dark-background percentile mask.
- **Bold text grouping**:
  small opening to remove thin noise, then dilation + closing to group glyphs into a
  label-sized connected component.
- **Adaptive dark body gate**:
  adaptive threshold plus closing/opening to require a solid dark label body.
- **Bimodality check**:
  Otsu split in an expanded ROI to require a meaningful dark/light composition.
- **Multi-attempt relaxation**:
  start strict, then relax thresholds if no candidate is found.
- **Score based on bold text, body overlap, contrast, and bimodality**.
- **Final expansion**:
  expand the selected candidate so adjacent label context is included.

Most of these ideas already exist in `pdiseg`, but they are currently used to refine
toward the product type badge. For the new target, they should anchor the product type
and then expand/merge to the complete local label cluster.

Do not import OpenCV just for this. The repo intentionally uses `scipy`/`skimage`
(ADR-0002), and equivalent operators are already present or easy to add.

## Proposed Design

Change the final target from `name_label` to `label_cluster`.

```text
preprocess
  -> broad candidate masks
  -> score product-type badge candidates
  -> validate product-type evidence
  -> search adjacent brand/context region
  -> expand/merge to full label cluster
  -> validate cluster still contains product-type evidence
  -> select one primary cluster by default
```

### Key Design Decisions

1. Product-type evidence is mandatory.

   The detector may include `SUPER FRANGO`, but no output may be brand-only. This means
   the final gate should run on the product-type anchor inside the cluster, not only on
   the expanded bbox.

2. The final operation should expand, not shrink.

   `refine_to_name_label` should no longer be the last geometry operation for production
   output. Keep a product-type anchor internally, then expand to include the nearest
   connected/adjacent label context.

3. Default to one primary cluster per frame.

   The user's updated requirement is at least one correct bbox with no false positives.
   Emitting exactly the strongest validated cluster is safer than emitting every
   plausible partial label. Multiple outputs can remain possible behind a strict
   confidence gap, but they must not be the default path until audited.

4. Fallback must be anchored, not arbitrary.

   The colleague script's largest-dark-component fallback helps recall but can create
   false positives. In this repo, fallback should only use candidates with product-type
   evidence, then expand them to label clusters.

## Task Plan

### TASK-01: Update target terminology and tests

**Type:** docs + tests  
**Owns:** `CONTEXT.md`, `docs/adr/`, `docs/PIPELINE.md`, tests

Scope:

1. Update domain terminology:
   - `Name label` remains the product-type dark badge.
   - `Brand badge` remains `SUPER FRANGO`.
   - `Label cluster` becomes the production crop target: brand badge plus product-type
     badge when both are visible in the local cluster.
2. Add or update an ADR replacing the old "refine to name label" final-output decision.
3. Rename production wording where useful:
   - keep public API compatibility (`detect_name_labels`) if changing names is too large;
   - document that returned boxes are now cluster boxes containing the product type.
4. Add synthetic tests for the new target:
   - product-type badge alone passes as a fallback only when no brand context exists;
   - brand-only region fails;
   - brand above/adjacent to product type returns the union cluster;
   - two unrelated text regions do not merge.

Acceptance:

- Tests describe the new target unambiguously.
- Docs no longer instruct future agents to shrink final output to only the product type.

### TASK-02: Add product-type anchor extraction

**Type:** implementation  
**Owns:** `src/pdiseg/detection/postprocess.py`, optional new helper module, tests

Scope:

1. Preserve the current strict final product-badge gate as the product-type anchor gate.
2. Rename/internalize the current dark-plaque logic as `extract_product_anchor`.
3. Return both:
   - the selected cluster candidate;
   - the product-type anchor box inside it.
4. Do not emit the anchor directly as final output.

Acceptance:

- Synthetic dark product badge with bright glyphs yields a valid anchor.
- Brand-only/light text candidate has no product-type anchor.
- Current false-negative sample classes still expose at least one anchor candidate before
  final selection relaxation.

### TASK-03: Implement label-cluster expansion around anchor

**Type:** implementation  
**Owns:** `postprocess.py`, `config.py`, tests

Scope:

1. Add `expand_to_label_cluster(image, work, anchor_box, candidate_pool, config)`.
2. Search for adjacent context candidates around the product anchor:
   - above, upper-left, upper-right, or touching the anchor;
   - enough horizontal/vertical overlap;
   - gap bounded by anchor dimensions;
   - high text/edge density or bimodality.
3. Merge the anchor with the best adjacent context region.
4. If no adjacent brand/context is found, expand the anchor by a larger directional pad
   that follows the observed layout:
   - mostly upward and sideways;
   - modest downward pad;
   - clamp to frame.
5. Re-score the expanded cluster but validate product evidence using the original anchor.

Suggested config fields:

- `cluster_expand_up_frac`
- `cluster_expand_side_frac`
- `cluster_expand_down_frac`
- `cluster_context_max_gap_frac`
- `cluster_context_min_axis_overlap`
- `cluster_context_min_edge_density`
- `cluster_context_min_bimodal_score`

Acceptance:

- Product badge below/near a logo returns one union bbox containing both.
- Product badge with no detectable logo still returns a padded cluster bbox containing
  the product type.
- Logo-only candidates never pass because no product anchor exists.

### TASK-04: Port the colleague's strict DoG candidate path as an anchor source

**Type:** implementation  
**Owns:** `masks.py`, `candidates.py`, `config.py`, tests

Scope:

1. Turn the existing `dog_text_mask` path into a first-class product-anchor candidate
   source instead of an optional off-by-default experiment.
2. Use the colleague's strict-first parameters as default attempt 0:
   - `sigma ~= 45`
   - `contrast threshold ~= 24`
   - dark background percentile around `66`
   - opening around `3`
   - dilation/closing around `11`
3. Add controlled relaxation only when no valid anchor is found:
   - lower contrast threshold;
   - slightly higher dark-background percentile;
   - lower minimum bold text area;
   - never fall back to arbitrary largest dark component.
4. Keep implementation in `scipy`/`skimage`.

Acceptance:

- DoG candidates are visible in debug output.
- At least one valid anchor is found in the `Asas_Resfriado_Selado` sample that currently
  returns zero final boxes.
- No OpenCV dependency is added.

### TASK-05: Change final selection to one primary validated cluster

**Type:** implementation  
**Owns:** `postprocess.py`, `config.py`, runtime tests

Scope:

1. Add a config mode such as `primary_cluster_only: bool = True`.
2. Select the highest-scoring validated label cluster by default.
3. Emit additional clusters only if:
   - each has an anchor;
   - each passes cluster validation;
   - score is close to the top score;
   - boxes are clearly separate packages.
4. Set `max_labels_per_frame` as a hard safety cap, not as a target.

Acceptance:

- Default output is one valid cluster on cluttered frames unless multiple clusters are
  exceptionally clear.
- No output is brand-only.
- The detector does not emit six partial product-type crops from one frame.

### TASK-06: Replace fallback with anchored recall recovery

**Type:** implementation  
**Owns:** `postprocess.py`, tests

Scope:

1. Remove any fallback that emits based only on largest dark area or highest generic
   score.
2. Add recall recovery attempts:
   - strict anchor search;
   - relaxed DoG anchor search;
   - relaxed dark-body + bright-on-dark anchor search;
   - cluster expansion from the best surviving anchor.
3. If no product-type anchor exists, emit zero rather than a false positive.
4. For the released dataset, audit zero-output frames and tune only anchor recovery.

Acceptance:

- Synthetic no-label clutter emits zero.
- Dataset sample frames with visible product labels emit at least one cluster.
- No fallback can output crate borders, FPS overlay, or `SUPER FRANGO` alone.

### TASK-07: Audit against the new requirement

**Type:** validation/tooling  
**Owns:** debug scripts/docs

Scope:

1. Build an audit set with representative frames from all 18 classes.
2. Save contact sheets for:
   - source + overlay;
   - final crops;
   - product anchor inside final cluster.
3. Manually classify each output:
   - `valid_cluster`
   - `anchor_only_incomplete`
   - `brand_only`
   - `wrong_text`
   - `noise`
   - `missed_visible_label`
4. Record metrics:
   - frames with at least one valid cluster;
   - false positive count;
   - brand-only count;
   - incomplete anchor-only count;
   - zero-output frames.

Acceptance:

- Audit contact sheets show final boxes contain product type and usually the adjacent
  brand cluster.
- No `brand_only`, `wrong_text`, or `noise` outputs in the fixed audit set.
- Zero-output frames are manually checked and addressed through anchored recall recovery.

### TASK-08: Full verification

**Type:** verification  
**Owns:** whole pipeline

Scope:

1. Run focused tests:
   - `uv run pytest tests/test_run.py tests/test_detector_validation.py -q`
2. Run `make check`.
3. Run the audit from TASK-07.
4. If time allows, run the full dataset and compare:
   - total crops;
   - frames with zero crops;
   - false-positive categories;
   - incomplete anchor-only crops.

Acceptance:

- `make check` passes.
- Released dataset audit has at least one valid cluster per visible-label frame.
- No audited output is brand-only or unrelated text/noise.

## Implementation Notes

- Do not replace `pdiseg` with the colleague's monolithic script.
- Do port the useful signals from that script:
  - DoG bright-on-dark text;
  - adaptive dark body;
  - strict-first relaxation;
  - bimodality;
  - NMS using overlap over the smaller box;
  - final context expansion.
- Keep the project stack as `numpy` + `scipy` + `skimage`; no OpenCV dependency.
- Keep class folder names out of detector logic.
- Keep crops from original frame geometry.

## Non-goals

- OCR or product classification.
- Per-class thresholds keyed on folder names.
- Logo-only detection.
- Arbitrary largest-dark-component fallback.
- Emitting many partial labels just to increase crop count.


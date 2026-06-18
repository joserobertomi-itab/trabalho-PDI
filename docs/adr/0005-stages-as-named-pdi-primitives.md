# Decompose detection stages into named, individually-inspectable PDI primitives

To grade a classical-PDI pipeline honestly, a human evaluator must see the *intermediate
rasters* each stage produces (text mask, closed binary, labeled components, the per-cluster
Otsu dark-mask) — not only the final bounding boxes. The monolithic stage functions
(`detect_clusters`, `refine_to_name_label`) computed these internally and discarded them.

We split each stage into small, public, single-operation primitives — e.g.
`text_mask(img)`, `close_mask(mask)`, `boxes_from_components(mask)`, and the Otsu refine
split — and recompose them inside the stage functions. The graded pipeline and the debug
notebook compose the **same** primitives, so the debug view is honest by construction: it
renders exactly what the grader runs, never a parallel re-implementation that could drift.

## Considered options

- **Named primitives (chosen)** — each PDI operation becomes individually named, unit-
  tested, and renderable. More refactoring of graded code, but every step gains a test and
  the inspection view cannot diverge from the pipeline.
- **Optional `trace` out-param** (rejected) — stages stay monolithic and fill a trace
  object on request. Less refactor, but stages stay coarse and the view depends on the
  trace being threaded correctly.
- **Parallel inspection module** (rejected) — re-derive intermediates beside the graded
  code. Zero changes to graded code, but the debug view could silently drift from the real
  pipeline — the exact "rendering a lie" failure mode we refuse.

## Consequences

- The public surface grows (the primitives are public), but each is a thin, testable PDI
  operation, which suits the educational/evaluation brief.
- The debug notebook (`debug.ipynb`, a dev-only tool) imports these primitives directly;
  it is never part of the graded Docker image (`--no-dev`).

## Caveat — the notebook's headline Stage 1 is the *proposed* path, not the graded one

The "renders exactly what the grader runs" guarantee holds **per primitive** (each `dark_mask`,
`close_mask`, etc. rendered in the notebook is the real pipeline code). But the notebook's headline
**Stage 1 composition** is `detect_dark_badges` (the dark-badge pivot, ADR 0006), whereas the graded
`detect_name_labels` still composes `detect_clusters` (text-density). So for Stage 1 specifically the
notebook shows the *proposed* composition, not the graded one — deliberately, to validate ADR 0006.
The text-density path is still rendered alongside (red-dashed contrast) so the divergence is visible.
This caveat dissolves once ADR 0006 is promoted and both compositions coincide.

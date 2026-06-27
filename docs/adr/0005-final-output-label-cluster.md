# Final output is the validated label cluster, not the isolated name-label anchor

The production segmentation target is now the **label cluster**: the product-type name
label plus adjacent brand/context when it belongs to the same local package label. The
name label remains mandatory evidence, but it is an internal **product anchor**, not the
final crop geometry.

This supersedes the final-output part of ADR-0001, which refined a detected cluster down
to the name label. The detector still uses `refine_to_name_label`, but only to extract
and validate the product anchor before expanding back to the local label cluster.

## Decision

- Keep the public API name `detect_name_labels` for compatibility.
- Return cluster boxes that contain a validated product anchor.
- Show selected product anchors as yellow boxes in overlays and final clusters as green
  boxes.
- Prefer one primary cluster per frame by default (`primary_cluster_only=True`).
- Use relaxed recall only after strict anchor search fails; never emit a crop from an
  arbitrary largest dark component.

## Consequences

- A crop containing only `SUPER FRANGO` is invalid because it has no product anchor.
- A product badge without visible adjacent brand/context remains valid and is returned
  with directional padding.
- Multiple crops are possible only when explicitly configured and each crop has its own
  product anchor.
- Debug/audit output should inspect both anchor and cluster geometry.

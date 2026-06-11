# Self-contained detector: detect on the preprocessed frame, crop from the original

`detect_name_labels` takes a raw frame and preprocesses it **internally** (median +
histogram equalization + FPS mask) before running the two-stage detection, but the
boxes it returns index the **original** frame, and `run` crops from that original.
So detection benefits from preprocessing while the delivered segmented image stays
faithful to the source — equalized/masked pixels never reach the output crop.

## Considered options

- **(A) Self-contained detector (chosen)** — preprocessing lives inside
  `detect_name_labels`; `run` exposes a single `detector` seam and keeps the
  original frame for cropping. Deepest module: "a frame in, its name labels out."
- **(B) Separate preprocessor seam** — `run` preprocesses and passes the working
  image to the detector. Rejected: the "preprocess first" obligation leaks to every
  caller and the detect/crop contract stays smeared across `run` and the detector.

## Consequences

- A future reader will see the detector preprocess yet the crop come from the
  unprocessed original and may try to "fix" the apparent inconsistency by cropping
  the preprocessed image. That would silently degrade the deliverable — don't.
- `preprocess` stays public for inspection (`dump_preprocessed`, calibration #6),
  it is just no longer a separate seam in `run`.
- The contract is pinned by `test_run_crops_from_the_original_frame` (crop is the
  original at the detector's box) and `test_detect_name_labels_finds_a_label_on_a_raw_frame`
  (detector accepts a raw frame).

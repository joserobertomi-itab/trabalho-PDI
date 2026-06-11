# Use the scikit-image / scipy.ndimage stack, not OpenCV

The pipeline is built on **NumPy + scipy.ndimage + scikit-image** (with imageio /
matplotlib for IO and visualisation), not OpenCV. This is surprising for a classical
computer-vision project, where OpenCV is the default. The reason is the assignment
constraint: only techniques **presented in class** may be used, and the course taught
this exact stack — the canonical class example (`skimage.data.coins`) runs
`threshold_otsu` → `binary_closing`/`binary_opening` → `area_opening` →
`ndimage.label` → optional `watershed`, which is essentially our pipeline. Staying on
the taught stack keeps every operation defensible during evaluation and avoids pulling
in any library that performs AI-based segmentation (explicitly forbidden).

## Consequences

- Operators are chosen from what was demonstrated in class (thresholding, binary
  morphology, connected components, distance-transform watershed, SLIC). Grayscale
  top-hat / black-hat morphology was **not** taught and is deliberately avoided.
- Porting to OpenCV later would mean re-validating every step against the taught
  techniques, so the choice is treated as fixed for this assignment.

# Review viewer directory contract

The review viewer is a **read-only** tool outside the graded pipeline. It never runs
detection; it renders overlays and crops from artifacts already on disk.

## Inputs

| Path | Required | Contents |
|------|----------|----------|
| `dataset/` (or `--dataset`) | Yes | Source frames: `dataset/<Class>/<image>.jpg` |
| `calibration/` (or `--calibration`) | Yes | `boxes.json`, `stats.csv`, optional sample `*_overlay.png` |
| `result/` (or `--result`) | No | Segmentation crops: `result/<Class>/<stem>_segmentada_<N>.png` |

Generate `boxes.json` and `stats.csv` with:

```sh
make calibrate
# or: uv run pdiseg-calibrate data/Train_and_Validation calibration
```

Generate `result/` with:

```sh
make run
# or: uv run pdiseg data/Train_and_Validation result
```

## `boxes.json` shape

Single JSON object keyed by **source path relative to the dataset root** (POSIX
slashes), e.g. `Peito_Congelado/img001.jpg`:

```json
{
  "Peito_Congelado/img001.jpg": {
    "candidates": [[x, y, w, h]],
    "kept": [[x, y, w, h]],
    "labels": [[x, y, w, h]]
  }
}
```

Each value is a list of `[x, y, width, height]` boxes in pixel coordinates on the
**original** (un-preprocessed) frame.

## Launch

```sh
make review
# or:
uv run pdiseg-review \
  --dataset data/Train_and_Validation \
  --calibration calibration \
  --result result \
  --port 8765
```

Open `http://127.0.0.1:8765/`.

## Degraded modes

- Missing `boxes.json` entry for a frame → source and disk crops still shown; overlay skipped.
- Missing `result/` crop → crop rendered on the fly from the green label box when metadata exists.
- Missing source image → frame listed with a notice; other artifacts still shown when available.

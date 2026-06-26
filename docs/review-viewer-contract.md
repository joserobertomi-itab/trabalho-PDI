# Review viewer

Web tool to inspect source frames, overlays, and crops. Does **not** run the detector.

## Folders

| Folder | Required | Contents |
|--------|----------|----------|
| `dataset/` | yes | Original images |
| `calibration/` | yes | `boxes.json`, `stats.csv` |
| `result/` | no | `*_segmented_N.png` |

Generate calibration:

```sh
make calibrate
```

Generate result:

```sh
make run
```

## `boxes.json`

Keys are paths relative to the dataset, e.g. `Peito_Congelado/img001.jpg`:

```json
{
  "Peito_Congelado/img001.jpg": {
    "candidates": [[x, y, w, h]],
    "kept": [[x, y, w, h]],
    "labels": [[x, y, w, h]]
  }
}
```

Coordinates are `[x, y, width, height]` on the original image.

## Start server

```sh
make review
```

or

```sh
uv run pdiseg-review --dataset data/Train_and_Validation --calibration calibration --result result
```

Open http://127.0.0.1:8765/

## Missing files

- No `boxes.json` entry → shows source and on-disk crops, no overlay.
- No PNG in `result/` → renders crop from green box metadata when available.
- No source image → lists the frame with a warning.

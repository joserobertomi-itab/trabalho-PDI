# Source layout

Python package lives under ``src/pdiseg/`` (src layout). The installable name remains **pdiseg**.

## Package map

```
src/pdiseg/
├── __init__.py          # Public re-exports (stable API)
├── __main__.py          # python -m pdiseg
├── cli/                 # Console scripts (segment, calibrate, review)
├── core/                # BBox geometry, crop, overlays
│   ├── boxes.py
│   └── imaging.py
├── io/                  # Dataset discovery and load_image
│   └── dataset.py
├── detection/           # Classical CV pipeline stages
│   ├── config.py
│   ├── preprocess.py
│   ├── masks.py
│   ├── candidates.py
│   ├── scoring.py
│   ├── postprocess.py
│   └── detector.py
├── runtime/             # Batch I/O loop over dataset
│   └── pipeline.py
├── calibration/         # Overlays + boxes.json export
│   └── service.py
├── debug/               # Notebook visualization (not graded output)
│   └── viz.py
└── review/              # FastAPI review viewer
    ├── model.py
    ├── server.py
    └── static/
```

## Data flow

1. **io.dataset** — `find_source_images`, `load_image`
2. **detection.preprocess** — CLAHE work image; mask FPS burn-in
3. **detection.candidates** — masks → raw boxes (text density from gray)
4. **detection.scoring** + **detection.postprocess** — rank, NMS, refine
5. **runtime.pipeline** — write crops to `result/<class>/`

## Import conventions

- Application code and tests: `import pdiseg` or `from pdiseg.detection.config import DetectionConfig`
- Inside **detection/**: relative imports between siblings (``.config``, ``.masks``, …)
- Cross-layer: absolute `pdiseg.core.*`, `pdiseg.io.*`

## Tooling

| Tool | Path |
|------|------|
| pytest `pythonpath` | `src` |
| ruff / mypy | `src/pdiseg`, `tests` |
| hatch `dev-mode-dirs` | `src` |

## CLIs

| Command | Module |
|---------|--------|
| `pdiseg` | `pdiseg.cli.segment` |
| `pdiseg-calibrate` | `pdiseg.cli.calibrate` |
| `pdiseg-review` | `pdiseg.cli.review` |

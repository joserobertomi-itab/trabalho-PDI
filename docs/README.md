# Project documentation

All committed documentation is in **English**. Portuguese appears only in dataset class folder names (they mirror the data) and in `enunciado.pdf` (original assignment brief).

## Start here

| Document | Contents |
|----------|----------|
| [requirements.md](../requirements.md) | Assignment objective, constraints, grading |
| [CONTEXT.md](../CONTEXT.md) | Glossary, dataset summary, I/O pattern |
| [PIPELINE.md](./PIPELINE.md) | **Full technical reference** — I/O, preprocessing, every detection stage, config, libraries |
| [src/ARCHITECTURE.md](./src/ARCHITECTURE.md) | Package layout and import conventions |
| [docker-compose.md](./docker-compose.md) | Container delivery and volumes |
| [review-viewer-contract.md](./review-viewer-contract.md) | Review UI folders and `boxes.json` schema |
| [PIPELINE_IMPROVEMENTS.md](./PIPELINE_IMPROVEMENTS.md) | Historical tuning metrics and reference integration log |
| [adr/](./adr/) | Architecture decision records |

## Quick links

- Run locally: [README.md §4–6](../README.md)
- Debug notebook: `debug.ipynb`
- Agent harness: [agents/harness.md](./agents/harness.md)
- Reference scripts: `reference_segment_label.py`, `reference_label_segmentation.ipynb`

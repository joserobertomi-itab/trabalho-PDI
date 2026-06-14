## Running with Docker Compose (submission path)

This is the **Docker Compose delivery** for issue #7. The frozen pipeline reads
`dataset/<Class>/...` and writes `resultado/<Class>/*_segmentada_N.png`.

### Quick start

```sh
mkdir -p resultado calibration
ln -sfn data/Train_and_Validation dataset

cp .env.example .env   # optional
docker compose up --build
# equivalent: make docker-up
```

Create output directories before the first run. The container **entrypoint** automatically
`chown`s writable bind mounts (`resultado/`, `calibration/`) to the app user (uid 1000),
so root-owned folders from earlier runs are fixed on startup.

On success, crops appear under `./resultado/`. The container exits when the walk
finishes; Compose stops automatically.

### All Docker services

| Command | What it does |
|---------|----------------|
| `docker compose up --build` | **Graded pipeline** — `dataset/` → `resultado/` |
| `docker compose --profile tools run --rm calibrate` | Calibration bundle (`boxes.json`, `stats.csv`) |
| `docker compose --profile tools up review` | Review viewer at http://localhost:8765/ |

Make shortcuts: `make docker-up`, `make docker-calibrate`, `make docker-review`.

Override mounts without editing the file:

```sh
DATA=./my_unseen_set OUT=./resultado docker compose up --build
```

### Layout contract (including unseen evaluation)

```
dataset/                          resultado/
├── Peito_Congelado/              ├── Peito_Congelado/
│   └── frame.jpg                 │   └── frame_segmentada_1.png
└── Moela/                        └── Moela/
    └── ...                           └── ...
```

Drop a new image set into `dataset/` with the same folder structure — **no code
changes** required.

### Security / ops notes

- Container runs as non-root user `pdiseg` (uid 1000).
- Root filesystem is read-only; only mounted output dirs are writable.
- Dataset mount is read-only.
- End-to-end smoke test: `make docker-smoke` (synthetic `dataset/` in `.docker-smoke/`).

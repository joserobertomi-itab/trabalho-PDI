## Running with Docker Compose (submission path)

This is the **Docker Compose delivery** for issue #7. The frozen pipeline reads
`dataset/<Class>/...` and writes `result/<Class>/*_segmentada_N.png`.

### Quick start

```sh
mkdir -p result calibration
cp .env.example .env

docker compose up --build
# equivalent: make docker-up
```

Create output directories before the first run. The container **entrypoint** automatically
`chown`s writable bind mounts (`result/`, `calibration/`) to the app user (uid 1000),
so root-owned folders from earlier runs are fixed on startup. Read-only mounts (the review
viewer) skip this step.

### Gentle on the host (laptops)

Defaults in `.env.example` cap the batch job so the desktop stays usable:

| Setting | Default | Effect |
|---------|---------|--------|
| `DOCKER_CPUS` | `1.0` | At most one CPU core for pipeline/calibrate |
| `DOCKER_MEMORY` | `1536m` | RAM ceiling (~1.5 GiB) |
| `OMP_NUM_THREADS` | `1` | NumPy/SciPy use a single thread (avoids spikes) |
| `NICE_LEVEL` | `10` | Lower CPU priority vs other apps |
| `IONICE_CLASS` | `3` | Disk I/O only when the disk is idle |

900 frames will take **longer**, but the PC should not freeze. On a weak machine, try:

```env
DOCKER_CPUS=0.75
DOCKER_MEMORY=1024m
NICE_LEVEL=15
```

On success, crops appear under `./result/`. The container exits when the walk
finishes; Compose stops automatically.

### All Docker services

| Command | What it does |
|---------|----------------|
| `docker compose up --build` | **Graded pipeline** — `dataset/` → `result/` |
| `docker compose --profile tools run --rm calibrate` | Calibration bundle (`boxes.json`, `stats.csv`) |
| `docker compose --profile tools up review` | Review viewer at http://localhost:8765/ |

Make shortcuts: `make docker-up`, `make docker-calibrate`, `make docker-review`.

Override mounts without editing the file:

```sh
DATA=./my_unseen_set OUT=./result docker compose up --build
```

### Layout contract (including unseen evaluation)

```
dataset/                          result/
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

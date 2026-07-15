## Docker Compose

### Production image

Use this path when the professor should not build the image locally:

```sh
mkdir -p result calibration docs/report
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up pipeline
docker compose -f docker-compose.prod.yml --profile tools run --rm --no-deps calibrate
docker compose -f docker-compose.prod.yml --profile tools run --rm --no-deps recognize
docker compose -f docker-compose.prod.yml --profile tools run --rm --no-deps report
docker compose -f docker-compose.prod.yml --profile review up review
```

The image is published by CI to:

```text
ghcr.io/joserobertomi-itab/trabalho-pdi:latest
```

`docker-compose.prod.yml` has the runtime environment embedded directly in the file, so
it does not require `.env`. It expects the dataset at `data/Train_and_Validation/`.

After the first publish, make the GitHub Package public in the package settings so the
professor can pull without authenticating.

### Delivery (issue #7)

Chosen vehicle: **Docker Compose**. Graded path is pipeline only:

```sh
mkdir -p resultado
DATA=./dataset OUT=./resultado docker compose up --build pipeline
```

Or with Make:

```sh
mkdir -p result calibration docs/report
cp .env.example .env
make docker-up DATA=dataset OUT=resultado
```

Full optional T1→T2 tools chain (ordered):

```sh
make docker-tools
```

### Review viewer (issue #8)

Read-only web UI — does **not** re-run detection:

```sh
make docker-calibrate   # writes boxes.json / overlays
make docker-review      # http://127.0.0.1:8765/
```

```sh
docker compose --profile review up --build review
```

### Local build

```sh
mkdir -p result calibration docs/report
cp .env.example .env
make docker-up
```

This runs with Docker settings from `.env`: workers, CPU cap, and automatic GPU
selection when NVIDIA Docker is available.

| Target | What it runs |
|--------|----------------|
| `make docker-up` | T1 pipeline only |
| `make docker-calibrate` | overlays + `boxes.json` |
| `make docker-templates` | bootstrap `templates/` |
| `make docker-recognize` | T2 recognition → `recognition.csv` |
| `make docker-report` | T2 PDF reports → `docs/report/` |
| `make docker-review` | review viewer (profile `review`) |
| `make docker-tools` | ordered: pipeline → calibrate → recognize → report |
| `make docker-smoke` | synthetic E2E smoke |

Do not run heavy services in parallel on a notebook.

Run in chunks:

```sh
make docker-up MAX_IMAGES=100 OFFSET=0
make docker-up MAX_IMAGES=100 OFFSET=100
```

Force CPU fallback:

```sh
make docker-up DOCKER_GPU=off PDISEG_BACKEND=cpu
```

### Named volumes (pipeline + review)

In `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
make docker-up
make docker-calibrate
make docker-review
make docker-export
```

Direct GPU override:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build pipeline
```

| `OUT` | Use |
|-------|-----|
| `./result` / `./resultado` | Delivery — professor sees folder on host |
| `pdiseg-result` | Same volume shared by pipeline and review |

### Services

| Service | Profile | Role |
|---------|---------|------|
| `pipeline` | (default) | Segment → `/data/output` |
| `calibrate` | `tools` | `boxes.json`, `stats.csv`, overlays |
| `recognize` | `tools` | T2 → `recognition.csv` + sweep CSV |
| `report` | `tools` | T2 PDFs → `/data/report` |
| `templates` | `bootstrap` | rebuild `templates/` from dataset |
| `review` | `review` | http://localhost:8765/ read-only UI |

Plain `docker compose up` starts **only** `pipeline`.

### Variables

`DATA`, `OUT`, `CALIB`, `TEMPLATES`, `REPORT`, `LIMIT`, `MAX_IMAGES`, `OFFSET`,
`PROGRESS_EVERY`, `THREADS`, `WORKERS`, `PDISEG_BACKEND`, `DOCKER_GPU`,
`DOCKER_CPUS`, `DOCKER_MEMORY`, `DOCKER_NICE`, `PORT` — see README / `.env.example`.

### New dataset (complementary evaluation)

Same folder structure under `dataset/`; re-run without code changes.

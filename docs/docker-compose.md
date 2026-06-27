## Docker Compose

### Delivery

```sh
mkdir -p result calibration
cp .env.example .env
make docker-up
make docker-calibrate
```

The Make targets pass conservative runtime defaults to Docker:

- `THREADS=1` limits NumPy/SciPy/OpenBLAS worker threads.
- `DOCKER_NICE=5` lowers container process priority enough to keep a notebook usable.
- `PROGRESS_EVERY=10` or `25` prints progress so long runs do not look frozen.

Run the full dataset:

```sh
make docker-up PROGRESS_EVERY=10
make docker-calibrate PROGRESS_EVERY=10
```

Run in chunks:

```sh
make docker-up MAX_IMAGES=100 OFFSET=0
make docker-up MAX_IMAGES=100 OFFSET=100
make docker-up MAX_IMAGES=100 OFFSET=200
```

If the machine is still busy:

```sh
make docker-up DOCKER_NICE=10
make docker-calibrate DOCKER_NICE=10
```

### Named volumes (pipeline + review)

In `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
THREADS=1 DOCKER_NICE=5 PROGRESS_EVERY=10 docker compose up --build pipeline
THREADS=1 DOCKER_NICE=5 PROGRESS_EVERY=10 docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
make docker-export
```

| `OUT` | Use |
|-------|-----|
| `./result` | Delivery — professor sees folder on host |
| `pdiseg-result` | Same volume shared by pipeline and review |

### Services

| Service | Command | Role |
|---------|---------|------|
| `pipeline` | `docker compose up pipeline` | Segment → `/data/output` |
| `calibrate` | `docker compose --profile tools run --rm calibrate` | `boxes.json`, `stats.csv` |
| `review` | `docker compose --profile tools up review` | http://localhost:8765/ |

### Make

```sh
make docker-up
make docker-calibrate
make docker-review
make docker-export
make docker-smoke
```

Variables: `DATA`, `OUT`, `CALIB`, `LIMIT`, `MAX_IMAGES`, `OFFSET`, `PROGRESS_EVERY`, `THREADS`, `DOCKER_NICE`, `PORT` — see README.

### New dataset (complementary evaluation)

Same folder structure under `dataset/`; re-run without code changes.

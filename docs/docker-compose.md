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
- `WORKERS=6` processes images concurrently.
- `DOCKER_CPUS=6.0` applies a hard Docker CPU cap; on a 12-logical-CPU notebook this is 50%.
- `DOCKER_MEMORY=4g` caps container memory.
- `DOCKER_GPU=auto` makes `make docker-up` and `make docker-calibrate` use `docker-compose.gpu.yml` when NVIDIA Docker is available.
- `PDISEG_BACKEND=auto` uses the CuPy/CUDA mask backend when the GPU image can access a CUDA device, otherwise CPU.
- `DOCKER_NICE=10` lowers container process priority.
- `PROGRESS_EVERY=25` prints progress so long runs do not look frozen.

Run the full dataset:

```sh
make docker-up
make docker-calibrate
```

Run in chunks:

```sh
make docker-up MAX_IMAGES=100 OFFSET=0
make docker-up MAX_IMAGES=100 OFFSET=100
make docker-up MAX_IMAGES=100 OFFSET=200
```

If the machine is still busy:

```sh
make docker-up DOCKER_CPUS=4.0 WORKERS=4
make docker-calibrate DOCKER_CPUS=4.0 WORKERS=4
```

Force CPU fallback:

```sh
make docker-up DOCKER_GPU=off PDISEG_BACKEND=cpu
make docker-calibrate DOCKER_GPU=off PDISEG_BACKEND=cpu
```

### Named volumes (pipeline + review)

In `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
THREADS=1 WORKERS=6 PDISEG_BACKEND=auto DOCKER_CPUS=6.0 DOCKER_MEMORY=4g DOCKER_NICE=10 PROGRESS_EVERY=25 docker compose up --build pipeline
THREADS=1 WORKERS=6 PDISEG_BACKEND=auto DOCKER_CPUS=6.0 DOCKER_MEMORY=4g DOCKER_NICE=10 PROGRESS_EVERY=25 docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
make docker-export
```

Direct GPU override:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build pipeline
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile tools run --rm calibrate
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

Variables: `DATA`, `OUT`, `CALIB`, `LIMIT`, `MAX_IMAGES`, `OFFSET`, `PROGRESS_EVERY`, `THREADS`, `WORKERS`, `PDISEG_BACKEND`, `DOCKER_GPU`, `DOCKER_CPUS`, `DOCKER_MEMORY`, `DOCKER_NICE`, `PORT` — see README.

### New dataset (complementary evaluation)

Same folder structure under `dataset/`; re-run without code changes.

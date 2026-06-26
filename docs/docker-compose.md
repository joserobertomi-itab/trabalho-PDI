## Docker Compose

### Delivery

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

### Named volumes (pipeline + review)

In `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
docker compose up --build
docker compose --profile tools run --rm calibrate
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
| `pipeline` | `docker compose up` | Segment → `/data/output` |
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

Variables: `DATA`, `OUT`, `CALIB`, `LIMIT`, `PORT` — see README.

### New dataset (complementary evaluation)

Same folder structure under `dataset/`; re-run without code changes.

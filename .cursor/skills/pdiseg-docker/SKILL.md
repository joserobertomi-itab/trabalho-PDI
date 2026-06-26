---
name: pdiseg-docker
description: >-
  Build and run pdiseg via Docker Compose and Makefile targets. Use when
  dockerizing, fixing compose profiles, delivery smoke tests, or export scripts.
---

# Docker delivery

## Makefile targets

| Target | Action |
|--------|--------|
| `docker-build` | Build `pdiseg:latest` |
| `docker-up` | Run pipeline |
| `docker-calibrate` | Profile `tools` → calibrate |
| `docker-review` | Profile `tools` → review UI |
| `docker-smoke` | E2E synthetic smoke |
| `docker-export` | Copy named volumes to host |

## Compose syntax

```sh
docker compose --profile tools up --build calibrate
```

Service names: `pipeline`, `calibrate`, `review` (not `calibration`).

## Env

Copy `.env.example` → `.env` for `OUT`, `CALIB`, `DATA` if permission issues on review.

Full reference: `.agents/workflows/docker-and-make.md`, `docs/docker-compose.md`.

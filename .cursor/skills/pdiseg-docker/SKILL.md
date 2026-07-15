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
| `docker-build` | Build `pdiseg:latest` (or GPU image when selected) |
| `docker-up` | T1 pipeline only (graded path, issue #7) |
| `docker-calibrate` | Profile `tools` → calibrate (`--no-deps`) |
| `docker-templates` | Profile `bootstrap` / tools → rebuild templates |
| `docker-recognize` | Profile `tools` → T2 recognize |
| `docker-report` | Profile `tools` → T2 PDF reports |
| `docker-review` | Profile `review` → read-only review UI (issue #8) |
| `docker-tools` | Ordered chain: pipeline → calibrate → recognize → report |
| `docker-smoke` | E2E synthetic smoke |
| `docker-export` | Copy named volumes to host |
| `prod-*` | Same flow against `docker-compose.prod.yml` (GHCR) |

## Compose syntax

```sh
# Graded segmentation only
docker compose up --build pipeline
DATA=./dataset OUT=./resultado docker compose up --build pipeline

# Tools chain (ordered)
docker compose --profile tools up --build

# One-shot tool (do not re-run pipeline)
docker compose --profile tools run --rm --no-deps recognize

# Review viewer (does not run detection)
docker compose --profile review up --build review
```

Service names: `pipeline`, `calibrate`, `templates`, `recognize`, `report`, `review`.

Profiles:

- default → `pipeline`
- `tools` → `calibrate`, `recognize`, `report` (+ depends_on chain)
- `bootstrap` → `templates`
- `review` → `review`

## Env

Copy `.env.example` → `.env` for `DATA`, `OUT`, `CALIB`, `TEMPLATES`, `REPORT`,
`PORT`, and review UID/GID if permission issues.

Full reference: `.agents/workflows/docker-and-make.md`, `docs/docker-compose.md`.

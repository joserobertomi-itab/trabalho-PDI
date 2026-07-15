# Workflow: Docker and Make

## Local (preferred for dev)

```sh
make setup
make run              # DATA → OUT
make calibrate        # LIMIT overlays per class
make recognize        # T2 on OUT crops
make report           # T2 PDFs → docs/report/
make review           # browser viewer
make check            # lint + mypy + test
```

## Docker

```sh
# Pipeline only (graded delivery — issue #7)
docker compose up --build pipeline
DATA=./dataset OUT=./resultado docker compose up --build pipeline

# Tools profile — flag BEFORE subcommand
docker compose --profile tools run --rm --no-deps calibrate
docker compose --profile tools run --rm --no-deps recognize
docker compose --profile tools run --rm --no-deps report

# Full ordered tools chain
docker compose --profile tools up --build

# Review viewer (issue #8 — profile review, not tools)
docker compose --profile review up review
```

Or Make: `docker-up`, `docker-calibrate`, `docker-recognize`, `docker-report`,
`docker-review`, `docker-tools`.

## Variables

| Var | Default |
|-----|---------|
| DATA | data/Train_and_Validation |
| OUT | result |
| CALIB | calibration |
| TEMPLATES | templates |
| REPORT | docs/report |
| LIMIT | 9999 |
| PORT | 8765 |

## Cleanup

```sh
docker compose --profile tools --profile review down -v --rmi all --remove-orphans
```

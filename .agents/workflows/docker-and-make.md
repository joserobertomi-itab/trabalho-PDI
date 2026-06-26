# Workflow: Docker and Make

## Local (preferred for dev)

```sh
make setup
make run              # DATA → OUT
make calibrate        # LIMIT overlays per class
make review           # browser viewer
make check            # lint + mypy + test
```

## Docker

```sh
# Pipeline only (default compose)
docker compose up --build pipeline

# Tools profile — flag BEFORE subcommand
docker compose --profile tools build calibrate
docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
```

Or: `make docker-up`, `make docker-calibrate`, `make docker-review`.

## Variables

| Var | Default |
|-----|---------|
| DATA | data/Train_and_Validation |
| OUT | result |
| CALIB | calibration |
| LIMIT | 3 |
| PORT | 8765 |

## Cleanup

```sh
docker compose --profile tools down -v --rmi all --remove-orphans
```

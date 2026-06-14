# Poultry-packaging name-label segmentation — local tasks (uv-backed).
# Override paths on the command line, e.g.: make run DATA=dataset OUT=resultado

UV      ?= uv
PY      := $(UV) run
DATA    ?= data/Train_and_Validation
OUT     ?= resultado
CALIB   ?= calibration
LIMIT   ?= 3

.DEFAULT_GOAL := help

.PHONY: help setup sync lock test lint format typecheck check ci run calibrate docker-build docker-run docker-calibrate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: sync  ## Install locked runtime + dev dependencies with uv

sync:  ## Sync the virtualenv from uv.lock (includes dev extras)
	$(UV) sync --extra dev

lock:  ## Refresh uv.lock after pyproject.toml dependency changes
	$(UV) lock

test:  ## Run the test suite with coverage
	$(PY) pytest --cov=pdiseg --cov-report=term-missing -q

lint:  ## Lint with ruff
	$(PY) ruff check pdiseg tests

format:  ## Format code with ruff
	$(PY) ruff format pdiseg tests
	$(PY) ruff check --fix pdiseg tests

typecheck:  ## Static type check with mypy
	$(PY) mypy pdiseg

check: lint typecheck test  ## Run lint, typecheck, and tests

ci: sync check  ## Full local CI gate (sync + check)

run:  ## Segment the base into $(OUT)/<Class>/<source>_segmentada_<N>.png
	$(PY) pdiseg $(DATA) $(OUT)

calibrate:  ## Run calibration harness: overlays + stats.csv into $(CALIB)/
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT)

docker-build:  ## Build the production Docker image
	docker compose build segment

docker-run: docker-build  ## Segment via Docker Compose (mounts DATA and OUT)
	docker compose run --rm segment

docker-calibrate: docker-build  ## Calibrate via Docker Compose (mounts DATA and CALIB)
	docker compose run --rm calibrate

clean:  ## Remove generated outputs and caches
	rm -rf $(OUT) $(CALIB) .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

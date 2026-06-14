# Poultry-packaging name-label segmentation — local tasks (uv-backed).
# Override paths on the command line, e.g.: make run DATA=dataset OUT=resultado

UV      ?= uv
PY      := $(UV) run
DATA    ?= data/Train_and_Validation
OUT     ?= resultado
CALIB   ?= calibration
LIMIT   ?= 3
PORT    ?= 8765
COMPOSE ?= docker compose

.DEFAULT_GOAL := help

.PHONY: help setup sync lock test lint format typecheck check ci run calibrate review \
	docker-build docker-up docker-calibrate docker-review docker-smoke clean

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

calibrate:  ## Run calibration harness: overlays + boxes.json + stats.csv into $(CALIB)/
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT)

review:  ## Launch the read-only review viewer on http://127.0.0.1:$(PORT)/
	$(PY) pdiseg-review --dataset $(DATA) --calibration $(CALIB) --resultado $(OUT) --port $(PORT)

docker-build:  ## Build the production Docker image (pipeline + tools)
	$(COMPOSE) build pipeline

docker-up: docker-build  ## Submission path: docker compose up (DATA → resultado/)
	DATA=./$(DATA) OUT=./resultado $(COMPOSE) up --build pipeline

docker-calibrate: docker-build  ## Write calibration/ via Docker (boxes.json + stats)
	DATA=./$(DATA) $(COMPOSE) --profile tools run --rm calibrate

docker-review: docker-build  ## Run the review viewer in Docker on port $(PORT)
	DATA=./$(DATA) OUT=./resultado CALIB=./calibration PORT=$(PORT) \
		$(COMPOSE) --profile tools up review

docker-smoke:  ## E2E Docker Compose smoke test (synthetic dataset/)
	bash scripts/docker-smoke.sh

clean:  ## Remove generated outputs and caches
	rm -rf $(OUT) $(CALIB) .docker-smoke .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

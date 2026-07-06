-include .env

UV      ?= uv
PY      := $(UV) run
HOST_CPUS := $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
DEFAULT_HALF_CPUS := $(shell awk 'BEGIN { n=$(HOST_CPUS); v=n/2; if (v<1) v=1; printf "%.1f", v }')
DEFAULT_WORKERS := $(shell awk 'BEGIN { n=$(HOST_CPUS); v=int(n/2); if (v<1) v=1; if (v>8) v=8; print v }')
DATA    ?= data/Train_and_Validation
OUT     ?= result
CALIB   ?= calibration
LIMIT   ?= 9999
MAX_IMAGES ?=
OFFSET ?= 0
PROGRESS_EVERY ?= 25
THREADS ?= 1
WORKERS ?= $(DEFAULT_WORKERS)
PDISEG_BACKEND ?= auto
PDISEG_BACKEND_LOG ?= 1
DOCKER_GPU ?= auto
DOCKER_CPUS ?= $(DEFAULT_HALF_CPUS)
DOCKER_MEMORY ?= 4g
DOCKER_NICE ?= 10
PORT    ?= 8765
COMPOSE ?= docker compose
DOCKER_GPU_AVAILABLE := $(shell if command -v nvidia-smi >/dev/null 2>&1 && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then echo 1; else echo 0; fi)
USE_DOCKER_GPU := $(if $(filter 1 true yes on,$(DOCKER_GPU)),1,$(if $(filter 0 false no off,$(DOCKER_GPU)),0,$(DOCKER_GPU_AVAILABLE)))
COMPOSE_RUN = $(COMPOSE) $(if $(filter 1,$(USE_DOCKER_GPU)),-f docker-compose.yml -f docker-compose.gpu.yml,)

.DEFAULT_GOAL := help

.PHONY: help setup sync lock test lint format typecheck check ci run calibrate review debug kernel \
	recognize templates \
	docker-build docker-up docker-calibrate docker-review docker-export docker-smoke \
	prod-up prod-calibrate prod-review \
	agent-check clean

help: ## List all targets (run with no arguments)
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  DATA=$(DATA)   OUT=$(OUT)   CALIB=$(CALIB)   LIMIT=$(LIMIT)   MAX_IMAGES=$(MAX_IMAGES)   OFFSET=$(OFFSET)   PROGRESS_EVERY=$(PROGRESS_EVERY)"
	@echo "  THREADS=$(THREADS)   WORKERS=$(WORKERS)   PDISEG_BACKEND=$(PDISEG_BACKEND)   DOCKER_GPU=$(DOCKER_GPU)   USE_DOCKER_GPU=$(USE_DOCKER_GPU)"
	@echo "  DOCKER_CPUS=$(DOCKER_CPUS)   DOCKER_MEMORY=$(DOCKER_MEMORY)   DOCKER_NICE=$(DOCKER_NICE)   PORT=$(PORT)"
	@echo "  Example: make run DATA=dataset OUT=result MAX_IMAGES=50 OFFSET=0"
	@echo "  Docker all images, 50% CPU + GPU auto: make docker-up"

setup: sync kernel ## Install dependencies (uv sync --extra dev)

kernel: ## Register Jupyter kernel for this project (.venv)
	$(PY) -m ipykernel install --user --name pdiseg --display-name "Python (pdiseg)"

sync: ## Sync venv with uv.lock
	$(UV) sync --extra dev

lock: ## Refresh uv.lock after pyproject.toml changes
	$(UV) lock

test: ## Run pytest with coverage
	$(PY) pytest --cov=src/pdiseg --cov-report=term-missing -q

lint: ## Ruff check
	$(PY) ruff check src tests

format: ## Ruff format + fix
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

typecheck: ## mypy on pdiseg/
	$(PY) mypy src/pdiseg

check: lint typecheck test ## lint + mypy + tests

ci: sync check ## Local CI parity

run: ## Segment DATA → OUT
	$(PY) pdiseg $(DATA) $(OUT) $(if $(MAX_IMAGES),--limit $(MAX_IMAGES),) --offset $(OFFSET) --progress-every $(PROGRESS_EVERY) --workers $(WORKERS)

calibrate: ## Write calibration/ (overlays, boxes.json, stats.csv)
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT) $(if $(MAX_IMAGES),--limit $(MAX_IMAGES),) --offset $(OFFSET) --progress-every $(PROGRESS_EVERY) --workers $(WORKERS)

templates: ## Bootstrap templates/ (one crop per class via T1 detector)
	$(PY) python scripts/build-templates.py $(DATA) templates

recognize: ## Recognize products in OUT segments against templates/ (T2)
	$(PY) pdiseg-recognize $(OUT) templates --sweep-csv $(CALIB)/recognition_sweep.csv --workers $(WORKERS) --progress-every $(PROGRESS_EVERY)

review: ## Open viewer at http://127.0.0.1:$(PORT)/
	$(PY) pdiseg-review --dataset $(DATA) --calibration $(CALIB) --result $(OUT) --port $(PORT)

debug: ## Run full pipeline on sample (1 image/class) → debug_result/
	$(PY) pdiseg-debug $(DATA) debug_result/result --bundle-root debug_result/bundles --per-class 1

docker-build: ## Build selected Docker image (CPU or GPU override)
	$(COMPOSE_RUN) build pipeline

docker-up: docker-build ## Run pipeline in Docker
	DATA="$(abspath $(DATA))" OUT="$(abspath $(OUT))" MAX_IMAGES="$(MAX_IMAGES)" OFFSET="$(OFFSET)" \
		PROGRESS_EVERY="$(PROGRESS_EVERY)" THREADS="$(THREADS)" WORKERS="$(WORKERS)" \
		PDISEG_BACKEND="$(PDISEG_BACKEND)" PDISEG_BACKEND_LOG="$(PDISEG_BACKEND_LOG)" \
		DOCKER_CPUS="$(DOCKER_CPUS)" DOCKER_MEMORY="$(DOCKER_MEMORY)" DOCKER_NICE="$(DOCKER_NICE)" \
		$(COMPOSE_RUN) up pipeline

docker-calibrate: docker-build ## Calibrate in Docker
	DATA="$(abspath $(DATA))" CALIB="$(abspath $(CALIB))" LIMIT="$(LIMIT)" MAX_IMAGES="$(MAX_IMAGES)" OFFSET="$(OFFSET)" \
		PROGRESS_EVERY="$(PROGRESS_EVERY)" THREADS="$(THREADS)" WORKERS="$(WORKERS)" \
		PDISEG_BACKEND="$(PDISEG_BACKEND)" PDISEG_BACKEND_LOG="$(PDISEG_BACKEND_LOG)" \
		DOCKER_CPUS="$(DOCKER_CPUS)" DOCKER_MEMORY="$(DOCKER_MEMORY)" DOCKER_NICE="$(DOCKER_NICE)" \
		$(COMPOSE_RUN) --profile tools run --rm calibrate

docker-review: docker-build ## Review viewer in Docker (port $(PORT))
	DATA="$(abspath $(DATA))" OUT="$(abspath $(OUT))" CALIB="$(abspath $(CALIB))" PORT=$(PORT) \
		THREADS="$(THREADS)" WORKERS="$(WORKERS)" PDISEG_BACKEND="$(PDISEG_BACKEND)" PDISEG_BACKEND_LOG="$(PDISEG_BACKEND_LOG)" \
		DOCKER_CPUS="$(DOCKER_CPUS)" DOCKER_MEMORY="$(DOCKER_MEMORY)" DOCKER_NICE="$(DOCKER_NICE)" \
		DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) \
		$(COMPOSE_RUN) --profile tools up review

docker-export: ## Copy named volumes to ./result on host
	bash scripts/docker-export-artifacts.sh

docker-smoke: ## Compose E2E test (synthetic dataset)
	bash scripts/docker-smoke.sh

prod-up: ## Run published GHCR image via docker-compose.prod.yml
	docker compose -f docker-compose.prod.yml up pipeline

prod-calibrate: ## Calibrate with published GHCR image via docker-compose.prod.yml
	docker compose -f docker-compose.prod.yml --profile tools run --rm calibrate

prod-review: ## Review with published GHCR image via docker-compose.prod.yml
	docker compose -f docker-compose.prod.yml --profile tools up review

agent-check: ## Validate agent harness (.agents, skills, rules)
	bash scripts/agent-harness-check.sh

clean: ## Remove result/, calibration/, caches
	rm -rf $(OUT) $(CALIB) .docker-smoke .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

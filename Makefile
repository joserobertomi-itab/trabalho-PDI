UV      ?= uv
PY      := $(UV) run
DATA    ?= data/Train_and_Validation
OUT     ?= result
CALIB   ?= calibration
LIMIT   ?= 3
PORT    ?= 8765
COMPOSE ?= docker compose

.DEFAULT_GOAL := help

.PHONY: help setup sync lock test lint format typecheck check ci run calibrate review debug kernel \
	docker-build docker-up docker-calibrate docker-review docker-export docker-smoke \
	agent-check clean

help: ## List all targets (run with no arguments)
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  DATA=$(DATA)   OUT=$(OUT)   CALIB=$(CALIB)   LIMIT=$(LIMIT)   PORT=$(PORT)"
	@echo "  Example: make run DATA=dataset OUT=result"

setup: sync kernel ## Install dependencies (uv sync --extra dev)

kernel: ## Register Jupyter kernel for this project (.venv)
	$(PY) -m ipykernel install --user --name pdiseg --display-name "Python (pdiseg)"

sync: ## Sync venv with uv.lock
	$(UV) sync --extra dev

lock: ## Refresh uv.lock after pyproject.toml changes
	$(UV) lock

test: ## Run pytest with coverage
	$(PY) pytest --cov=pdiseg --cov-report=term-missing -q

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
	$(PY) pdiseg $(DATA) $(OUT)

calibrate: ## Write calibration/ (overlays, boxes.json, stats.csv)
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT)

review: ## Open viewer at http://127.0.0.1:$(PORT)/
	$(PY) pdiseg-review --dataset $(DATA) --calibration $(CALIB) --result $(OUT) --port $(PORT)

debug: ## Run full pipeline on sample (1 image/class) → debug_result/
	$(PY) pdiseg-debug $(DATA) debug_result/result --bundle-root debug_result/bundles --per-class 1

docker-build: ## Build pdiseg:latest image
	$(COMPOSE) build pipeline

docker-up: docker-build ## Run pipeline in Docker
	DATA=./$(DATA) OUT=./$(OUT) $(COMPOSE) up --build pipeline

docker-calibrate: docker-build ## Calibrate in Docker
	DATA=./$(DATA) CALIB=./$(CALIB) LIMIT=$(LIMIT) $(COMPOSE) --profile tools run --rm calibrate

docker-review: docker-build ## Review viewer in Docker (port $(PORT))
	DATA=./$(DATA) OUT=./$(OUT) CALIB=./$(CALIB) PORT=$(PORT) \
		DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) \
		$(COMPOSE) --profile tools up review

docker-export: ## Copy named volumes to ./result on host
	bash scripts/docker-export-artifacts.sh

docker-smoke: ## Compose E2E test (synthetic dataset)
	bash scripts/docker-smoke.sh

agent-check: ## Validate agent harness (.agents, skills, rules)
	bash scripts/agent-harness-check.sh

clean: ## Remove result/, calibration/, caches
	rm -rf $(OUT) $(CALIB) .docker-smoke .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

UV      ?= uv
PY      := $(UV) run
DATA    ?= data/Train_and_Validation
OUT     ?= result
CALIB   ?= calibration
LIMIT   ?= 3
PORT    ?= 8765
COMPOSE ?= docker compose

.DEFAULT_GOAL := help

.PHONY: help setup sync lock test lint format typecheck check ci run calibrate review \
	docker-build docker-up docker-calibrate docker-review docker-export docker-smoke clean

help:  ##
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: sync  ## Install deps (uv sync --extra dev)

sync:  ##
	$(UV) sync --extra dev

lock:  ##
	$(UV) lock

test:  ##
	$(PY) pytest --cov=pdiseg --cov-report=term-missing -q

lint:  ##
	$(PY) ruff check pdiseg tests

format:  ##
	$(PY) ruff format pdiseg tests
	$(PY) ruff check --fix pdiseg tests

typecheck:  ##
	$(PY) mypy pdiseg

check: lint typecheck test  ##

ci: sync check  ##

run:  ##
	$(PY) pdiseg $(DATA) $(OUT)

calibrate:  ##
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT)

review:  ##
	$(PY) pdiseg-review --dataset $(DATA) --calibration $(CALIB) --result $(OUT) --port $(PORT)

docker-build:  ##
	$(COMPOSE) build pipeline

docker-up: docker-build  ##
	DATA=./$(DATA) OUT=./$(OUT) $(COMPOSE) up --build pipeline

docker-calibrate: docker-build  ##
	DATA=./$(DATA) $(COMPOSE) --profile tools run --rm calibrate

docker-review: docker-build  ##
	DATA=./$(DATA) OUT=./$(OUT) CALIB=./$(CALIB) PORT=$(PORT) \
		DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) \
		$(COMPOSE) --profile tools up review

docker-export:  ##
	bash scripts/docker-export-artifacts.sh

docker-smoke:  ##
	bash scripts/docker-smoke.sh

clean:  ##
	rm -rf $(OUT) $(CALIB) .docker-smoke .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

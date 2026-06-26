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
	docker-build docker-up docker-calibrate docker-review docker-export docker-smoke \
	agent-check clean

help: ## Lista todos os targets (rode sem argumentos)
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variáveis (sobrescreva na linha de comando):"
	@echo "  DATA=$(DATA)   OUT=$(OUT)   CALIB=$(CALIB)   LIMIT=$(LIMIT)   PORT=$(PORT)"
	@echo "  Ex.: make run DATA=dataset OUT=result"

setup: sync ## Instala dependências (uv sync --extra dev)

sync: ## Sincroniza o venv com uv.lock
	$(UV) sync --extra dev

lock: ## Atualiza uv.lock depois de mudar pyproject.toml
	$(UV) lock

test: ## Roda pytest com cobertura
	$(PY) pytest --cov=pdiseg --cov-report=term-missing -q

lint: ## Ruff check
	$(PY) ruff check src tests

format: ## Ruff format + fix
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

typecheck: ## mypy em pdiseg/
	$(PY) mypy src/pdiseg

check: lint typecheck test ## lint + mypy + testes

ci: sync check ## Igual ao CI local

run: ## Segmenta DATA → OUT
	$(PY) pdiseg $(DATA) $(OUT)

calibrate: ## Gera calibration/ (overlays, boxes.json, stats.csv)
	$(PY) pdiseg-calibrate $(DATA) $(CALIB) --per-class-limit $(LIMIT)

review: ## Abre o viewer em http://127.0.0.1:$(PORT)/
	$(PY) pdiseg-review --dataset $(DATA) --calibration $(CALIB) --result $(OUT) --port $(PORT)

docker-build: ## Build da imagem pdiseg:latest
	$(COMPOSE) build pipeline

docker-up: docker-build ## Roda o pipeline no Docker
	DATA=./$(DATA) OUT=./$(OUT) $(COMPOSE) up --build pipeline

docker-calibrate: docker-build ## Calibrate no Docker
	DATA=./$(DATA) CALIB=./$(CALIB) LIMIT=$(LIMIT) $(COMPOSE) --profile tools run --rm calibrate

docker-review: docker-build ## Review viewer no Docker (porta $(PORT))
	DATA=./$(DATA) OUT=./$(OUT) CALIB=./$(CALIB) PORT=$(PORT) \
		DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) \
		$(COMPOSE) --profile tools up review

docker-export: ## Copia volumes nomeados para ./result no host
	bash scripts/docker-export-artifacts.sh

docker-smoke: ## Teste E2E do Compose (dataset sintético)
	bash scripts/docker-smoke.sh

agent-check: ## Valida estrutura do harness (.agents, skills, rules)
	bash scripts/agent-harness-check.sh

clean: ## Apaga result/, calibration/, caches
	rm -rf $(OUT) $(CALIB) .docker-smoke .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

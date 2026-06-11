# Poultry-packaging name-label segmentation — local tasks.
# Override paths on the command line, e.g.: make calibrate CALIB=out LIMIT=5

VENV  ?= .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

DATA  ?= data/Train_and_Validation
OUT   ?= resultado
CALIB ?= calibration
LIMIT ?= 3

.DEFAULT_GOAL := help

.PHONY: help setup test run calibrate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install the runtime + dev dependencies
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements-dev.txt

test:  ## Run the test suite
	$(PY) -m pytest -q

run:  ## Segment the base into $(OUT)/<Class>/<source>_segmentada_<N>.png
	$(PY) -m pdiseg $(DATA) $(OUT)

calibrate:  ## Run the calibration harness: overlays + stats.csv into $(CALIB)/
	$(PY) -m pdiseg.calibrate_cli $(DATA) $(CALIB) --per-class-limit $(LIMIT)

clean:  ## Remove generated outputs and caches
	rm -rf $(OUT) $(CALIB) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

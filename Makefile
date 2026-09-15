PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
export PYTHONPATH := $(CURDIR)/src

CONFIG ?= configs/train/medgemma_lora.yaml
EVAL_CONFIG ?= configs/eval/medgemma.yaml

.PHONY: help venv install install-flash train eval notebook lint clean

help:
	@echo "Targets:"
	@echo "  venv          Create virtualenv at $(VENV)"
	@echo "  install       Install dependencies"
	@echo "  install-flash Install flash-attn (optional, needs CUDA)"
	@echo "  train         Run training (CONFIG=path)"
	@echo "  eval          Run classification eval (EVAL_CONFIG=path)"
	@echo "  notebook      Start Jupyter"
	@echo "  lint          Run ruff on src and scripts"
	@echo "  clean         Remove venv and caches"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

install: venv
	$(PIP) install -r requirements.txt

install-flash: install
	$(PIP) install "flash-attn>=2.7.0" --no-build-isolation

train:
	$(PY) scripts/train.py $(CONFIG)

eval:
	$(PY) scripts/eval.py $(EVAL_CONFIG)

notebook:
	$(BIN)/jupyter notebook notebooks/prepare/

lint:
	$(BIN)/ruff check src scripts

clean:
	rm -rf $(VENV) .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

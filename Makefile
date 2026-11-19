PY ?= .venv/bin/python
GATEDRAG ?= ../gated-llm-service
.PHONY: install test lint dryrun
install:
	python3 -m venv .venv && $(PY) -m pip install -q -e ".[train,dev]" -e $(GATEDRAG)
test:
	PYTHONPATH=src $(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts && $(PY) -m mypy src
dryrun:         ## end-to-end on a laptop (fake generator, tiny model): wiring only, not a result
	rm -rf data/interim/dry && mkdir -p data/interim/dry/in
	GATEDRAG_DRYRUN=1 P2_INPUT=data/interim/dry/in P2_OUT=data/interim/dry/out P2_GATEDRAG=$(GATEDRAG) P2_SELF=. \
	  P2_GOLDEN=$(GATEDRAG)/data/golden/pilot_frozen $(PY) scripts/kaggle/run_project2.py

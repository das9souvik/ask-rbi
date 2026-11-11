.PHONY: setup fetch build eval app test deploy

PY ?= .venv/bin/python

setup:            ## create a Python 3.12 environment and install everything
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[app,dev]"

fetch:            ## download the RBI PDFs in data/manifest.toml
	$(PY) -m ask_rbi.cli fetch

build:            ## parse, chunk and embed into index/
	$(PY) -m ask_rbi.cli build

eval:             ## run the evaluation set, writes eval/results.md
	$(PY) -m ask_rbi.cli eval

app:              ## run the demo on http://127.0.0.1:7860
	$(PY) app.py

test:             ## offline tests
	ASK_RBI_EMBEDDER=hashing $(PY) -m pytest

deploy:           ## push to your Hugging Face Space (set SPACE=username/ask-rbi)
	$(PY) scripts/deploy_space.py --space $(SPACE)

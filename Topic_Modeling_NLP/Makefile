PYTHON := python3
VENV := NLP_env
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: all setup install clean distclean test

all: setup

setup: $(VENV)/bin/activate install

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install:
	$(PIP) install -r requirements.txt
	$(PIP) install \
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

test:
	$(PY) -c "import spacy; nlp=spacy.load('en_core_web_sm'); print('spaCy OK')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

distclean: clean
	rm -rf $(VENV)

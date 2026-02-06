PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
UV := $(shell command -v uv 2> /dev/null || command which uv 2> /dev/null)

.PHONY: install dev-install install_conda dev-install_conda
.PHONY: tests tests_uv doc doc_uv docupdate docupdate_uv servedoc
.PHONY: lint lint_uv typeannot typeannot_uv coverage, coverage_uv

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

uvcheck:
ifndef UV
	$(error "Ensure uv is in your PATH")
endif
	@echo Using uv: $(UV)

install:
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt && $(PIP) install -e .

install_conda:
	conda env create -f environment.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pyproximal && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pyproximal && pip install -e .

dev-install_conda_arm:
	conda env create -f environment-dev-arm.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pyproximal && pip install -e .

tests:
	make pythoncheck
	pytest

tests_uv:
	make uvcheck
	$(UV) run pytest

doc:
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && make html && cd ..

doc_uv:
	make uvcheck
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && $(UV) run make html && cd ..

docupdate:
	cd docs && make html && cd ..

docupdate_uv:
	make uvcheck
	cd docs && $(UV) run make html && cd ..

servedoc:
	make pythoncheck
	$(PYTHON) -m http.server --directory docs/build/html/

servedoc_uv:
	make uvcheck
	$(UV) run python -m http.server --directory docs/build/html/

lint:
	ruff check docs/source examples/ pyproximal/ pytests/ tutorials/

lint_uv:
	make uvcheck
	$(UV) run ruff check docs/source examples/ pyproximal/ pytests/ tutorials/

typeannot:
	mypy pyproximal/

typeannot_uv:
	make uvcheck
	$(UV) run mypy pyproximal/

coverage:
	coverage run -m pytest && coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/

coverage_uv:
	make uvcheck
	$(UV) run coverage run -m pytest  &&\
	$(UV) run coverage xml &&\
	$(UV) run coverage html  &&\
	$(UV) run python -m http.server --directory htmlcov/

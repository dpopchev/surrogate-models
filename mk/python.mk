# mk/python.mk -- the Python layer, included by the base Makefile once the
# project becomes Python (add-python), at the helper and the package level.
# Uses the base's UV, LOCAL, BUILD, log_done / log_warn; appends its checks to
# CHECKS. The package level adds mk/wheel.mk (the wheel).

# ------------------------------------------------------------------------------
### Python
# ------------------------------------------------------------------------------

PYVER := .python-version
# `=` (not `:=`): read at recipe time, so the $(PYVER) rule can create the file first.
PYTHON_VERSION = $(shell cat $(PYVER) 2>/dev/null)
DEFAULT_PYTHON ?= 3.13

RUN := $(UV) run

# Tool residuals never pollute the tree. Tools with a pyproject key are pointed at
# $(LOCAL) there (pytest, ruff); tools that only read the environment are pointed
# here. Add a line for every tool you add that writes a cache.
export PYTHONPYCACHEPREFIX          := $(abspath $(LOCAL))/pycache
export HYPOTHESIS_STORAGE_DIRECTORY := $(abspath $(LOCAL))/.hypothesis
export MYPY_CACHE_DIR               := $(abspath $(LOCAL))/.mypy_cache

# Colocated level: a module run by path imports shared code as `shared.<m>`,
# so the repository root goes on PYTHONPATH once shared/ exists.
ifneq ($(wildcard shared/.),)
export PYTHONPATH := $(CURDIR)
endif

CHECKS += lint typecheck test

$(PYVER):
	@$(UV) python pin $(DEFAULT_PYTHON)
	$(call log_warn,$(PYVER) was missing -- pinned python $(DEFAULT_PYTHON); edit it and re-run make sync)

.PHONY: sync
sync: $(PYVER) pyproject.toml | $(LOCAL) ## Create or refresh .venv and uv.lock
	@$(UV) sync
	$(call log_done,environment synced -- python $(PYTHON_VERSION)$(comma) uv.lock current)

.PHONY: lint
lint: ## Ruff check without fixes
	@$(RUN) ruff check .
	$(call log_done,ruff check clean)

.PHONY: format
format: ## Ruff format and import ordering
	@$(RUN) ruff format .
	@$(RUN) ruff check --fix --select I .
	$(call log_done,formatted)

.PHONY: typecheck
typecheck: ## Pyright over the configured paths
	@$(RUN) pyright
	$(call log_done,pyright clean)

.PHONY: test
test: ## Pytest with doctests
	@$(RUN) pytest
	$(call log_done,tests passed)

.PHONY: test-quick
test-quick: ## Pytest stopping at the first failure
	@$(RUN) pytest -x -q
	$(call log_done,quick tests passed)

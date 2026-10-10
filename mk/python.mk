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
CLEANS += clean-python

$(PYVER):
	@$(UV) python pin $(DEFAULT_PYTHON)
	$(call log_warn,$(PYVER) was missing -- pinned python $(DEFAULT_PYTHON); edit it and re-run make sync)

.PHONY: sync
sync: $(PYVER) pyproject.toml | $(LOCAL) ## Create or refresh .venv and uv.lock
	@$(UV) sync
	$(call log_done,environment synced -- python $(PYTHON_VERSION)$(comma) uv.lock current)

.PHONY: lint
lint: ## Ruff check and format check without fixes
	@$(RUN) ruff check .
	@$(RUN) ruff format --check .
	$(call log_done,ruff check clean and the code ruff-formatted)

.PHONY: format
format: ## Ruff format and import ordering
	@$(RUN) ruff format .
	@$(RUN) ruff check --fix --select I .
	$(call log_done,formatted)

.PHONY: typecheck
typecheck: ## Pyright over the configured paths
	@$(RUN) pyright
	$(call log_done,pyright clean)

# pytest exits 5 when it collects nothing -- a fresh layer before its first test:
# warned in log_warn's format, never failed; every other non-zero exit breaks.
define _pytest
	@rc=0; $(RUN) pytest $(1) || rc=$$?; \
	if [ $$rc -eq 5 ]; then \
	  printf "$(BOLD)$(YELLOW)%s$(RESET) %s\n" "[WARN]" "pytest collected no tests -- write the first test_<name>.py beside its module" >&2; \
	elif [ $$rc -ne 0 ]; then exit $$rc; fi
endef

# M=<category> selects one category marker (rules/python.md, Tests); unset, every category
# but integration runs -- integration runs only by name or through test-integration.
# K=<expression> is pytest's -k, the selector a Task's Check names (skills/board, plan step 3).
M ?=
K ?=
.PHONY: test
test: ## Pytest with doctests (M=<marker> one category, K=<expression> matching tests)
	$(call _pytest,-m "$(if $(M),$(M),not integration)" $(if $(K),-k "$(K)"))
	$(call log_done,tests passed)

.PHONY: test-quick
test-quick: ## Pytest stopping at the first failure (M= and K= as for test)
	$(call _pytest,-x -q -m "$(if $(M),$(M),not integration)" $(if $(K),-k "$(K)"))
	$(call log_done,quick tests passed)

.PHONY: test-integration
test-integration: ## Pytest of the integration tests against the real systems
	$(call _pytest,-m integration)
	$(call log_done,integration tests passed)

.PHONY: test-stats
test-stats: ## Show the recorded test durations
	@$(RUN) pytest --suite-stats
	$(call log_done,suite stats from the stats_file in pyproject.toml)

.PHONY: clean-python
clean-python: ## Remove the python tool caches in local/
	@rm -rf $(LOCAL)/pycache $(LOCAL)/.pytest_cache $(LOCAL)/.ruff_cache $(LOCAL)/.mypy_cache $(LOCAL)/.hypothesis
	$(call log_done,removed the python tool caches in $(LOCAL)/)

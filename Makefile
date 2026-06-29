.SUFFIXES:
.DELETE_ON_ERROR:

SHELL       := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules --no-builtin-variables
MAKEFLAGS += --output-sync=target

.DEFAULT_GOAL := help

# =============================================================================
### Help
# =============================================================================

.PHONY: help
help: ## Show this help
	@grep -E '^(###[ ].+|[a-zA-Z0-9_%/-]+:.*##[^#])' $(firstword $(MAKEFILE_LIST)) \
		| sed -E \
		-e 's|^### (.+)|\x1b[1;36m\1\x1b[0m|' \
		-e 's|^([a-zA-Z0-9_%/-]+):.*## (.+)|  \x1b[32m\1\x1b[0m:\2|' \
		| awk -F: '{ \
		if ($$0 !~ /:/) { printf "\n%s\n", $$0 } \
		else { printf "  %-20s %s\n", $$1, $$2 } \
		}'

# =============================================================================
# Environment
# =============================================================================

DOTENV := .env

ifneq ($(wildcard $(DOTENV)),)
	include $(DOTENV)
	export
endif

# =============================================================================
# Logging
# =============================================================================

BOLD   := \033[1m
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

#open Suppress --warn-undefined-variables false positives for $(call) arguments
1 :=
2 :=
3 :=

define _log_raw
{ \
	_tag="[$(2)]"; \
	_msg="$(3)"; \
	if _c=$$(tput cols 2>/dev/null); then _cols=$$_c; else _cols=80; fi; \
	_max=$$(( _cols - $${#_tag} - 4 )); \
	if [ $${#_msg} -gt $$_max ] && [ $$_max -gt 0 ]; then \
	_msg="$${_msg:0:$$_max}..."; \
	fi; \
	printf "$(BOLD)$(1)%s$(RESET) %s\n" "$$_tag" "$$_msg" >&2; \
	}
endef

log_info = $(call _log_raw,$(CYAN),INFO,$(1))
log_ok   = $(call _log_raw,$(GREEN),DONE,$(1))
log_warn = $(call _log_raw,$(YELLOW),WARN,$(1))
log_err  = $(call _log_raw,$(RED),FAIL,$(1))

# =============================================================================
# File Helpers
# =============================================================================

# add line if absent, appending at the end -- dedups without a global sort,
# so an existing structured file keeps its order. Useful for gitignore and a-like.
define add_line
grep --quiet --line-regexp --fixed-strings -- $(1) $(2) 2>/dev/null || echo $(1) >> $(2)
endef

# del line if present, preserving the file order (no global sort),
# useful for gitignore and a-like
define del_line
if [[ -e $(2) ]]; then sed --in-place '\,\b$(1)\b,d' $(2); fi
endef

# =============================================================================
# Configuration
# =============================================================================

PYTHON_VERSION_FILE := .python-version
VENV                := .venv
VENV_STAMP          := $(VENV)/pyvenv.cfg
GITIGNORE           := .gitignore

PYMANAGER := uv
PYVENV    := $(PYMANAGER) venv
PYSYNC    := $(PYMANAGER) sync
PYINSTALL := $(PYMANAGER) pip install
PYRUN     := $(PYMANAGER) run python
PYTEST    := $(PYMANAGER) run pytest
PYMYPY    := $(PYMANAGER) run mypy
PYIMPORTS := $(PYMANAGER) run lint-imports
PYBUILD   := $(PYMANAGER) build
PYAPP     := $(PYMANAGER) run regression_surrogate_pde_solver

# =============================================================================
### Virtual Environment
# =============================================================================

.PHONY: venv
venv: $(VENV_STAMP) ## Build local .venv with uv from .python-version

$(VENV_STAMP): $(PYTHON_VERSION_FILE)
	@if ! command -v $(PYMANAGER) >/dev/null 2>&1; then \
		$(call log_err,$(PYMANAGER) not found on PATH); \
		exit 1; \
	fi
	@$(call log_info,Creating $(VENV) with Python $$(cat $(PYTHON_VERSION_FILE)))
	@$(PYVENV) --python "$$(cat $(PYTHON_VERSION_FILE))" $(VENV)
	@$(call add_line,$(VENV),$(GITIGNORE))
	@$(call log_ok,$(VENV) ready)

.PHONY: sync
sync: venv ## Sync dependencies into .venv from the lockfile
	@$(call log_info,Syncing dependencies with $(PYSYNC))
	@$(PYSYNC)
	@$(call log_ok,dependencies synced)

.PHONY: install
install: venv ## Install the projeck into .venv
	@$(call log_info,Installing project with $(PYINSTALL))
	@$(PYINSTALL) -e .
	@$(call log_ok,project installed)

# =============================================================================
### Quality
# =============================================================================

.PHONY: test
test: ## Run the full test suite (doctests + coverage)
	@$(call log_info,Running tests with coverage)
	@$(PYTEST) --cov=regression_surrogate_pde_solver --cov-report=term-missing
	@$(call log_ok,tests passed)

.PHONY: test-quick
test-quick: ## Run the test suite quietly (per TDD cycle)
	@$(PYTEST) -q

.PHONY: typecheck
typecheck: ## Static type-check src with mypy
	@$(call log_info,Type-checking with mypy)
	@$(PYMYPY)
	@$(call log_ok,types clean)

.PHONY: lint-imports
lint-imports: ## Check onion-ring import contracts with import-linter
	@$(call log_info,Checking import contracts)
	@$(PYIMPORTS)
	@$(call log_ok,imports clean)

.PHONY: lint
lint: lint-imports ## Lint, format-check (ruff) and import contracts
	@$(call log_info,Linting with ruff)
	@$(PYMANAGER) run ruff check src tests
	@$(PYMANAGER) run ruff format --check src tests
	@$(call log_ok,lint clean)

.PHONY: format
format: ## Auto-format and fix lint with ruff (src + tests)
	@$(call log_info,Formatting with ruff)
	@$(PYMANAGER) run ruff format src tests
	@$(PYMANAGER) run ruff check --fix src tests
	@$(call log_ok,formatted)

# =============================================================================
### Run
# =============================================================================

.PHONY: run
run: ## Start the regression_surrogate_pde_solver package (uv run regression_surrogate_pde_solver) and show its help menu
	@$(PYAPP) --help

# =============================================================================
### Build
# =============================================================================

.PHONY: build
build: test | dist ## Build the sdist + wheel into dist/ (tests must pass first)
	@$(call log_info,Building sdist + wheel with $(PYBUILD))
	@$(PYBUILD)
	@$(call log_ok,artifacts in dist/)

# Order-only prerequisite of build: ensure the output dir exists and is ignored.
dist:
	@mkdir -p $@
	@$(call add_line,$@,$(GITIGNORE))

# =============================================================================
### JupyterLab
# =============================================================================

NOTEBOOKS_DIR        := notebooks
NOTEBOOKS_GITATTR    := $(NOTEBOOKS_DIR)/.gitattributes
JUPYTER_DIR          := .jupyter
LAB_STAMP            := $(VENV)/.lab-installed
LAB_GROUP            := lab
LAB_PORT             ?= 8888
LAB_HOST             ?= 127.0.0.1
LAB_USER_DIR         := $(JUPYTER_DIR)/lab/user-settings
LAB_FIXTURES_DIR     := etc/jupyter/user-settings
PERSONAL_LAB_DIR     ?= $(HOME)/.myjupyter-settings
PERSONAL_LAB_SRC     := $(PERSONAL_LAB_DIR)/user-settings


LAB_PACKAGES := \
				jupyterlab \
				jupyterlab-vim \
				jupyterlab-lsp \
				jupyter-lsp \
				python-lsp-server[all] \
				python-lsp-ruff \
				nbstripout

PYLAB     := $(PYMANAGER) run jupyter lab
PYNBSTRIP := $(PYMANAGER) run nbstripout

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

.PHONY: lab-install
lab-install: $(LAB_STAMP) ## Install JupyterLab + vim + LSP + nbstripout into .venv (group: lab)

$(LAB_STAMP): $(firstword $(MAKEFILE_LIST)) | venv
	@$(call log_info,Adding $(words $(LAB_PACKAGES)) packages to dependency-group [$(LAB_GROUP)])
	@$(PYMANAGER) add --group $(LAB_GROUP) $(LAB_PACKAGES)
	@$(PYMANAGER) sync --group $(LAB_GROUP)
	@mkdir -p $(dir $@) && touch $@
	@$(call add_line,.jupyter/,$(GITIGNORE))
	@$(call log_ok,JupyterLab toolchain ready)

# -----------------------------------------------------------------------------
# Notebooks dir + .gitattributes (so nbstripout actually filters .ipynb)
# -----------------------------------------------------------------------------

$(NOTEBOOKS_DIR):
	@$(call log_info,Creating $(NOTEBOOKS_DIR)/)
	@mkdir -p $@
	@$(call add_line,$(NOTEBOOKS_DIR)/.ipynb_checkpoints/,$(GITIGNORE))
	@$(call add_line,$(NOTEBOOKS_DIR)/.virtual_documents/,$(GITIGNORE))
	@$(call log_ok,$(NOTEBOOKS_DIR)/ ready)


$(NOTEBOOKS_GITATTR): | $(NOTEBOOKS_DIR)
	@$(call log_info,Writing $@)
	@printf '%s\n' \
		'*.ipynb filter=nbstripout' \
		'*.ipynb diff=ipynb' \
		'*.ipynb linguist-language=Jupyter\ Notebook' \
		> $@
	@$(call log_ok,$@ written)

# -----------------------------------------------------------------------------
# Jupyter config dir
# -----------------------------------------------------------------------------

$(JUPYTER_DIR):
	@mkdir -p $@
	@$(call add_line,$(JUPYTER_DIR)/lab/,$(GITIGNORE))
	@$(call add_line,$(JUPYTER_DIR)/runtime/,$(GITIGNORE))

# -----------------------------------------------------------------------------
# Project fixtures (optional; empty if etc/jupyter/user-settings/ doesn't exist)
# -----------------------------------------------------------------------------

LAB_FIXTURES := $(shell find $(LAB_FIXTURES_DIR) -type f -name '*.jupyterlab-settings' 2>/dev/null)
LAB_LINKS    := $(patsubst $(LAB_FIXTURES_DIR)/%,$(LAB_USER_DIR)/%,$(LAB_FIXTURES))

.PHONY: lab-config
lab-config: $(LAB_LINKS) ## Symlink project fixtures into .jupyter/lab/user-settings (no-op if none)

$(LAB_USER_DIR)/%.jupyterlab-settings: $(LAB_FIXTURES_DIR)/%.jupyterlab-settings | $(JUPYTER_DIR)
	@mkdir -p $(dir $@)
	@ln -sfn $(abspath $<) $@
	@$(call log_ok,linked project $*)

# -----------------------------------------------------------------------------
# Personal overlay (~/.myjupyter-settings) -- project wins on conflicts
# -----------------------------------------------------------------------------

.PHONY: lab-config-personal
lab-config-personal: $(LAB_STAMP) lab-config | $(JUPYTER_DIR) ## Overlay ~/.myjupyter-settings onto .jupyter/lab/user-settings
	@if [ ! -d "$(PERSONAL_LAB_SRC)" ]; then \
		printf "$(BOLD)$(YELLOW)[WARN]$(RESET) %s not found -- skipping personal overlay\n" "$(PERSONAL_LAB_SRC)" >&2; \
		exit 0; \
		fi; \
		dest_root="$(abspath $(LAB_USER_DIR))"; \
		if [ -z "$$dest_root" ] || [ "$$dest_root" = "/" ]; then \
		printf "$(BOLD)$(RED)[FAIL]$(RESET) LAB_USER_DIR resolved to '%s' -- aborting\n" "$$dest_root" >&2; \
		exit 1; \
		fi; \
		mkdir -p "$$dest_root"; \
		cd "$(PERSONAL_LAB_SRC)" && find . -type f -name '*.jupyterlab-settings' | while read -r f; do \
		rel="$${f#./}"; \
		target="$$dest_root/$$rel"; \
		if [ -e "$$target" ] || [ -L "$$target" ]; then \
		printf "$(BOLD)$(CYAN)[SKIP]$(RESET) %s (project fixture wins)\n" "$$rel" >&2; \
		continue; \
		fi; \
		mkdir -p "$$(dirname "$$target")"; \
		ln -sfn "$(abspath $(PERSONAL_LAB_SRC))/$$rel" "$$target"; \
		printf "$(BOLD)$(GREEN)[LINK]$(RESET) %s -> %s\n" "$$rel" "$$target" >&2; \
		done; \
		printf "$(BOLD)$(GREEN)[DONE]$(RESET) personal overlay applied from %s\n" "$(PERSONAL_LAB_SRC)" >&2

# -----------------------------------------------------------------------------
# Git filter for clean notebook diffs
# -----------------------------------------------------------------------------

.PHONY: lab-hooks
lab-hooks: $(LAB_STAMP) $(NOTEBOOKS_GITATTR) ## Install nbstripout git filter + notebooks/.gitattributes
	@$(call log_info,Installing nbstripout git filter)
	@$(PYNBSTRIP) --install
	@$(call log_ok,nbstripout hook active in $(CURDIR))

# -----------------------------------------------------------------------------
# Run lab
# -----------------------------------------------------------------------------

.PHONY: lab
lab: $(LAB_STAMP) lab-config-personal | $(NOTEBOOKS_DIR) ## Run JupyterLab rooted at notebooks/
	@$(call log_info,Starting JupyterLab at http://$(LAB_HOST):$(LAB_PORT) root=$(NOTEBOOKS_DIR)/)
	@JUPYTER_CONFIG_DIR=$(abspath $(JUPYTER_DIR)) \
		$(PYLAB) \
		--notebook-dir=$(NOTEBOOKS_DIR) \
		--ServerApp.ip=$(LAB_HOST) \
		--ServerApp.port=$(LAB_PORT) \
		--no-browser

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

.PHONY: lab-config-clean
lab-config-clean: ## Wipe linked user-settings (forces re-link on next lab-config-personal)
	@rm -rf $(LAB_USER_DIR)
	@$(call log_ok,$(LAB_USER_DIR) wiped)

.PHONY: lab-clean
lab-clean: lab-config-clean ## Drop install stamp + linked settings
	@rm -f $(LAB_STAMP)
	@$(call log_ok,lab stamp removed)

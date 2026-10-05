.SUFFIXES:
.DELETE_ON_ERROR:

SHELL       := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules --no-builtin-variables
MAKEFLAGS += --output-sync=target

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

BOLD   := \033[1m
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

# Suppress --warn-undefined-variables false positives for $(call) arguments
1 :=
2 :=
3 :=

# The `@` lives inside the define: call log_done / log_warn WITHOUT a leading `@`.
# A literal comma inside the message is $(comma) -- make splits $(call) on commas.
comma := ,

define _log_raw
	@{ \
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

# [DONE] closes every target and names the artifact; [INFO] reports a step on the
# way; [WARN] only for an actionable, handled condition. The tags are four letters
# wide. There is no log_error: a failure breaks the recipe.
log_done = $(call _log_raw,$(GREEN),DONE,$(1))
log_info = $(call _log_raw,$(CYAN),INFO,$(1))
log_warn = $(call _log_raw,$(YELLOW),WARN,$(1))

# ------------------------------------------------------------------------------
### Help
# ------------------------------------------------------------------------------

# A `###` banner names a help section; a `## description` after a target lists it.
# Every file in MAKEFILE_LIST is read (grep -h), so an included mk/*.mk shows too.
# %-30s: the padded field holds 9 invisible color bytes, leaving 21 visible columns.
.PHONY: help
help: ## Show this help
	@grep -hE '^(###[ ].+|[a-zA-Z0-9_%/-]+:.*##[^#])' $(MAKEFILE_LIST) \
	  | sed -E \
	      -e 's|^### (.+)|\x1b[1;36m\1\x1b[0m|' \
	      -e 's|^([a-zA-Z0-9_%/-]+):.*## (.+)|  \x1b[32m\1\x1b[0m:\2|' \
	  | awk -F: '{ \
	      if ($$0 !~ /:/) { printf "\n%s\n", $$0 } \
	      else { printf "  %-30s %s\n", $$1, $$2 } \
	    }'

# ------------------------------------------------------------------------------
### Environment
# ------------------------------------------------------------------------------

# uv runs the board checker (a PEP 723 script) with no project of this repo.
UV := uv

# Directory convention (rules/makefile.md -> Directories):
#   $(BUILD) the end product -- `make clean` removes it whole
#   $(LOCAL) machine-local and interim: inputs, state, keys, scratch, tool
#            residuals -- never tracked; make clean removes only state/ and
#            the layers' residuals
#   $(STATE) interim results and artifacts of any step, when a step needs
#            them -- written by explicit file targets
BUILD ?= build
LOCAL ?= local
OUT     := $(BUILD)/outputs
REPORTS := $(BUILD)/reports
STATE   := $(LOCAL)/state

$(LOCAL):
	@mkdir -p $(LOCAL)/keys $(LOCAL)/scratch $(LOCAL)/inputs $(STATE)
	$(call log_warn,created $(LOCAL)/ -- inputs/$(comma) state/$(comma) keys/ and scratch/ live here; git-ignored$(comma) make clean removes only state/)

$(OUT) $(REPORTS) $(STATE):
	@mkdir -p $@
	$(call log_warn,created $@/ -- an artifact directory$(comma) removed by make clean)

.PHONY: clean-build clean-state
clean-build: ## Remove build/ -- the end products
	@rm -rf $(BUILD)
	$(call log_done,removed $(BUILD)/)

clean-state: ## Remove local/state/ -- the interim results
	@rm -rf $(STATE)
	$(call log_done,removed $(STATE)/)

# Layers (mk/python.mk, ...) add their targets and append to CHECKS and CLEANS.
# Included here, before `check` and `clean`, because make expands prerequisites
# when it reads a rule.
CHECKS :=
CLEANS := clean-build clean-state
include $(wildcard mk/*.mk)

.PHONY: clean
clean: $(CLEANS) ## Run every clean -- inputs, keys, settings, scratch, .board/ .venv/ untouched
	$(call log_done,cleaned: $(CLEANS) -- inputs$(comma) keys$(comma) settings$(comma) scratch$(comma) initial-data$(comma) .board/ and .venv/ untouched)

# ------------------------------------------------------------------------------
### Quality
# ------------------------------------------------------------------------------

.PHONY: check
check: board-check $(CHECKS) ## Run every check -- the board and each layer
	$(call log_done,all checks passed)

# ------------------------------------------------------------------------------
### Board
# ------------------------------------------------------------------------------

# The work record lives in .board/ (rules/board.md); its checker is the board
# skill's PEP 723 script, run through uv so no project dependency is needed.
# Transitions take arguments a fixed target cannot carry -- run them ad hoc:
#   $(BOARD) move T-032 done --evidence "commit abc1234; make check PASS"
#   $(BOARD) block W-002 --reason "..." --waiting-on W-006
#   $(BOARD) unblock W-002 --why "..." | note T-032 "..." | archive E-001
BOARD ?= $(UV) run $(HOME)/.claude/skills/board/scripts/board.py

# .board/ may be tracked or gitignored (rules/board.md): a checkout without it --
# a fresh clone, CI -- warns once and passes instead of failing every check.
HAS_BOARD = $(wildcard .board/board.toml)
NO_BOARD  := no board in this checkout -- .board/ is local or gitignored

.PHONY: board-check
board-check: ## Validate .board/ -- C1-C6 and V1-V18
	$(if $(HAS_BOARD),@$(BOARD) check)
	$(if $(HAS_BOARD),$(call log_done,board well-typed),$(call log_warn,$(NO_BOARD)))

.PHONY: board-status
board-status: ## Show the open path and blocked items
	$(if $(HAS_BOARD),@$(BOARD) status)
	$(if $(HAS_BOARD),$(call log_done,board status shown),$(call log_warn,$(NO_BOARD)))

.PHONY: board-next
board-next: ## Show the current and next item per level
	$(if $(HAS_BOARD),@$(BOARD) next)
	$(if $(HAS_BOARD),$(call log_done,board next shown),$(call log_warn,$(NO_BOARD)))

.PHONY: board-knowledge
board-knowledge: ## List open questions and undistilled facts
	$(if $(HAS_BOARD),@$(BOARD) knowledge)
	$(if $(HAS_BOARD),$(call log_done,board knowledge listed),$(call log_warn,$(NO_BOARD)))

.PHONY: board-render
board-render: board-check ## Regenerate .board/BOARD.md and the optional views
	$(if $(HAS_BOARD),@$(BOARD) render)
	$(if $(HAS_BOARD),$(call log_done,board views rendered),$(call log_warn,$(NO_BOARD)))

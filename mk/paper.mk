# mk/paper.mk -- the paper layer: assembles the chapter folders into one flat
# LaTeX project per flavour under $(BUILD)/paper/ and compiles it. Uses the
# base's BUILD, log_done / log_info.

# ------------------------------------------------------------------------------
### Paper
# ------------------------------------------------------------------------------

# A flavour is an entry file 00_metadata/<flavour>.tex (documentclass + inputs);
# it becomes main.tex of $(BUILD)/paper/<flavour>/. Add a flavour by adding its
# entry file and its name here.
PAPER_FLAVOURS := article
PAPER_FLAVOUR  ?= article
PAPER_ENTRY    := 00_metadata/$(PAPER_FLAVOUR).tex
PAPER_DIR      := $(BUILD)/paper/$(PAPER_FLAVOUR)

# Every .tex and .bib of the chapter folders except the flavour entries. The
# build is flat, so basenames must be unique -- compile refuses a clash.
PAPER_SRC = $(filter-out $(PAPER_FLAVOURS:%=00_metadata/%.tex), \
              $(shell find [0-9][0-9]_* -type f \( -name '*.tex' -o -name '*.bib' \) | sort))

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -silent

.PHONY: compile
compile: ## Build the flat paper and its PDF under build/paper/
	@dups=$$(printf '%s\n' $(notdir $(PAPER_SRC)) | sort | uniq -d); \
	  if [ -n "$$dups" ]; then \
	    printf 'duplicate source basenames (the build is flat): %s\n' "$$dups" >&2; exit 1; \
	  fi
	@rm -rf $(PAPER_DIR) && mkdir -p $(PAPER_DIR)
	@cp $(PAPER_SRC) $(PAPER_DIR)/
	@cp $(PAPER_ENTRY) $(PAPER_DIR)/main.tex
	$(call log_info,assembled $(words $(PAPER_SRC)) sources into $(PAPER_DIR)/ -- compiling)
	@$(LATEXMK) -cd $(PAPER_DIR)/main.tex
	$(call log_done,compiled $(PAPER_DIR)/main.pdf)

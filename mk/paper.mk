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
PAPER_ZIP      := $(BUILD)/paper/$(PAPER_FLAVOUR).zip
PAPER_PDF      := $(BUILD)/paper/$(PAPER_FLAVOUR).pdf
# LaTeX residue and the PDF stay out of $(PAPER_DIR): that folder is what Overleaf's
# Upload folder takes, sources and assets only.
PAPER_LATEX    := $(abspath $(BUILD))/paper/.latex/$(PAPER_FLAVOUR)

# Every .tex and .bib of the chapter folders except the flavour entries. The
# build is flat, so basenames must be unique -- compile refuses a clash.
PAPER_SRC = $(filter-out $(PAPER_FLAVOURS:%=00_metadata/%.tex), \
              $(shell find [0-9][0-9]_* -type f \( -name '*.tex' -o -name '*.bib' \) | sort))

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -silent

# Generated assets: one file per asset, named <dir>_<name>.<ext>, written by the
# script beside its section and copied flat next to the sources.
ASSETS       := $(BUILD)/assets
PAPER_ASSETS := $(ASSETS)/00_metadata_build_stamp.tex

# The stamp reads git state (commit, uncommitted changes) that make cannot watch,
# so it is rebuilt on every run.
.PHONY: FORCE
FORCE:

$(ASSETS)/00_metadata_build_stamp.tex: 00_metadata/build_stamp.py FORCE
	@$(RUN) python $< $@
	$(call log_done,build stamp written to $@)

# Section 4.1: the NS EDA writes the figures paper.toml selects, their numbers and the
# figures .tex; the .tex stands for the whole set (the script clears its own stale assets).
NS_EDA        := 40_data_analysis/41_neutron_stars/eda_neutron_stars.py
NS_EDA_ASSETS := $(ASSETS)/41_neutron_stars_figures.tex
PAPER_ASSETS  += $(NS_EDA_ASSETS)

$(NS_EDA_ASSETS): $(NS_EDA) paper.toml shared/config.py shared/plots.py $(STATE)/neutron_stars.parquet
	@$(RUN) python $(NS_EDA) $(STATE)/neutron_stars.parquet $(ASSETS)
	$(call log_done,NS EDA figures and numbers written to $(ASSETS)/)

.PHONY: compile
compile: $(PAPER_ASSETS) ## Build the flat paper and its PDF under build/paper/
	@dups=$$(printf '%s\n' $(notdir $(PAPER_SRC)) | sort | uniq -d); \
	  if [ -n "$$dups" ]; then \
	    printf 'duplicate source basenames (the build is flat): %s\n' "$$dups" >&2; exit 1; \
	  fi
	@rm -rf $(PAPER_DIR) && mkdir -p $(PAPER_DIR)
	@cp $(PAPER_SRC) $(ASSETS)/* $(PAPER_DIR)/
	@cp $(PAPER_ENTRY) $(PAPER_DIR)/main.tex
	$(call log_info,assembled $(words $(PAPER_SRC)) sources and the assets of $(ASSETS)/ into $(PAPER_DIR)/ -- compiling)
	@$(LATEXMK) -cd -outdir=$(PAPER_LATEX) $(PAPER_DIR)/main.tex
	@cp $(PAPER_LATEX)/main.pdf $(PAPER_PDF)
	@rm -f $(PAPER_ZIP)
	@zip -q -j -X $(PAPER_ZIP) $(PAPER_DIR)/*
	$(call log_done,compiled $(PAPER_PDF) -- upload folder $(PAPER_DIR)/$(comma) spare $(PAPER_ZIP))

# Clean room: the upload folder alone must compile, as Overleaf will see it.
PAPER_VERIFY := $(BUILD)/paper/verify

CHECKS += paper-verify

.PHONY: paper-verify
paper-verify: compile ## Compile a copy of the upload folder in isolation
	@rm -rf $(PAPER_VERIFY) && mkdir -p $(PAPER_VERIFY)
	@cp -R $(PAPER_DIR)/. $(PAPER_VERIFY)/
	@$(LATEXMK) -cd $(PAPER_VERIFY)/main.tex
	@test -f $(PAPER_VERIFY)/main.pdf
	$(call log_done,$(PAPER_DIR)/ compiles on its own -- $(PAPER_VERIFY)/main.pdf)

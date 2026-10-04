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

.PHONY: compile
compile: $(PAPER_ASSETS) ## Build the flat paper and its PDF under build/paper/
	@dups=$$(printf '%s\n' $(notdir $(PAPER_SRC)) | sort | uniq -d); \
	  if [ -n "$$dups" ]; then \
	    printf 'duplicate source basenames (the build is flat): %s\n' "$$dups" >&2; exit 1; \
	  fi
	@rm -rf $(PAPER_DIR) && mkdir -p $(PAPER_DIR)
	@cp $(PAPER_SRC) $(PAPER_ASSETS) $(PAPER_DIR)/
	@cp $(PAPER_ENTRY) $(PAPER_DIR)/main.tex
	$(call log_info,assembled $(words $(PAPER_SRC)) sources and $(words $(PAPER_ASSETS)) assets into $(PAPER_DIR)/ -- compiling)
	@$(LATEXMK) -cd $(PAPER_DIR)/main.tex
	@rm -f $(PAPER_ZIP)
	@zip -q -j -X $(PAPER_ZIP) $(PAPER_DIR)/main.tex \
	  $(addprefix $(PAPER_DIR)/,$(notdir $(PAPER_SRC) $(PAPER_ASSETS)))
	$(call log_done,compiled $(PAPER_DIR)/main.pdf -- Overleaf upload $(PAPER_ZIP))

# Clean room: the zip alone must compile, as Overleaf will see it.
PAPER_VERIFY := $(BUILD)/paper/verify

CHECKS += paper-verify

.PHONY: paper-verify
paper-verify: compile ## Compile the unpacked zip in an empty folder
	@rm -rf $(PAPER_VERIFY) && mkdir -p $(PAPER_VERIFY)
	@unzip -q $(PAPER_ZIP) -d $(PAPER_VERIFY)
	@$(LATEXMK) -cd $(PAPER_VERIFY)/main.tex
	@test -f $(PAPER_VERIFY)/main.pdf
	$(call log_done,$(PAPER_ZIP) compiles on its own -- $(PAPER_VERIFY)/main.pdf)

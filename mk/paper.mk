# mk/paper.mk -- the paper layer: assembles the chapter folders into one flat
# LaTeX project, the Overleaf upload folder under $(BUILD)/paper/, and compiles
# it. Uses the base's BUILD, LOCAL, log_done / log_info.

# ------------------------------------------------------------------------------
### Paper
# ------------------------------------------------------------------------------

# A flavour is an entry file 00_metadata/<flavour>.tex (documentclass + inputs);
# it becomes main.tex of the upload folder $(BUILD)/paper/overleaf/. Add a
# flavour by adding its entry file and its name here.
PAPER_FLAVOURS := article
PAPER_FLAVOUR  ?= article
PAPER_ENTRY    := 00_metadata/$(PAPER_FLAVOUR).tex
PAPER_DIR      := $(BUILD)/paper/overleaf
PAPER_ZIP      := $(BUILD)/paper/overleaf.zip
PAPER_PDF      := $(BUILD)/paper/$(PAPER_FLAVOUR).pdf
# LaTeX residue and the stand-alone check live under $(WORK), so $(BUILD)/paper/ holds
# the deliverables only: the PDF, the zip and the upload folder (sources and assets).
WORK           := $(BUILD)/.work
PAPER_LATEX    := $(abspath $(WORK))/latex/$(PAPER_FLAVOUR)

# Every .tex and .bib of the chapter folders except the flavour entries. The
# build is flat, so basenames must be unique -- compile refuses a clash.
PAPER_SRC = $(filter-out $(PAPER_FLAVOURS:%=00_metadata/%.tex), \
              $(shell find [0-9][0-9]_* -type f \( -name '*.tex' -o -name '*.bib' \) | sort))

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -silent

# Generated assets: one folder per section, $(ASSETS)/<section>/, each file named
# <section>_<kind>_<name>.<ext> (fig, tab; num for the numbers) -- a figure's PNG and its
# .tex wrapper share one stem. The section prefix keeps basenames unique once compile
# copies every asset flat next to the sources.
ASSETS       := $(BUILD)/assets
PAPER_ASSETS := $(ASSETS)/00_metadata/00_metadata_build_stamp.tex

# The stamp reads git state (commit, uncommitted changes) that make cannot watch,
# so it is rebuilt on every run.
.PHONY: FORCE
FORCE:

$(ASSETS)/00_metadata/00_metadata_build_stamp.tex: 00_metadata/build_stamp.py FORCE
	@$(RUN) python $< $@
	$(call log_done,build stamp written to $@)

# Section 3.1: the NS EDA writes the figures and tables paper.toml selects and their numbers;
# the numbers file, written on every run, stands for the whole set (the script empties its
# own folder first).
NS_EDA        := 40_data_analysis/41_neutron_stars/eda_neutron_stars.py
NS_EDA_ASSETS := $(ASSETS)/41_neutron_stars/41_neutron_stars_num.tex
PAPER_ASSETS  += $(NS_EDA_ASSETS)

$(NS_EDA_ASSETS): $(NS_EDA) paper.toml shared/config.py shared/plots.py shared/eda.py $(STATE)/neutron_stars.parquet
	@$(RUN) python $(NS_EDA) $(STATE)/neutron_stars.parquet $(ASSETS)
	$(call log_done,NS EDA figures and numbers written to $(@D)/)

# Section 3.2: the BH EDA, the same way; its numbers file stands for the whole set.
BH_EDA        := 40_data_analysis/42_black_holes/eda_black_holes.py
BH_EDA_ASSETS := $(ASSETS)/42_black_holes/42_black_holes_num.tex
PAPER_ASSETS  += $(BH_EDA_ASSETS)

$(BH_EDA_ASSETS): $(BH_EDA) paper.toml shared/config.py shared/plots.py shared/eda.py $(STATE)/black_holes.parquet
	@$(RUN) python $(BH_EDA) $(STATE)/black_holes.parquet $(ASSETS)
	$(call log_done,BH EDA figures and numbers written to $(@D)/)

# Section 3.3: the split and charge-floor numbers, from the split file and both tables.
PREP_NUMBERS := 40_data_analysis/43_preprocessing/preprocessing_numbers.py
PREP_ASSETS  := $(ASSETS)/43_preprocessing/43_preprocessing_num.tex
PAPER_ASSETS += $(PREP_ASSETS)

$(PREP_ASSETS): $(PREP_NUMBERS) paper.toml shared/config.py $(STATE)/split.parquet $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet
	@$(RUN) python $(PREP_NUMBERS) $(STATE)/split.parquet $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(ASSETS) --seed $(SPLIT_SEED)
	$(call log_done,Section 3.3 numbers written to $@)

# Section 4.1: the baseline surrogate trains a network, minutes on this CPU, so it is its own
# target and not in PAPER_ASSETS -- make check (paper-verify) would retrain it on every change
# of a shared module (T-080). The numbers file stands for the whole set; the epoch lines stream
# while it trains.
BASELINE        := 50_methodology/51_algorithms/fit_baseline.py
BASELINE_ASSETS := $(ASSETS)/51_algorithms/51_algorithms_num.tex

$(BASELINE_ASSETS): $(BASELINE) paper.toml shared/config.py shared/design.py shared/surrogate.py shared/plots.py shared/eda.py shared/runs.py shared/diagnostics.py $(STATE)/neutron_stars.parquet $(STATE)/split.parquet
	$(call log_info,fitting the NS mass baseline -- one line per epoch; make follow shows it from any terminal)
	@$(RUN) python $(BASELINE) $(STATE)/neutron_stars.parquet $(STATE)/split.parquet $(ASSETS) $(STATE)
	$(call log_done,baseline numbers and parity figure written to $(@D)/ -- run diagnostics under $(STATE)/51_algorithms/)

.PHONY: baseline
baseline: $(BASELINE_ASSETS) ## Fit the baseline surrogate for Section 4.1
	$(call log_done,baseline assets current in $(dir $(BASELINE_ASSETS)))

# Each baseline run logs into its own <run>/train.log and repoints latest.log at it (W-054);
# tail -F follows the name, so one make follow keeps up across runs. It ends with Ctrl-C, so
# its log_done never prints.
BASELINE_LOG := $(STATE)/51_algorithms/latest.log

.PHONY: follow
follow: ## Follow the running baseline's log live
	$(call log_info,following $(BASELINE_LOG) -- waits for a run if none started yet; Ctrl-C ends it)
	@tail -n 40 -F $(BASELINE_LOG)
	$(call log_done,stopped following $(BASELINE_LOG))

# The runs at a glance (W-057): one line per run from its run.json, newest first; make run
# lists one run's files, the latest unless R names a run folder.
LIST_RUNS := 50_methodology/51_algorithms/list_runs.py
R         ?= latest

.PHONY: runs
runs: ## List the baseline runs, newest first
	@$(RUN) python $(LIST_RUNS) $(STATE)
	$(call log_done,baseline runs listed from $(STATE)/51_algorithms/)

.PHONY: run
run: ## Show one baseline run's files -- R=<run>, default latest
	@$(RUN) python $(LIST_RUNS) $(STATE) --run $(R)
	$(call log_done,files of baseline run $(R) listed)

.PHONY: assets
assets: $(PAPER_ASSETS) ## Generate the section assets under build/assets/
	$(call log_done,assets in $(ASSETS)/: $(sort $(notdir $(patsubst %/,%,$(dir $(PAPER_ASSETS))))))

.PHONY: compile
compile: assets ## Build the flat paper and its PDF under build/paper/
	@dups=$$({ printf '%s\n' $(notdir $(PAPER_SRC)); find $(ASSETS) -type f -printf '%f\n'; } \
	          | sort | uniq -d); \
	  if [ -n "$$dups" ]; then \
	    printf 'duplicate source or asset basenames (the build is flat): %s\n' "$$dups" >&2; exit 1; \
	  fi
	@rm -rf $(PAPER_DIR) && mkdir -p $(PAPER_DIR)
	@cp $(PAPER_SRC) $(PAPER_DIR)/
	@find $(ASSETS) -type f -exec cp {} $(PAPER_DIR)/ \;
	@cp $(PAPER_ENTRY) $(PAPER_DIR)/main.tex
	$(call log_info,assembled $(words $(PAPER_SRC)) sources and the assets of $(ASSETS)/ into $(PAPER_DIR)/ -- compiling)
	@$(LATEXMK) -cd -outdir=$(PAPER_LATEX) $(PAPER_DIR)/main.tex
	@cp $(PAPER_LATEX)/main.pdf $(PAPER_PDF)
	@rm -f $(PAPER_ZIP)
	@zip -q -j -X $(PAPER_ZIP) $(PAPER_DIR)/*
	$(call log_done,compiled $(PAPER_PDF) -- upload folder $(PAPER_DIR)/$(comma) spare $(PAPER_ZIP))

# Clean room: the upload folder alone must compile, as Overleaf will see it.
PAPER_VERIFY := $(WORK)/verify

CHECKS += paper-verify

.PHONY: paper-verify
paper-verify: compile ## Compile a copy of the upload folder in isolation
	@rm -rf $(PAPER_VERIFY) && mkdir -p $(PAPER_VERIFY)
	@cp -R $(PAPER_DIR)/. $(PAPER_VERIFY)/
	@$(LATEXMK) -cd $(PAPER_VERIFY)/main.tex
	@test -f $(PAPER_VERIFY)/main.pdf
	$(call log_done,$(PAPER_DIR)/ compiles on its own -- $(PAPER_VERIFY)/main.pdf)

# Overleaf, free plan, one project: open it, Upload, drag in every file of
# $(PAPER_DIR)/ (same names overwrite), then delete by hand the files the run
# lists as removed. The list of the last run is machine-local and outside
# $(STATE), so make clean keeps it (CLAUDE.md, Gate deviations).
OVERLEAF_UPLOAD := 00_metadata/overleaf_upload.py
OVERLEAF_LIST   := $(LOCAL)/overleaf/last-upload.txt

.PHONY: overleaf
overleaf: compile ## Prepare the Overleaf upload and list what changed
	@$(RUN) python $(OVERLEAF_UPLOAD) $(PAPER_DIR) $(OVERLEAF_LIST)
	$(call log_done,upload $(PAPER_DIR)/ or $(PAPER_ZIP) -- last upload listed in $(OVERLEAF_LIST))

# Section previews for review: each top-level section (an \input of the entry's
# document body) compiles on its own in $(SECTIONS_WORK)/<section>/, beside a copy
# of the upload folder and the full build's main.aux, so its references to the
# rest of the paper resolve through xr-hyper.
SECTION_ENTRY := 00_metadata/section_entry.py
SECTIONS      := $(shell sed -n '/begin{document}/,/end{document}/s/^\\input{\(.*\)}$$/\1/p' $(PAPER_ENTRY))
SECTIONS_WORK := $(WORK)/sections
SECTIONS_OUT  := $(BUILD)/paper/sections
S ?=

ifneq ($(filter section,$(MAKECMDGOALS)),)
ifeq ($(filter $(S),$(SECTIONS)),)
$(error make section S=<section> -- one of $(SECTIONS))
endif
endif

$(SECTIONS_OUT)/%.pdf: compile
	@rm -rf $(SECTIONS_WORK)/$* && mkdir -p $(SECTIONS_WORK)/$* $(@D)
	@cp -R $(PAPER_DIR)/. $(PAPER_LATEX)/main.aux $(SECTIONS_WORK)/$*/
	@$(RUN) python $(SECTION_ENTRY) $(PAPER_ENTRY) $* $(SECTIONS_WORK)/$*
	@$(LATEXMK) -cd $(SECTIONS_WORK)/$*/section_$*.tex
	@cp $(SECTIONS_WORK)/$*/section_$*.pdf $@
	$(call log_done,section $* compiled on its own to $@)

.PHONY: section
section: $(SECTIONS_OUT)/$(S).pdf ## Compile one section alone, e.g. S=50_methodology

.PHONY: sections
sections: $(SECTIONS:%=$(SECTIONS_OUT)/%.pdf) ## Compile each section alone into build/paper/sections/
	$(call log_done,$(words $(SECTIONS)) sections compiled to $(SECTIONS_OUT)/)

# The help menu holds one short line per target (rules/makefile.md); the worked examples of the
# common commands live here, so they are one command away (W-057).
.PHONY: examples
examples: ## Show common commands with examples
	@printf '%s\n' \
	  'Train and watch the baseline surrogate' \
	  '  make baseline                   fit it (minutes) -> Section 4.1 numbers and parity figure' \
	  '  make follow                     watch the running fit live from any terminal (Ctrl-C ends)' \
	  '  make runs                       one line per run, newest first -- MARE, epochs, commit' \
	  '  make run                        the latest run files -- log, run.json, diagnostics PNGs' \
	  '  make run R=<run folder>         one older run, its folder name taken from make runs' \
	  '' \
	  'The paper' \
	  '  make section S=50_methodology   one section as its own PDF -> $(SECTIONS_OUT)/50_methodology.pdf' \
	  '  make sections                   every section as its own PDF -> $(SECTIONS_OUT)/' \
	  '  make compile                    the whole paper -> $(PAPER_PDF)' \
	  '  make overleaf                   the upload folder and what changed since the last upload' \
	  '  sections for S: $(SECTIONS)' \
	  '' \
	  'Checks and the board' \
	  '  make check                      every check -- board, lint, types, tests, the paper build' \
	  '  make test-quick                 the tests, stopping at the first failure' \
	  '  make board-status               the open path and what is blocked' \
	  '  make board-next                 the current and the next item per level'
	$(call log_done,examples shown -- make help lists every target)

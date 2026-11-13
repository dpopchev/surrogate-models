# mk/paper.mk -- the paper layer: assembles the chapter folders into one flat
# LaTeX project, the Overleaf upload folder under $(BUILD)/paper/, and compiles
# it. Uses the base's BUILD, LOCAL, log_done / log_info.

# ------------------------------------------------------------------------------
### Paper
# ------------------------------------------------------------------------------

# The paper PDF is a deliverable: make paper builds it, and make deliverables builds it with the rest.
DELIVERABLES += paper

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
              $(shell find [0-9][0-9]_* paperlib/latex -type f \( -name '*.tex' -o -name '*.bib' \) | sort))

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -silent

# Generated assets: one folder per section, $(ASSETS)/<section>/, each file named
# <section>_<kind>_<name>.<ext> (fig, tab; num for the numbers) -- a figure's PNG and its
# .tex wrapper share one stem. The section prefix keeps basenames unique once compile
# copies every asset flat next to the sources.
ASSETS       := $(BUILD)/sections
PAPER_ASSETS := $(ASSETS)/00_metadata/00_metadata_build_stamp.tex
# Every asset rule depends on every paperlib module, so a module a script imports is never
# missing from its list; a change to one reruns them all.
PAPERLIB     := $(filter-out paperlib/test_%,$(wildcard paperlib/*.py))

# The stamp reads git state (commit, uncommitted changes) that make cannot watch,
# so it is rebuilt on every run.
.PHONY: FORCE
FORCE:

$(ASSETS)/00_metadata/00_metadata_build_stamp.tex: paperlib/tools/build_stamp.py FORCE
	@$(RUN) python $< $@
	$(call log_done,build stamp written to $@)

# The sections' own rules -- the scripts, knobs, recorded batches and one target named after the
# folder -- live in <chapter>/<section>/<section>.mk, included by the Makefile after mk/*.mk;
# each appends its assets to PAPER_ASSETS. The recipes below read their variables at recipe time.

# Every variant of each figure that has them is a file of its own under
# $(ASSETS)/<section>/figures/ (make figures): paper.toml records the pick, the figure's .tex
# wrapper includes the picked variant's file and make compile copies it beside the sources
# (W-138). Never an asset. The results figures are drawn from the recorded runs -- the
# baseline's ledger entries and the batches below -- and nothing is fitted (W-113).

# Three representative figures, every variant of each, under each style preset of paper.toml
# [plot.presets], as files of their own under $(ASSETS)/<section>/figures/presets/<name>/
# (W-115). The presets are configuration: the sections that show a figure under each one (91, 98,
# 99) redraw it in their own <section>_figures recipe, for each preset of PLOT_PRESETS, with
# PAPER__PLOT__PRESET=<name> into $(PRESETS_WORK)/<name>/<section>/figures/ from the stored data
# and the recorded batches, nothing fitted, and copy its variants to the presets folder.
PRESETS_WORK    := $(WORK)/presets
PLOT_PRESETS    := paper light large

# Each section with a <section>_figures target appends it here; the Makefile gives the figures
# target these prerequisites after the includes, as it gives assets PAPER_ASSETS.
SECTION_FIGURES :=

.PHONY: figures
figures: ## Draw every figure variant as its own PNG under build/sections/<section>/figures/, and three figures under each preset
	$(call log_done,figure variants in $(ASSETS)/<section>/figures/ and three figures under each preset ($(PLOT_PRESETS)) in $(ASSETS)/<section>/figures/presets/)

# The fits as live trials (W-081): optuna-dashboard on a study journal -- make baseline's, or a
# screen's or tuning's batch journal under $(STATE)/<section>/ (W-092). By default the one
# written last, so a running batch is the one shown; JOURNAL=<path> opens another. It ends with
# Ctrl-C, so its log_done never prints.
JOURNAL        ?= $(firstword $(shell ls -t $(STATE)/optuna/journal.log $(STATE)/*/*.journal 2>/dev/null))
DASHBOARD_PORT ?= 8080

.PHONY: dashboard
dashboard: ## Follow the fits live in optuna-dashboard -- the newest study journal, or JOURNAL=<path>
	@test -f "$(JOURNAL)" || { echo "no study journal yet under $(STATE): make baseline, a screen or make tuning writes one" >&2; exit 1; }
	$(call log_info,optuna-dashboard on http://127.0.0.1:$(DASHBOARD_PORT) for $(JOURNAL) -- Ctrl-C ends it)
	@$(RUN) optuna-dashboard $(JOURNAL) --port $(DASHBOARD_PORT)
	$(call log_done,stopped optuna-dashboard)

# The knobs the text cites (W-116): the baseline network's settings from paper.toml and the
# screens' knobs. Each screen's .mk appends its flag to KNOBS_FLAGS and its own path to
# KNOBS_DEPS, so the knobs are rebuilt when a screen's knobs change; the Makefile gives the
# rule those dependencies after the includes, as it gives assets PAPER_ASSETS.
KNOBS        := paperlib/tools/paper_knobs.py
KNOBS_ASSETS := $(ASSETS)/00_metadata/00_metadata_knobs_num.tex
KNOBS_FLAGS  :=
KNOBS_DEPS   :=
PAPER_ASSETS += $(KNOBS_ASSETS)

$(KNOBS_ASSETS): $(KNOBS) paper.toml $(PAPERLIB)
	@$(RUN) python $(KNOBS) $@ $(KNOBS_FLAGS)
	$(call log_done,knobs written to $@)

.PHONY: assets
assets: ## Generate the section assets under build/sections/
	$(call log_done,assets in $(ASSETS)/: $(sort $(notdir $(patsubst %/,%,$(dir $(PAPER_ASSETS))))))

# The assets comparison: record the sha256 sums of every asset (the build stamp left out, it
# carries the commit), then check a later build against them. The baseline lives under
# $(STATE) because a recipe writes $(LOCAL) only there.
ASSETS_BASELINE := $(STATE)/assets-baseline.sha

.PHONY: assets-baseline
assets-baseline: assets ## Record the sha256 sums of the assets as the baseline to compare against
	@mkdir -p $(STATE)
	@find $(ASSETS) -maxdepth 2 -type f ! -name '00_metadata_build_stamp.tex' | sort | xargs sha256sum > $(ASSETS_BASELINE)
	$(call log_done,$(ASSETS_BASELINE) written: $$(wc -l < $(ASSETS_BASELINE)) lines)

.PHONY: assets-same
assets-same: assets ## Check the assets against the recorded baseline
	@test -f $(ASSETS_BASELINE) || { echo "no baseline $(ASSETS_BASELINE): run make assets-baseline at the commit to compare against" >&2; exit 1; }
	@sha256sum --quiet -c $(ASSETS_BASELINE) >&2
	$(call log_done,assets match $(ASSETS_BASELINE))

.PHONY: compile
compile: assets figures ## Build the flat paper and its PDF under build/paper/
	@picked=$$(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' $(ASSETS)/*/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); \
	  dups=$$({ printf '%s\n' $(notdir $(PAPER_SRC)); find $(ASSETS) -maxdepth 2 -type f -printf '%f\n'; \
	            for stem in $$picked; do printf '%s.png\n' $$stem; done; } | sort | uniq -d); \
	  if [ -n "$$dups" ]; then \
	    printf 'duplicate source or asset basenames (the build is flat): %s\n' "$$dups" >&2; exit 1; \
	  fi
	@rm -rf $(PAPER_DIR) && mkdir -p $(PAPER_DIR)
	@cp $(PAPER_SRC) $(PAPER_DIR)/
	@find $(ASSETS) -maxdepth 2 -type f -name '*.tex' -exec cp {} $(PAPER_DIR)/ \;
	@# The picked variants: the PNG each figure wrapper includes, from build/sections/<section>/figures/.
	@for stem in $$(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' $(ASSETS)/*/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do \
	  test -f $(ASSETS)/*/figures/$$stem.png || { printf 'no figure %s.png under %s/<section>/figures/: make figures draws it\n' $$stem $(ASSETS) >&2; exit 1; }; \
	  cp $(ASSETS)/*/figures/$$stem.png $(PAPER_DIR)/; \
	done
	@cp $(PAPER_ENTRY) $(PAPER_DIR)/main.tex
	$(call log_info,assembled $(words $(PAPER_SRC)) sources and the assets of $(ASSETS)/ into $(PAPER_DIR)/ -- compiling)
	@# -g: latexmk records no file an \InputIfFileExists missed, so a figure a pick adds or
	@# removes would leave the PDF of the previous compile (W-122).
	@$(LATEXMK) -g -cd -outdir=$(PAPER_LATEX) $(PAPER_DIR)/main.tex
	@cp $(PAPER_LATEX)/main.pdf $(PAPER_PDF)
	@rm -f $(PAPER_ZIP)
	@zip -q -j -X $(PAPER_ZIP) $(PAPER_DIR)/*
	$(call log_done,compiled $(PAPER_PDF) -- upload folder $(PAPER_DIR)/$(comma) spare $(PAPER_ZIP))

# The deliverable is the compiled PDF; compile's own exit status is the proof (developer,
# W-099: opening it would test the PDF tools, not the paper).
.PHONY: paper
paper: compile ## Compile the paper -> $(PAPER_PDF)
	$(call log_done,deliverable built: $(PAPER_PDF))

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
OVERLEAF_UPLOAD := paperlib/tools/overleaf_upload.py
OVERLEAF_LIST   := $(LOCAL)/overleaf/last-upload.txt

.PHONY: overleaf
overleaf: compile ## Prepare the Overleaf upload and list what changed
	@$(RUN) python $(OVERLEAF_UPLOAD) $(PAPER_DIR) $(OVERLEAF_LIST)
	$(call log_done,upload $(PAPER_DIR)/ or $(PAPER_ZIP) -- last upload listed in $(OVERLEAF_LIST))

# Section previews for review: each top-level section (an \input of the entry's
# document body) compiles on its own in $(SECTIONS_WORK)/<section>/, beside a copy
# of the upload folder and the full build's main.aux, so its references to the
# rest of the paper resolve through xr-hyper.
SECTION_ENTRY := paperlib/tools/section_entry.py
SECTIONS      := $(shell sed -n '/begin{document}/,/end{document}/s/^\\input{\(.*\)}$$/\1/p' $(PAPER_ENTRY))
SECTIONS_WORK := $(WORK)/sections
SECTIONS_OUT  := $(BUILD)/sections
S ?=

ifneq ($(filter section,$(MAKECMDGOALS)),)
ifeq ($(filter $(S),$(SECTIONS)),)
$(error make section S=<section> -- one of $(SECTIONS))
endif
endif

# The preview of a section is $(SECTIONS_OUT)/<section>/<section>.pdf, beside its assets: the
# stem of the pattern is <section>/<section>, so $(*F) is the section.
$(SECTIONS_OUT)/%.pdf: compile
	@rm -rf $(SECTIONS_WORK)/$(*F) && mkdir -p $(SECTIONS_WORK)/$(*F) $(@D)
	@cp -R $(PAPER_DIR)/. $(PAPER_LATEX)/main.aux $(SECTIONS_WORK)/$(*F)/
	@$(RUN) python $(SECTION_ENTRY) $(PAPER_ENTRY) $(*F) $(SECTIONS_WORK)/$(*F)
	@$(LATEXMK) -cd $(SECTIONS_WORK)/$(*F)/section_$(*F).tex
	@cp $(SECTIONS_WORK)/$(*F)/section_$(*F).pdf $@
	$(call log_done,compiled on its own to $@)

.PHONY: section
section: $(SECTIONS_OUT)/$(S)/$(S).pdf ## Compile one section alone, e.g. S=40_pipeline

.PHONY: sections
sections: $(foreach s,$(SECTIONS),$(SECTIONS_OUT)/$(s)/$(s).pdf) ## Compile each section alone into build/sections/<section>/
	$(call log_done,$(words $(SECTIONS)) sections compiled to $(SECTIONS_OUT)/)

# The help menu holds one short line per target (rules/makefile.md); the worked examples of the
# common commands live here, so they are one command away (W-057).
.PHONY: examples
examples: ## Show common commands with examples
	@printf '%s\n' \
	  'Train and watch the baseline surrogate' \
	  '  make baseline                   fit it (minutes) -> baseline numbers and parity figure' \
	  '  make follow                     watch the running fit live from any terminal (Ctrl-C ends)' \
	  '  make runs                       one line per run, newest first -- MARE, epochs, commit' \
	  '  make run                        the latest run files -- log, run.json, diagnostics PNGs' \
	  '  make run R=<run folder>         one older run, its folder name taken from make runs' \
	  '' \
	  'The paper' \
	  '  make section S=40_pipeline      one section as its own PDF -> $(SECTIONS_OUT)/40_pipeline/40_pipeline.pdf' \
	  '  make sections                   every section as its own PDF -> $(SECTIONS_OUT)/<section>/' \
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

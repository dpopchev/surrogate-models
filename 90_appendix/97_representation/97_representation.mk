# Section 5.1: the representation screen (W-066), one factor at a time from the baseline at
# equal model -- about 60 harness jobs, hours on the workers, so like the baseline it is its own
# target; make assets only rebuilds the table from the recorded batch (W-130). Its knobs are written here only (Decision on W-066): the
# k-NN's k, the local RBF's neighbours, the curve-wise spline's knots, the MLP seeds counted
# from paper.toml's seed. The table is read back from the run's ledger batch.
REPRESENTATION        := 90_appendix/97_representation/screen_representation.py
REPRESENTATION_ASSETS := $(ASSETS)/97_representation/97_representation_tab_representation.tex
# 100 neighbours: 50 leave the RBF singular on NS folds whose neighbourhoods span two curves (W-086).
REPRESENTATION_KNOBS  := --k 8 --neighbours 100 --knots 8 --seeds 3
# The text cites them: paper_knobs.py reads the flag, and a change here rebuilds the knobs asset.
KNOBS_FLAGS           += --representation "$(REPRESENTATION_KNOBS)"
KNOBS_DEPS            += 90_appendix/97_representation/97_representation.mk
# The recorded batch Section 5.1 reports (2026-10-08, clean at 700c05e): the table is rebuilt from
# its ledger entries in seconds. Empty it to refit the whole screen (hours):
# make representation REPRESENTATION_BATCH=
REPRESENTATION_BATCH  ?= 01a11acc-9832-7478-8831-1adfdc78f228

# The script stores the table and numbers as JSON under $(STATE)/97_representation/ (after
# emptying the asset folder); render_assets.py writes the table and numbers assets from the JSON.
# One grouped rule, so a build that lost the table reruns the script, as one that lost the
# numbers does.
RENDER_ASSETS        := paperlib/tools/render_assets.py
REPRESENTATION_STATE := $(STATE)/97_representation

$(REPRESENTATION_STATE)/numbers.json $(REPRESENTATION_ASSETS) &: $(REPRESENTATION) $(RENDER_ASSETS) paper.toml $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	$(call log_info,$(if $(REPRESENTATION_BATCH),rebuilding the representation table from batch $(REPRESENTATION_BATCH),screening the representations ($(REPRESENTATION_KNOBS)) -- follow the trials with make dashboard))
	@$(RUN) python $(REPRESENTATION) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(REPRESENTATION_KNOBS) $(if $(REPRESENTATION_BATCH),--batch $(REPRESENTATION_BATCH))
	@$(RUN) python $(RENDER_ASSETS) $(REPRESENTATION_STATE) $(ASSETS)
	$(call log_done,representation table and numbers rendered to $(dir $(REPRESENTATION_ASSETS)))

# A paper asset while its batch is recorded: make assets rebuilds it in seconds, never a screen.
PAPER_ASSETS += $(if $(REPRESENTATION_BATCH),$(REPRESENTATION_ASSETS))

.PHONY: representation 97_representation
representation: $(REPRESENTATION_ASSETS) ## Run the representation screen for Section 5.1
	$(call log_done,representation assets current in $(dir $(REPRESENTATION_ASSETS)))

97_representation: representation ## Build the Section 5.1 assets -- the same as make representation

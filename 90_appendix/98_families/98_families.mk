# Section 5.2: the family screen (W-067), successive halving over the training rows -- every
# family pointwise and curve-wise at the first budget, the top 3 of each unit per pair at the
# second, the top 2 of each at the third, each unit ranked apart because a thinned NS design
# keeps few curves (Decisions on W-067); a fit past 30 min or 4 GB is the family's wall there
# (Decision on W-067), on 3 workers. Hours, so its own target, in make assets only from its
# recorded batch (W-130); one line per finished job in the log. Its knobs are written here
# only. The k-NN, RBF and spline knobs are the representation screen's.
FAMILIES        := 90_appendix/98_families/screen_families.py
FAMILIES_TOML   := 90_appendix/98_families/98_families.toml
FAMILIES_ASSETS := $(ASSETS)/98_families/98_families_tab_families.tex
FAMILIES_KNOBS  := --k 8 --neighbours 100 --knots 8 --per-curve 10 \
                   --budgets 1000 10000 100000 --keeps 3 2 --minutes 30 --memory-gb 4 --workers 3
# The text cites them: paper_knobs.py reads the flag, and a change here rebuilds the knobs asset.
KNOBS_FLAGS     += --families "$(FAMILIES_KNOBS)"
KNOBS_DEPS      += 90_appendix/98_families/98_families.mk
# The recorded batch Section 5.2 reports (2026-10-08, clean at 037443f; 88 outcomes): its saved
# outcomes (local/state/98_families/<batch>.json) rebuild the table and figures in seconds.
# Empty it to refit the whole screen (hours): make families FAMILIES_BATCH=
FAMILIES_BATCH  ?= 01a11be4-6eca-7427-a5dd-a97dd7a0194d

# The script stores the table and numbers as JSON under $(STATE)/98_families/ and still writes
# the figures; render_assets.py writes the table and numbers assets from the JSON, after the
# script has emptied the folder. One grouped rule, so a build that lost the table reruns the
# script, as one that lost the numbers does.
RENDER_ASSETS  := paperlib/tools/render_assets.py
FAMILIES_STATE := $(STATE)/98_families

$(FAMILIES_STATE)/numbers.json $(FAMILIES_ASSETS) &: $(FAMILIES) $(RENDER_ASSETS) paper.toml $(FAMILIES_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	$(call log_info,$(if $(FAMILIES_BATCH),rebuilding the family table from batch $(FAMILIES_BATCH),screening the families ($(FAMILIES_KNOBS)) -- one line per finished job))
	@$(RUN) python $(FAMILIES) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(FAMILIES_KNOBS) $(if $(FAMILIES_BATCH),--batch $(FAMILIES_BATCH))
	@$(RUN) python $(RENDER_ASSETS) $(FAMILIES_STATE) $(ASSETS)
	$(call log_done,family table and numbers rendered to $(dir $(FAMILIES_ASSETS)))

# A paper asset while its batch is recorded: make assets rebuilds it in seconds, never a screen.
PAPER_ASSETS += $(if $(FAMILIES_BATCH),$(FAMILIES_ASSETS))

.PHONY: families 98_families
families: $(FAMILIES_ASSETS) ## Run the family screen for Section 5.2
	$(call log_done,family assets current in $(dir $(FAMILIES_ASSETS)))

98_families: families ## Build the Section 5.2 assets -- the same as make families

# Every figure variant as its own PNG under build/sections/98_families/figures/, redrawn from the
# recorded batch; nothing is screened. FAMILIES_BATCH must name a recorded batch.
FAMILIES_FIGURES := $(ASSETS)/98_families/figures

SECTION_FIGURES += 98_families_figures

# The errors figure's variants under each preset of PLOT_PRESETS, redrawn from the same batch.
.PHONY: 98_families_figures
98_families_figures: $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet ## Draw every family figure variant as its own PNG under build/sections/98_families/figures/, and the errors figure under each preset
	@test -n "$(FAMILIES_BATCH)" || { printf 'FAMILIES_BATCH is empty: no recorded batch to draw from\n' >&2; exit 1; }
	@$(RUN) python $(FAMILIES) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(FAMILIES_KNOBS) --batch $(FAMILIES_BATCH) --figures $(ASSETS)
	@for preset in $(PLOT_PRESETS); do \
	  work=$(PRESETS_WORK)/$$preset; out=$(FAMILIES_FIGURES)/presets/$$preset; \
	  rm -rf $$work/98_families $$out && mkdir -p $$work $$out && \
	  export PAPER__PLOT__PRESET=$$preset && \
	  $(RUN) python $(FAMILIES) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $$work $(STATE) $(FAMILIES_KNOBS) --batch $(FAMILIES_BATCH) --figures $$work && \
	  cp $$work/98_families/figures/98_families_fig_errors_*.png $$out/ || exit 1; \
	done
	$(call log_done,family figure variants in $(FAMILIES_FIGURES)/)

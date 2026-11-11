# Section 5.3's tuning (W-068): every candidate on the family screen's short list searched with
# the same Optuna trials over its own hyperparameters, scored on the frozen folds; a fit past the
# wall fails its trial. Hours, so its own target, in make assets only from its recorded batch
# (W-130); one line per finished job in the log, the trials live in optuna-dashboard on the
# batch's journal under $(STATE)/99_precision/. Its knobs are written here only; k, neighbours
# and knots are the screen's, each overridden by a trial that searches it. The short list and
# the screen's figures come from its recorded batch in the ledger: per pair, the candidates
# within --within figures of the best on the most rows (Section 5.2).
TUNING        := 90_appendix/99_precision/tune_survivors.py
TUNING_TOML   := 90_appendix/99_precision/99_precision.toml
TUNING_ASSETS := $(ASSETS)/99_precision/99_precision_tab_tuning.tex
TUNING_KNOBS  := --k 8 --neighbours 100 --knots 8 --trials 40 --minutes 30 --workers 3 --within 1.0
# The recorded batch (T-206, 2026-10-10): make tuning rebuilds the table from its outcomes in
# seconds; TUNING_BATCH= (empty) runs the whole tuning again (about 33 hours).
TUNING_BATCH  ?= 01a11d49-1647-735a-af05-a58e497428f2

# The script stores the table and numbers as JSON under $(STATE)/99_precision/ and still writes
# the figures; render_assets.py writes the table and numbers assets from the JSON. One grouped
# rule, so a build that lost the table reruns the script, as one that lost the numbers does.
RENDER_ASSETS := paperlib/tools/render_assets.py
TUNING_STATE  := $(STATE)/99_precision

$(TUNING_STATE)/numbers.json $(TUNING_ASSETS) &: $(TUNING) $(RENDER_ASSETS) paper.toml $(TUNING_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	$(call log_info,$(if $(TUNING_BATCH),rebuilding the tuning table from batch $(TUNING_BATCH),tuning the short list ($(TUNING_KNOBS)) -- one line per finished job))
	@$(RUN) python $(TUNING) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(TUNING_KNOBS) --screen-batch $(FAMILIES_BATCH) $(if $(TUNING_BATCH),--batch $(TUNING_BATCH))
	@$(RUN) python $(RENDER_ASSETS) $(TUNING_STATE) $(ASSETS)
	$(call log_done,tuning table and numbers rendered to $(dir $(TUNING_ASSETS)))

# A paper asset while its batch is recorded: make assets rebuilds it in seconds, never a tuning.
PAPER_ASSETS += $(if $(TUNING_BATCH),$(TUNING_ASSETS))

.PHONY: tuning 99_precision
tuning: $(TUNING_ASSETS) ## Run the tuning of the short list for Section 5.3
	$(call log_done,tuning assets current in $(dir $(TUNING_ASSETS)))

99_precision: tuning ## Build the Section 5.3 assets -- the same as make tuning

# Every figure variant as its own PNG under build/sections/99_precision/figures/, redrawn from
# the recorded batch and its journal; nothing is tuned. TUNING_BATCH must name a recorded batch.
TUNING_FIGURES := $(ASSETS)/99_precision/figures

SECTION_FIGURES += 99_precision_figures

# The history figure's variants under each preset of PLOT_PRESETS, redrawn from the same batch.
.PHONY: 99_precision_figures
99_precision_figures: $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet ## Draw every tuning figure variant as its own PNG under build/sections/99_precision/figures/, and the history figure under each preset
	@test -n "$(TUNING_BATCH)" || { printf 'TUNING_BATCH is empty: no recorded batch to draw from\n' >&2; exit 1; }
	@$(RUN) python $(TUNING) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(TUNING_KNOBS) --screen-batch $(FAMILIES_BATCH) --batch $(TUNING_BATCH) --figures $(ASSETS)
	@for preset in $(PLOT_PRESETS); do \
	  work=$(PRESETS_WORK)/$$preset; out=$(TUNING_FIGURES)/presets/$$preset; \
	  rm -rf $$work/99_precision $$out && mkdir -p $$work $$out && \
	  export PAPER__PLOT__PRESET=$$preset && \
	  $(RUN) python $(TUNING) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $$work $(STATE) $(TUNING_KNOBS) --screen-batch $(FAMILIES_BATCH) --batch $(TUNING_BATCH) --figures $$work && \
	  cp $$work/99_precision/figures/99_precision_fig_history_*.png $$out/ || exit 1; \
	done
	$(call log_done,tuning figure variants in $(TUNING_FIGURES)/)

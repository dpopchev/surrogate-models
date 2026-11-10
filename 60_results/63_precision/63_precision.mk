# Section 5.3's tuning (W-068): every candidate on the family screen's short list searched with
# the same Optuna trials over its own hyperparameters, scored on the frozen folds; a fit past the
# wall fails its trial. Hours, so its own target, in make assets only from its recorded batch
# (W-130); one line per finished job in the log, the trials live in optuna-dashboard on the
# batch's journal under $(STATE)/63_precision/. Its knobs are written here only; k, neighbours
# and knots are the screen's, each overridden by a trial that searches it. The short list and
# the screen's figures come from its recorded batch in the ledger: per pair, the candidates
# within --within figures of the best on the most rows (Section 5.2).
TUNING        := 60_results/63_precision/tune_survivors.py
TUNING_TOML   := 60_results/63_precision/63_precision.toml
TUNING_ASSETS := $(ASSETS)/63_precision/63_precision_tab_tuning.tex
TUNING_KNOBS  := --k 8 --neighbours 100 --knots 8 --trials 40 --minutes 30 --workers 3 --within 1.0
# The recorded batch (T-206, 2026-10-10): make tuning rebuilds the table from its outcomes in
# seconds; TUNING_BATCH= (empty) runs the whole tuning again (about 33 hours).
TUNING_BATCH  ?= 01a11d49-1647-735a-af05-a58e497428f2

$(TUNING_ASSETS): $(TUNING) paper.toml $(TUNING_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	$(call log_info,$(if $(TUNING_BATCH),rebuilding the tuning table from batch $(TUNING_BATCH),tuning the short list ($(TUNING_KNOBS)) -- one line per finished job))
	@$(RUN) python $(TUNING) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) $(TUNING_KNOBS) --screen-batch $(FAMILIES_BATCH) $(if $(TUNING_BATCH),--batch $(TUNING_BATCH))
	$(call log_done,tuning table written to $@)

# A paper asset while its batch is recorded: make assets rebuilds it in seconds, never a tuning.
PAPER_ASSETS += $(if $(TUNING_BATCH),$(TUNING_ASSETS))

.PHONY: tuning 63_precision
tuning: $(TUNING_ASSETS) ## Run the tuning of the short list for Section 5.3
	$(call log_done,tuning assets current in $(dir $(TUNING_ASSETS)))

63_precision: tuning ## Build the Section 5.3 assets -- the same as make tuning

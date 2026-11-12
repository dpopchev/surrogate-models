# Section 4.1's pipeline (W-152): the tuning's winner of each pair fitted on every non-test row and
# saved under $(STATE)/41_algorithm/, each printed with its test figures beside its ceiling. Its
# inputs are data under $(STATE): the prepared tables and the split, the audit's ceilings and the
# tuning's winners (no new search). It fits for minutes, so it is no prerequisite of make assets
# or make check. The knobs are the values of TUNING_KNOBS (90_appendix/99_precision), over which a
# winner's params are laid; the seed is paper.toml's.
PIPELINE       := 40_pipeline/41_algorithm/build_surrogates.py
PIPELINE_DIR   := $(STATE)/41_algorithm
PIPELINE_KNOBS = $(foreach f,--k --neighbours --knots,$(subst @, ,$(filter $(f)@%,$(subst $(f) ,$(f)@,$(TUNING_KNOBS)))))
PIPELINE_SEED  := $(shell sed -n 's/^seed = \([0-9]*\).*/\1/p' paper.toml)
PIPELINE_JSON  := $(STATE)/91_neutron_stars/reachable_figures.json $(STATE)/92_black_holes/reachable_figures.json $(STATE)/99_precision/winners.json

.PHONY: pipeline 41_algorithm
pipeline: $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(PIPELINE_JSON) $(PAPERLIB) ## Fit and save the surrogate of each map and target under local/state/41_algorithm/ (minutes)
	@$(RUN) python $(PIPELINE) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) $(PIPELINE_KNOBS) --seed $(PIPELINE_SEED)
	$(call log_done,4 surrogates and 2 domains saved under $(notdir $(LOCAL))/state/41_algorithm/)

41_algorithm: pipeline ## Run the Section 4.1 pipeline -- the same as make pipeline

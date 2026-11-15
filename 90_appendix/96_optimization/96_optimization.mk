# The zones of every tuned candidate (W-169): the recorded tuning batch's test predictions scored
# on the interior curves, the edge curves and the physical edge, at the 95th percentile and at
# the worst row, beside the pair's ceiling. Nothing is fitted; the figures are also saved to
# $(STATE)/96_optimization/zones.json. The physical edge is the first ZONES fraction of a
# black-hole curve (the existence edge) and the last of a neutron-star curve (the stable edge).
# TUNING_BATCH is defined later, in 90_appendix/99_precision/99_precision.mk, so the recipe reads it.
ZONES       := 90_appendix/96_optimization/score_zones.py
ZONES_KNOBS := --bh-edge 0.005 --ns-edge 0.05

.PHONY: zones
zones: ## Print every tuned candidate's figures per zone from the recorded tuning batch
	@test -n "$(TUNING_BATCH)" || { printf 'TUNING_BATCH is empty: no recorded batch to score\n' >&2; exit 1; }
	@$(RUN) python $(ZONES) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) --batch $(TUNING_BATCH) $(ZONES_KNOBS)
	$(call log_done,zones of batch $(TUNING_BATCH) saved to $(STATE)/96_optimization/zones.json)

# The extrapolation of every tuned candidate (W-170): each fitted once at its tuned params without
# the edge curves and the test curves, scored on the edge curves it never saw beside the interior
# test curves. Hours of fits, so its own target; one line per finished job, the outcomes saved after
# each job to $(STATE)/96_optimization/<batch>-extrapolation.json. The base k, neighbours and knots
# are the tuning's (as PIPELINE_KNOBS filters them, read when the recipe runs) and the seed is
# paper.toml's. EXTRAPOLATION_PAIRS limits the run to dataset:target pairs (the one-pair smoke
# run); EXTRAPOLATION_BATCH, once the run is pinned, reprints that saved batch and fits nothing.
EXTRAPOLATION        := 90_appendix/96_optimization/measure_extrapolation.py
EXTRAPOLATION_KNOBS  = --minutes 30 --workers 3 $(PIPELINE_KNOBS) --seed $(PIPELINE_SEED)
EXTRAPOLATION_PAIRS  ?=
EXTRAPOLATION_BATCH  ?= 01a1dd99-bb3a-763d-9f56-c0899dbc2aa8

.PHONY: extrapolation
extrapolation: ## Fit every tuned candidate without the edge curves and print its figures on them (hours; EXTRAPOLATION_BATCH= reprints a saved batch)
	@test -n "$(TUNING_BATCH)" || { printf 'TUNING_BATCH is empty: no recorded batch to fit\n' >&2; exit 1; }
	@$(RUN) python $(EXTRAPOLATION) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) $(EXTRAPOLATION_KNOBS) --tuning-batch $(TUNING_BATCH) $(foreach p,$(EXTRAPOLATION_PAIRS),--pair $(p)) $(if $(EXTRAPOLATION_BATCH),--batch $(EXTRAPOLATION_BATCH))
	$(call log_done,extrapolation of batch $(TUNING_BATCH) printed)

# The zone table of the article (W-174): the zones run's interior and physical-edge figures, the
# extrapolation batch's edge figures, the slope counts, the ceiling and the tuning's verdict per
# candidate, with the macros of each pair's winner. The script stores the table and numbers as
# JSON under $(STATE)/96_optimization/ and render_assets.py writes the assets from it; nothing
# is fitted, seconds. It joins PAPER_ASSETS while EXTRAPOLATION_BATCH is set (TUNING_BATCH, set
# later by 99_precision.mk, is checked when the recipe runs).
ZONES_STATE  := $(STATE)/96_optimization
ZONES_ASSETS := $(ASSETS)/96_optimization/96_optimization_tab_zones.tex $(ASSETS)/96_optimization/96_optimization_num.tex
ZONES_RENDER := paperlib/tools/render_assets.py

$(ZONES_STATE)/numbers.json $(ZONES_STATE)/zones.json $(ZONES_ASSETS) &: $(ZONES) $(ZONES_RENDER) paper.toml $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	@test -n "$(TUNING_BATCH)" || { printf 'TUNING_BATCH is empty: no recorded batch to score\n' >&2; exit 1; }
	@$(RUN) python $(ZONES) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) --batch $(TUNING_BATCH) $(ZONES_KNOBS) --extrapolation-batch $(EXTRAPOLATION_BATCH)
	@$(RUN) python $(ZONES_RENDER) $(ZONES_STATE) $(ASSETS)
	$(call log_done,zone table and numbers rendered to $(ASSETS)/96_optimization/)

PAPER_ASSETS += $(if $(EXTRAPOLATION_BATCH),$(ZONES_ASSETS))

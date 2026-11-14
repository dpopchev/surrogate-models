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

# The black-hole dataset: the black-hole table, prepared from the raw input into runtime state under
# $(STATE). Uses INITIAL_DATA (mk/data.mk), RUN and the base's log_done.

BH_SCRIPT := 30_physical_framework/33_black_holes/prepare_black_holes.py
BH_RAW    := $(INITIAL_DATA)/black-holes/black-holes-zero-phi0.dat

$(STATE)/black_holes.parquet: $(BH_SCRIPT) $(BH_RAW) | $(STATE)
	@$(RUN) python $(BH_SCRIPT) $(BH_RAW) $@
	$(call log_done,black-hole table written to $@)

.PHONY: 33_black_holes
33_black_holes: $(STATE)/black_holes.parquet ## Prepare the black-hole table
	$(call log_done,black-hole table current in $(STATE)/)

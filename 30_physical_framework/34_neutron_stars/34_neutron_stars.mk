# The neutron-star dataset: the neutron-star table, prepared from the raw input into runtime state under
# $(STATE). Uses INITIAL_DATA (mk/data.mk), RUN and the base's log_done.

NS_SCRIPT := 30_physical_framework/34_neutron_stars/prepare_neutron_stars.py
NS_RAW    := $(INITIAL_DATA)/neutron-stars/neutron-stars.dat

$(STATE)/neutron_stars.parquet: $(NS_SCRIPT) $(NS_RAW) | $(STATE)
	@$(RUN) python $(NS_SCRIPT) $(NS_RAW) $@
	$(call log_done,neutron-star table written to $@)

.PHONY: 34_neutron_stars
34_neutron_stars: $(STATE)/neutron_stars.parquet ## Prepare the neutron-star table
	$(call log_done,neutron-star table current in $(STATE)/)

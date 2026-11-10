# Section 3.1: the NS EDA writes the figures and tables paper.toml and this section's toml
# select and their numbers; the numbers file, written on every run, stands for the whole set
# (the script empties its own folder first).
NS_EDA        := 40_data_analysis/41_neutron_stars/eda_neutron_stars.py
NS_EDA_TOML   := 40_data_analysis/41_neutron_stars/41_neutron_stars.toml
NS_EDA_ASSETS := $(ASSETS)/41_neutron_stars/41_neutron_stars_num.tex
PAPER_ASSETS  += $(NS_EDA_ASSETS)
# The EDA probes the split with the split's own folds and seed (51_algorithms.mk); the NS pile-up
# window in M is the band below a curve's M_max that Section 3.1 counts.
NS_PILE_UP_WINDOW := 0.09

$(NS_EDA_ASSETS): $(NS_EDA) paper.toml $(NS_EDA_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet
	@$(RUN) python $(NS_EDA) $(STATE)/neutron_stars.parquet $(ASSETS) --folds $(SPLIT_FOLDS) --seed $(SPLIT_SEED) --window $(NS_PILE_UP_WINDOW)
	$(call log_done,NS EDA figures and numbers written to $(@D)/)

.PHONY: 41_neutron_stars
41_neutron_stars: $(NS_EDA_ASSETS) ## Build the neutron-star EDA assets for Section 3.1
	$(call log_done,Section 3.1 assets current in $(dir $(NS_EDA_ASSETS)))

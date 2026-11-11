# Section 3.1: the NS EDA writes the figures and tables paper.toml and this section's toml
# select and their numbers; the numbers file, written on every run, stands for the whole set
# (the script empties its own folder first).
NS_EDA        := 90_appendix/91_neutron_stars/eda_neutron_stars.py
NS_EDA_TOML   := 90_appendix/91_neutron_stars/91_neutron_stars.toml
NS_EDA_ASSETS := $(ASSETS)/91_neutron_stars/91_neutron_stars_num.tex
PAPER_ASSETS  += $(NS_EDA_ASSETS)
# The EDA probes the split with the split's own folds and seed (95_algorithms.mk); the NS pile-up
# window in M is the band below a curve's M_max that Section 3.1 counts.
NS_PILE_UP_WINDOW := 0.09

# The script stores the tables and numbers as JSON under $(STATE)/91_neutron_stars/ and still
# writes the figures; render_assets.py writes the table and numbers assets from the JSON, after
# the script has emptied the folder. One grouped rule, so a build that lost the figures reruns
# the script, as one that lost the numbers does.
RENDER_ASSETS := paperlib/tools/render_assets.py
NS_EDA_STATE  := $(STATE)/91_neutron_stars

$(NS_EDA_STATE)/numbers.json $(NS_EDA_STATE)/reachable_figures.json $(NS_EDA_ASSETS) &: $(NS_EDA) $(RENDER_ASSETS) paper.toml $(NS_EDA_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet
	@$(RUN) python $(NS_EDA) $(STATE)/neutron_stars.parquet $(ASSETS) --state $(STATE) --folds $(SPLIT_FOLDS) --seed $(SPLIT_SEED) --window $(NS_PILE_UP_WINDOW)
	@$(RUN) python $(RENDER_ASSETS) $(NS_EDA_STATE) $(ASSETS)
	$(call log_done,NS EDA figure wrappers written to $(@D)/ and tables and numbers to $(NS_EDA_STATE)/; make figures draws the PNGs)

.PHONY: 91_neutron_stars
91_neutron_stars: $(NS_EDA_ASSETS) ## Build the neutron-star EDA assets for Section 3.1
	$(call log_done,Section 3.1 assets current in $(dir $(NS_EDA_ASSETS)))

# Every figure variant as its own PNG under build/sections/91_neutron_stars/figures/, drawn from
# the figure data the main run stored; nothing is computed.
NS_EDA_FIGURES := $(ASSETS)/91_neutron_stars/figures

SECTION_FIGURES += 91_neutron_stars_figures

# The mass-density figure's variants under each preset of PLOT_PRESETS, redrawn from the same data.
.PHONY: 91_neutron_stars_figures
91_neutron_stars_figures: $(NS_EDA_STATE)/numbers.json ## Draw every NS EDA figure variant as its own PNG under build/sections/91_neutron_stars/figures/, and the mass-density figure under each preset
	@$(RUN) python $(NS_EDA) --figures $(ASSETS) --state $(STATE)
	@for preset in $(PLOT_PRESETS); do \
	  work=$(PRESETS_WORK)/$$preset; out=$(NS_EDA_FIGURES)/presets/$$preset; \
	  rm -rf $$work/91_neutron_stars $$out && mkdir -p $$work $$out && \
	  export PAPER__PLOT__PRESET=$$preset && \
	  $(RUN) python $(NS_EDA) --figures $$work --state $(STATE) && \
	  cp $$work/91_neutron_stars/figures/91_neutron_stars_fig_mass_density_*.png $$out/ || exit 1; \
	done
	$(call log_done,NS EDA figure variants in $(NS_EDA_FIGURES)/)

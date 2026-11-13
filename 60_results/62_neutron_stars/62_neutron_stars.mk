# The neutron-star results: the neutron-star surrogates against the data ceiling (W-155). The figure scores the
# saved surrogates of $(STATE)/41_algorithm/ (make pipeline) on the held-out curves, so its assets
# join make assets only while those exist: make pipeline fits for minutes and is no prerequisite.
RESNS           := 60_results/62_neutron_stars/results_neutron_stars.py
RESNS_TOML      := 60_results/62_neutron_stars/62_neutron_stars.toml
RESNS_STATE     := $(STATE)/62_neutron_stars
RESNS_NUM       := $(ASSETS)/62_neutron_stars/62_neutron_stars_num.tex
RESNS_FIG       := $(ASSETS)/62_neutron_stars/62_neutron_stars_fig_errors.tex
RESNS_RENDER    := paperlib/tools/render_assets.py
RESNS_SAVED     := $(STATE)/41_algorithm/neutron_stars-mass.pkl $(STATE)/41_algorithm/neutron_stars-charge.pkl
RESNS_CEILINGS  := $(STATE)/91_neutron_stars/reachable_figures.json

ifneq ($(wildcard $(STATE)/41_algorithm/neutron_stars-mass.pkl),)
PAPER_ASSETS += $(RESNS_NUM)
endif

# As the black-hole results: the script stores the JSON and writes the wrapper, the render step the numbers.
$(RESNS_STATE)/numbers.json $(RESNS_NUM) $(RESNS_FIG) &: $(RESNS) $(RESNS_RENDER) paper.toml $(RESNS_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/split.parquet $(RESNS_SAVED) $(RESNS_CEILINGS)
	@$(RUN) python $(RESNS) $(STATE)/neutron_stars.parquet $(STATE)/split.parquet $(STATE) $(ASSETS)
	@$(RUN) python $(RESNS_RENDER) $(RESNS_STATE) $(ASSETS)
	$(call log_done,error figure wrapper and numbers written to $(@D)/; make figures draws the PNGs)

.PHONY: 62_neutron_stars
62_neutron_stars: $(RESNS_NUM) ## Build the neutron-star results assets
	$(call log_done,assets current in $(dir $(RESNS_NUM)))

# Every figure variant as its own PNG under build/sections/62_neutron_stars/figures/, drawn from the
# figure data the main run stored; nothing is computed.
RESNS_FIGURES := $(ASSETS)/62_neutron_stars/figures

SECTION_FIGURES += 62_neutron_stars_figures

.PHONY: 62_neutron_stars_figures
62_neutron_stars_figures: $(RESNS_STATE)/numbers.json ## Draw every error figure variant as its own PNG under build/sections/62_neutron_stars/figures/
	@$(RUN) python $(RESNS) --figures $(ASSETS) --state $(STATE)
	$(call log_done,error figure variants in $(RESNS_FIGURES)/)

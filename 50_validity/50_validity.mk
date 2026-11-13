# The domain of validity (W-153). The figure draws the saved surrogates of
# $(STATE)/41_algorithm/ (make pipeline) against the physical domain of each map, so its assets join
# make assets only while those exist: make pipeline fits for minutes and is no prerequisite.
VALIDITY        := 50_validity/check_domain.py
VALIDITY_TOML   := 50_validity/50_validity.toml
VALIDITY_STATE  := $(STATE)/50_validity
VALIDITY_NUM    := $(ASSETS)/50_validity/50_validity_num.tex
VALIDITY_FIG    := $(ASSETS)/50_validity/50_validity_fig_domain.tex
VALIDITY_RENDER := paperlib/tools/render_assets.py
VALIDITY_SAVED  := $(foreach m,neutron_stars black_holes,$(STATE)/41_algorithm/$(m)-domain.json $(STATE)/41_algorithm/$(m)-mass.pkl)

ifneq ($(wildcard $(STATE)/41_algorithm/black_holes-domain.json),)
PAPER_ASSETS += $(VALIDITY_NUM)
endif

# As the black-hole data appendix: the script stores the JSON and writes the wrapper, the render step the numbers.
$(VALIDITY_STATE)/numbers.json $(VALIDITY_NUM) $(VALIDITY_FIG) &: $(VALIDITY) $(VALIDITY_RENDER) paper.toml $(VALIDITY_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(VALIDITY_SAVED)
	@$(RUN) python $(VALIDITY) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) $(ASSETS)
	@$(RUN) python $(VALIDITY_RENDER) $(VALIDITY_STATE) $(ASSETS)
	$(call log_done,domain figure wrapper and numbers written to $(@D)/; make figures draws the PNGs)

.PHONY: 50_validity
50_validity: $(VALIDITY_NUM) ## Build the domain-of-validity assets
	$(call log_done,assets current in $(dir $(VALIDITY_NUM)))

# Every figure variant as its own PNG under build/sections/50_validity/figures/, drawn from the
# figure data the main run stored; nothing is computed.
VALIDITY_FIGURES := $(ASSETS)/50_validity/figures

SECTION_FIGURES += 50_validity_figures

.PHONY: 50_validity_figures
50_validity_figures: $(VALIDITY_STATE)/numbers.json ## Draw every domain figure variant as its own PNG under build/sections/50_validity/figures/
	@$(RUN) python $(VALIDITY) --figures $(ASSETS) --state $(STATE)
	$(call log_done,domain figure variants in $(VALIDITY_FIGURES)/)

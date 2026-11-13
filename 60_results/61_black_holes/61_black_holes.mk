# The black-hole results: the black-hole surrogates against the data ceiling (W-154). The figure scores the
# saved surrogates of $(STATE)/41_algorithm/ (make pipeline) on the held-out curves, so its assets
# join make assets only while those exist: make pipeline fits for minutes and is no prerequisite.
RESBH           := 60_results/61_black_holes/results_black_holes.py
RESBH_TOML      := 60_results/61_black_holes/61_black_holes.toml
RESBH_STATE     := $(STATE)/61_black_holes
RESBH_NUM       := $(ASSETS)/61_black_holes/61_black_holes_num.tex
RESBH_FIG       := $(ASSETS)/61_black_holes/61_black_holes_fig_errors.tex
RESBH_RENDER    := paperlib/tools/render_assets.py
RESBH_SAVED     := $(STATE)/41_algorithm/black_holes-domain.json $(STATE)/41_algorithm/black_holes-mass.pkl $(STATE)/41_algorithm/black_holes-charge.pkl
RESBH_CEILINGS  := $(STATE)/92_black_holes/reachable_figures.json

ifneq ($(wildcard $(STATE)/41_algorithm/black_holes-domain.json),)
PAPER_ASSETS += $(RESBH_NUM)
endif

# As the domain of validity: the script stores the JSON and writes the wrapper, the render step the numbers.
$(RESBH_STATE)/numbers.json $(RESBH_NUM) $(RESBH_FIG) &: $(RESBH) $(RESBH_RENDER) paper.toml $(RESBH_TOML) $(PAPERLIB) $(STATE)/black_holes.parquet $(STATE)/split.parquet $(RESBH_SAVED) $(RESBH_CEILINGS)
	@$(RUN) python $(RESBH) $(STATE)/black_holes.parquet $(STATE)/split.parquet $(STATE) $(ASSETS)
	@$(RUN) python $(RESBH_RENDER) $(RESBH_STATE) $(ASSETS)
	$(call log_done,error figure wrapper and numbers written to $(@D)/; make figures draws the PNGs)

.PHONY: 61_black_holes
61_black_holes: $(RESBH_NUM) ## Build the black-hole results assets
	$(call log_done,assets current in $(dir $(RESBH_NUM)))

# Every figure variant as its own PNG under build/sections/61_black_holes/figures/, drawn from the
# figure data the main run stored; nothing is computed.
RESBH_FIGURES := $(ASSETS)/61_black_holes/figures

SECTION_FIGURES += 61_black_holes_figures

.PHONY: 61_black_holes_figures
61_black_holes_figures: $(RESBH_STATE)/numbers.json ## Draw every error figure variant as its own PNG under build/sections/61_black_holes/figures/
	@$(RUN) python $(RESBH) --figures $(ASSETS) --state $(STATE)
	$(call log_done,error figure variants in $(RESBH_FIGURES)/)

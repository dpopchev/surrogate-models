# The black-hole data appendix: the BH EDA, the same way; its numbers file stands for the whole set.
BH_EDA        := 90_appendix/92_black_holes/eda_black_holes.py
BH_EDA_TOML   := 90_appendix/92_black_holes/92_black_holes.toml
BH_EDA_ASSETS := $(ASSETS)/92_black_holes/92_black_holes_num.tex
PAPER_ASSETS  += $(BH_EDA_ASSETS)

RENDER_ASSETS := paperlib/tools/render_assets.py
BH_EDA_STATE  := $(STATE)/92_black_holes

# As the neutron-star data appendix: the script stores the JSON and writes the figures, the render step the assets.
$(BH_EDA_STATE)/numbers.json $(BH_EDA_STATE)/reachable_figures.json $(BH_EDA_ASSETS) &: $(BH_EDA) $(RENDER_ASSETS) paper.toml $(BH_EDA_TOML) $(PAPERLIB) $(STATE)/black_holes.parquet
	@$(RUN) python $(BH_EDA) $(STATE)/black_holes.parquet $(ASSETS) --state $(STATE) --folds $(SPLIT_FOLDS) --seed $(SPLIT_SEED)
	@$(RUN) python $(RENDER_ASSETS) $(BH_EDA_STATE) $(ASSETS)
	$(call log_done,BH EDA figure wrappers written to $(@D)/ and tables and numbers to $(BH_EDA_STATE)/; make figures draws the PNGs)

.PHONY: 92_black_holes
92_black_holes: $(BH_EDA_ASSETS) ## Build the black-hole EDA assets
	$(call log_done,black-hole data appendix assets current in $(dir $(BH_EDA_ASSETS)))

# Every figure variant as its own PNG under build/sections/92_black_holes/figures/, drawn from
# the figure data the main run stored; nothing is computed.
BH_EDA_FIGURES := $(ASSETS)/92_black_holes/figures

SECTION_FIGURES += 92_black_holes_figures

.PHONY: 92_black_holes_figures
92_black_holes_figures: $(BH_EDA_STATE)/numbers.json ## Draw every BH EDA figure variant as its own PNG under build/sections/92_black_holes/figures/
	@$(RUN) python $(BH_EDA) --figures $(ASSETS) --state $(STATE)
	$(call log_done,BH EDA figure variants in $(BH_EDA_FIGURES)/)

# Section 3.2: the BH EDA, the same way; its numbers file stands for the whole set.
BH_EDA        := 40_data_analysis/42_black_holes/eda_black_holes.py
BH_EDA_TOML   := 40_data_analysis/42_black_holes/42_black_holes.toml
BH_EDA_ASSETS := $(ASSETS)/42_black_holes/42_black_holes_num.tex
PAPER_ASSETS  += $(BH_EDA_ASSETS)

RENDER_ASSETS := paperlib/tools/render_assets.py
BH_EDA_STATE  := $(STATE)/42_black_holes

# As Section 3.1: the script stores the JSON and writes the figures, the render step the assets.
$(BH_EDA_STATE)/numbers.json $(BH_EDA_ASSETS) &: $(BH_EDA) $(RENDER_ASSETS) paper.toml $(BH_EDA_TOML) $(PAPERLIB) $(STATE)/black_holes.parquet
	@$(RUN) python $(BH_EDA) $(STATE)/black_holes.parquet $(ASSETS) --state $(STATE) --folds $(SPLIT_FOLDS) --seed $(SPLIT_SEED)
	@$(RUN) python $(RENDER_ASSETS) $(BH_EDA_STATE) $(ASSETS)
	$(call log_done,BH EDA figure wrappers written to $(@D)/ and tables and numbers to $(BH_EDA_STATE)/; make figures draws the PNGs)

.PHONY: 42_black_holes
42_black_holes: $(BH_EDA_ASSETS) ## Build the black-hole EDA assets for Section 3.2
	$(call log_done,Section 3.2 assets current in $(dir $(BH_EDA_ASSETS)))

# Every figure variant as its own PNG under build/figures/42_black_holes/, drawn from the
# figure data the main run stored; nothing is computed.
BH_EDA_FIGURES := $(BUILD)/figures

.PHONY: 42_black_holes_figures
42_black_holes_figures: $(BH_EDA_STATE)/numbers.json ## Draw every BH EDA figure variant as its own PNG under build/figures/
	@$(RUN) python $(BH_EDA) --figures $(BH_EDA_FIGURES) --state $(STATE)
	$(call log_done,BH EDA figure variants in $(BH_EDA_FIGURES)/42_black_holes/)

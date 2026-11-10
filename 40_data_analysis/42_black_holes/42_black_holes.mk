# Section 3.2: the BH EDA, the same way; its numbers file stands for the whole set.
BH_EDA        := 40_data_analysis/42_black_holes/eda_black_holes.py
BH_EDA_TOML   := 40_data_analysis/42_black_holes/42_black_holes.toml
BH_EDA_ASSETS := $(ASSETS)/42_black_holes/42_black_holes_num.tex
PAPER_ASSETS  += $(BH_EDA_ASSETS)

$(BH_EDA_ASSETS): $(BH_EDA) paper.toml $(BH_EDA_TOML) $(PAPERLIB) $(STATE)/black_holes.parquet
	@$(RUN) python $(BH_EDA) $(STATE)/black_holes.parquet $(ASSETS) --folds $(SPLIT_FOLDS) --seed $(SPLIT_SEED)
	$(call log_done,BH EDA figures and numbers written to $(@D)/)

.PHONY: 42_black_holes
42_black_holes: $(BH_EDA_ASSETS) ## Build the black-hole EDA assets for Section 3.2
	$(call log_done,Section 3.2 assets current in $(dir $(BH_EDA_ASSETS)))

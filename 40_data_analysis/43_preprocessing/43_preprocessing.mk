# Section 3.3: the split numbers, from the split file (51_algorithms.mk).
PREP_NUMBERS := 40_data_analysis/43_preprocessing/preprocessing_numbers.py
PREP_ASSETS  := $(ASSETS)/43_preprocessing/43_preprocessing_num.tex
PAPER_ASSETS += $(PREP_ASSETS)

$(PREP_ASSETS): $(PREP_NUMBERS) $(STATE)/split.parquet paper.toml
	@$(RUN) python $(PREP_NUMBERS) $(STATE)/split.parquet $(ASSETS) --seed $(SPLIT_SEED)
	$(call log_done,Section 3.3 numbers written to $@)

.PHONY: 43_preprocessing
43_preprocessing: $(PREP_ASSETS) ## Build the split numbers for Section 3.3
	$(call log_done,Section 3.3 assets current in $(dir $(PREP_ASSETS)))

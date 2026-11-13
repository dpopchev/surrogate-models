# The preprocessing appendix: the split numbers, from the split file (95_algorithms.mk).
PREP_NUMBERS := 90_appendix/93_preprocessing/preprocessing_numbers.py
PREP_ASSETS  := $(ASSETS)/93_preprocessing/93_preprocessing_num.tex
PAPER_ASSETS += $(PREP_ASSETS)

RENDER_ASSETS := paperlib/tools/render_assets.py
PREP_STATE    := $(STATE)/93_preprocessing

$(PREP_STATE)/numbers.json: $(PREP_NUMBERS) $(STATE)/split.parquet paper.toml
	@$(RUN) python $(PREP_NUMBERS) $(STATE)/split.parquet $(STATE) --seed $(SPLIT_SEED)
	$(call log_done,preprocessing appendix numbers stored in $@)

$(PREP_ASSETS): $(PREP_STATE)/numbers.json $(RENDER_ASSETS) $(PAPERLIB)
	@$(RUN) python $(RENDER_ASSETS) $(PREP_STATE) $(ASSETS)
	$(call log_done,preprocessing appendix numbers written to $@)

.PHONY: 93_preprocessing
93_preprocessing: $(PREP_ASSETS) ## Build the split numbers
	$(call log_done,preprocessing appendix assets current in $(dir $(PREP_ASSETS)))

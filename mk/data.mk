# mk/data.mk -- the data layer: prepares the initial datasets into runtime state
# under $(STATE) (rules/makefile.md, Directories); build/ products read it as
# prerequisites. clean-state (part of make clean) removes the state and make data
# rebuilds it; the raw input in $(INITIAL_DATA) is never cleaned. Uses the
# base's LOCAL, STATE and log_done and the python layer's RUN.

# ------------------------------------------------------------------------------
### Data
# ------------------------------------------------------------------------------

INITIAL_DATA := $(LOCAL)/initial-data

# Each prepared artifact is a file rule from its script and raw input; DATA_STATE
# collects them for `data` (defined above the rule, since make expands
# prerequisites when it reads one). A missing raw input stops make with its path.
DATA_STATE := $(STATE)/black_holes.parquet $(STATE)/neutron_stars.parquet $(STATE)/split.parquet

BH_SCRIPT := 30_physical_framework/33_black_holes/prepare_black_holes.py
BH_RAW    := $(INITIAL_DATA)/black-holes/black-holes-zero-phi0.dat

$(STATE)/black_holes.parquet: $(BH_SCRIPT) $(BH_RAW) | $(STATE)
	@$(RUN) python $(BH_SCRIPT) $(BH_RAW) $@
	$(call log_done,black-hole table written to $@)

NS_SCRIPT := 30_physical_framework/34_neutron_stars/prepare_neutron_stars.py
NS_RAW    := $(INITIAL_DATA)/neutron-stars/neutron-stars.dat

$(STATE)/neutron_stars.parquet: $(NS_SCRIPT) $(NS_RAW) | $(STATE)
	@$(RUN) python $(NS_SCRIPT) $(NS_RAW) $@
	$(call log_done,neutron-star table written to $@)

# The frozen curve-grouped split of both tables (W-019): the seed is written once
# here (Section 4.3 reports it); test fraction and folds are the script's defaults.
SPLIT_SCRIPT := 50_methodology/51_algorithms/split_datasets.py
SPLIT_SEED   := 20261005

$(STATE)/split.parquet: $(SPLIT_SCRIPT) shared/eda.py $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet | $(STATE)
	@$(RUN) python $(SPLIT_SCRIPT) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $@ --seed $(SPLIT_SEED)
	$(call log_done,curve split written to $@)

.PHONY: data
data: $(DATA_STATE) ## Prepare the datasets into local/state/
	$(call log_done,data state current in $(STATE)/ -- $(words $(DATA_STATE)) artifacts)

# mk/data.mk -- the data layer: prepares the initial datasets into runtime state
# under $(STATE). State is machine-local and survives make clean (CLAUDE.md, Gate
# deviations); build/ products read it as prerequisites. Uses the base's LOCAL,
# log_done and the python layer's RUN.

# ------------------------------------------------------------------------------
### Data
# ------------------------------------------------------------------------------

STATE        := $(LOCAL)/state
INITIAL_DATA := $(LOCAL)/initial-data

# Each prepared artifact is a file rule from its script and raw input; DATA_STATE
# collects them for `data` (defined above the rule, since make expands
# prerequisites when it reads one). A missing raw input stops make with its path.
DATA_STATE := $(STATE)/black_holes.parquet

BH_SCRIPT := 30_physical_framework/32_black_holes/prepare_black_holes.py
BH_RAW    := $(INITIAL_DATA)/black-holes/black-holes-zero-phi0.dat

$(STATE)/black_holes.parquet: $(BH_SCRIPT) $(BH_RAW)
	@$(RUN) python $(BH_SCRIPT) $(BH_RAW) $@
	$(call log_done,black-hole table written to $@)

.PHONY: data
data: $(DATA_STATE) ## Prepare the datasets into local/state/
	$(call log_done,data state current in $(STATE)/ -- $(words $(DATA_STATE)) artifacts)

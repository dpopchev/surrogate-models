# mk/data.mk -- the data layer: prepares the initial datasets into runtime state
# under $(STATE). State is machine-local and survives make clean (CLAUDE.md, Gate
# deviations); build/ products read it as prerequisites. Uses the base's LOCAL,
# log_done and the python layer's RUN.

# ------------------------------------------------------------------------------
### Data
# ------------------------------------------------------------------------------

STATE        := $(LOCAL)/state
INITIAL_DATA := $(LOCAL)/initial-data

DATA_STATE :=

.PHONY: data
data: $(DATA_STATE) ## Prepare the datasets into local/state/
	$(call log_done,data state current in $(STATE)/ -- $(words $(DATA_STATE)) artifacts)

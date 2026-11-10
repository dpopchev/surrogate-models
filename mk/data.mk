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

# The rules that write them live with their sections: the black-hole table in
# 33_black_holes.mk, the neutron-star table in 34_neutron_stars.mk and the split in
# 95_algorithms.mk, included by the Makefile after mk/*.mk.

.PHONY: data
data: $(DATA_STATE) ## Prepare the datasets into local/state/
	$(call log_done,data state current in $(STATE)/ -- $(words $(DATA_STATE)) artifacts)

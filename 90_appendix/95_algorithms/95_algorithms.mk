# The baseline appendix: the frozen split, the baseline surrogate, its table and the tools that read its
# runs and ledger. Uses the base's STATE and log helpers and PAPERLIB (mk/paper.mk).

# The frozen curve-grouped split of both tables (W-019): its seed, test fraction and folds are
# written once here (the preprocessing appendix reports them; the EDA probes of the sections reuse them).
SPLIT_SCRIPT        := 90_appendix/95_algorithms/split_datasets.py
SPLIT_SEED          := 20261005
SPLIT_FOLDS         := 5
SPLIT_TEST_FRACTION := 0.15

$(STATE)/split.parquet: $(SPLIT_SCRIPT) paperlib/eda.py $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet | $(STATE)
	@$(RUN) python $(SPLIT_SCRIPT) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $@ --seed $(SPLIT_SEED) --folds $(SPLIT_FOLDS) --test-fraction $(SPLIT_TEST_FRACTION)
	$(call log_done,curve split written to $@)

# The baseline surrogate trains a network, minutes on this CPU, so it is its own
# target and not in PAPER_ASSETS -- make check (paper-verify) would retrain it on every change
# of a shared module (T-080). The numbers file stands for the whole set; the epoch lines stream
# while it trains.
BASELINE        := 90_appendix/95_algorithms/fit_baseline.py
BASELINE_TOML   := 90_appendix/95_algorithms/95_algorithms.toml
BASELINE_ASSETS := $(ASSETS)/95_algorithms/95_algorithms_num.tex

# The script stores the table and numbers as JSON under $(STATE)/95_algorithms/ and still writes
# the parity figure (after emptying the asset folder); render_assets.py writes the table and
# numbers assets from the JSON. One grouped rule, so a build that lost the numbers reruns the fit.
RENDER_ASSETS  := paperlib/tools/render_assets.py
BASELINE_STATE := $(STATE)/95_algorithms

$(BASELINE_STATE)/numbers.json $(BASELINE_ASSETS) &: $(BASELINE) $(RENDER_ASSETS) paper.toml $(BASELINE_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
	$(call log_info,fitting the baseline on NS and BH$(comma) mass and charge -- one line per epoch; make follow shows it from any terminal)
	@$(RUN) python $(BASELINE) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE)
	@$(RUN) python $(RENDER_ASSETS) $(BASELINE_STATE) $(ASSETS)
	$(call log_done,baseline table and numbers rendered and parity wrapper written to $(@D)/ -- run diagnostics under $(STATE)/95_algorithms/)

.PHONY: baseline
baseline: $(BASELINE_ASSETS) ## Fit the baseline surrogate
	$(call log_done,baseline assets current in $(dir $(BASELINE_ASSETS)))

# The baseline table alone is a paper asset: rebuilt from the latest baseline batch of the
# ledger in seconds, nothing fitted, so a change to baseline_table reaches the paper without a
# refit (W-128). A ledger without a baseline batch keeps the table as make baseline last wrote it.
BASELINE_TABLE := $(ASSETS)/95_algorithms/95_algorithms_tab_baseline.tex
BASELINE_DATA  := $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet
PAPER_ASSETS   += $(BASELINE_TABLE)

$(BASELINE_TABLE): $(BASELINE) $(RENDER_ASSETS) paper.toml $(BASELINE_TOML) $(PAPERLIB) $(BASELINE_DATA) $(wildcard $(STATE)/ledger)
	@if out=$$($(RUN) python $(BASELINE) $(BASELINE_DATA) $(ASSETS) $(STATE) --tables 2>&1) \
	  && out=$$($(RUN) python $(RENDER_ASSETS) $(BASELINE_STATE) $(ASSETS) 2>&1); then \
	  printf "$(BOLD)$(GREEN)%s$(RESET) %s\n" "[DONE]" "baseline table rebuilt from the ledger -> $@" >&2; \
	elif printf '%s' "$$out" | grep -q 'no baseline batch in the ledger'; then \
	  printf "$(BOLD)$(YELLOW)%s$(RESET) %s\n" "[WARN]" "no baseline batch in the ledger -- $@ kept; make baseline fits one" >&2; \
	else printf '%s\n' "$$out" >&2; exit 1; fi

# The section's assets are the split and the baseline table; the fit itself is make baseline.
.PHONY: 95_algorithms
95_algorithms: $(STATE)/split.parquet $(BASELINE_TABLE) ## Build the split and the baseline table -- nothing fitted
	$(call log_done,baseline appendix assets current in $(dir $(BASELINE_TABLE)))

# Each baseline run logs into its own <run>/train.log and repoints latest.log at it (W-054);
# tail -F follows the name, so one make follow keeps up across runs. It ends with Ctrl-C, so
# its log_done never prints.
BASELINE_LOG := $(STATE)/95_algorithms/latest.log

.PHONY: follow
follow: ## Follow the running baseline's log live
	$(call log_info,following $(BASELINE_LOG) -- waits for a run if none started yet; Ctrl-C ends it)
	@tail -n 40 -F $(BASELINE_LOG)
	$(call log_done,stopped following $(BASELINE_LOG))

# The runs at a glance (W-057): one line per run from its run.json, newest first; make run
# lists one run's files, the latest unless R names a run folder.
LIST_RUNS := 90_appendix/95_algorithms/list_runs.py
R         ?= latest

.PHONY: runs
runs: ## List the baseline runs, newest first
	@$(RUN) python $(LIST_RUNS) $(STATE)
	$(call log_done,baseline runs listed from $(STATE)/95_algorithms/)

.PHONY: run
run: ## Show one baseline run's files -- R=<run>, default latest
	@$(RUN) python $(LIST_RUNS) $(STATE) --run $(R)
	$(call log_done,files of baseline run $(R) listed)

# The run ledger (W-077): one line per harness run of every make baseline, newest first; B=<batch
# id or its first characters> lists one batch.
LIST_LEDGER := 90_appendix/95_algorithms/list_ledger.py
LEDGER      := $(STATE)/ledger
B           ?=

.PHONY: ledger
ledger: ## List the run ledger -- B=<batch> for one batch
	@$(RUN) python $(LIST_LEDGER) $(STATE) $(if $(B),--batch $(B))
	$(call log_done,ledger listed from $(LEDGER)/ -- $(words $(wildcard $(LEDGER)/*.json)) entries$(if $(B), in all$(comma) batch $(B) shown))

# Every parity variant as its own PNG under build/sections/95_algorithms/figures/, redrawn from
# the baseline batch recorded in the ledger; nothing is fitted.
BASELINE_FIGURES := $(ASSETS)/95_algorithms/figures

SECTION_FIGURES += 95_algorithms_figures

.PHONY: 95_algorithms_figures
95_algorithms_figures: $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet ## Draw every baseline figure variant as its own PNG under build/sections/95_algorithms/figures/
	@$(RUN) python $(BASELINE) $(STATE)/neutron_stars.parquet $(STATE)/black_holes.parquet $(STATE)/split.parquet $(ASSETS) $(STATE) --figures $(ASSETS)
	$(call log_done,baseline figure variants in $(BASELINE_FIGURES)/)

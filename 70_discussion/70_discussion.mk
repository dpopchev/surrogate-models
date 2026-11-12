# Section 6: the maximum neutron-star mass the surrogate predicts (W-156). The figure reads the
# saved mass surrogate of $(STATE)/41_algorithm/ (make pipeline), so its assets join make assets
# only while that exists: make pipeline fits for minutes and is no prerequisite.
DISC            := 70_discussion/mass_max.py
DISC_TOML       := 70_discussion/70_discussion.toml
DISC_STATE      := $(STATE)/70_discussion
DISC_NUM        := $(ASSETS)/70_discussion/70_discussion_num.tex
DISC_FIG        := $(ASSETS)/70_discussion/70_discussion_fig_mass_max.tex
DISC_RENDER     := paperlib/tools/render_assets.py
DISC_SAVED      := $(STATE)/41_algorithm/neutron_stars-mass.pkl
DISC_CEILINGS   := $(STATE)/91_neutron_stars/reachable_figures.json

ifneq ($(wildcard $(STATE)/41_algorithm/neutron_stars-mass.pkl),)
PAPER_ASSETS += $(DISC_NUM)
endif

# As Section 5.2: the script stores the JSON and writes the wrapper, the render step the numbers.
$(DISC_STATE)/numbers.json $(DISC_NUM) $(DISC_FIG) &: $(DISC) $(DISC_RENDER) paper.toml $(DISC_TOML) $(PAPERLIB) $(STATE)/neutron_stars.parquet $(STATE)/split.parquet $(DISC_SAVED) $(DISC_CEILINGS)
	@$(RUN) python $(DISC) $(STATE)/neutron_stars.parquet $(STATE)/split.parquet $(STATE) $(ASSETS)
	@$(RUN) python $(DISC_RENDER) $(DISC_STATE) $(ASSETS)
	$(call log_done,maximum-mass figure wrapper and numbers written to $(@D)/; make figures draws the PNGs)

.PHONY: 70_discussion
70_discussion: $(DISC_NUM) ## Build the maximum-mass assets for Section 6
	$(call log_done,assets current in $(dir $(DISC_NUM)))

# Every figure variant as its own PNG under build/sections/70_discussion/figures/, drawn from the
# figure data the main run stored; nothing is computed.
DISC_FIGURES := $(ASSETS)/70_discussion/figures

SECTION_FIGURES += 70_discussion_figures

.PHONY: 70_discussion_figures
70_discussion_figures: $(DISC_STATE)/numbers.json ## Draw every maximum-mass figure variant as its own PNG under build/sections/70_discussion/figures/
	@$(RUN) python $(DISC) --figures $(ASSETS) --state $(STATE)
	$(call log_done,maximum-mass figure variants in $(DISC_FIGURES)/)

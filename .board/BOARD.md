# Board -- surrogate-models
record: local
open path: E-009   | blocked: 1 | todo roots: 3 | done: 152 | pruned: 4

## Tree

### E-005 [doing] Baseline surrogate through one shared pipeline

#### W-034 issue [done] NS mass baseline runs end to end with live feedback

- T-076 [done] Add torch and skorch -> pyproject.toml and uv.lock
- T-077 [done] Write the design-matrix contract test-first -> shared/design.py with shared/test_design.py
- T-078 [done] Write the estimator, metrics and training callbacks test-first -> shared/surrogate.py with shared/test_surrogate.py
- T-079 [done] Add the baseline settings to paper.toml -> a typed methodology section in shared/config.py
- T-080 [done] Write the baseline fit script and make baseline -> build/assets/51_algorithms/
- T-090 [done] Restore the best epoch's weights at early stopping -> shared/surrogate.py
- T-091 [done] Print the baseline error macros in scientific notation -> 50_methodology/51_algorithms/fit_baseline.py

#### W-035 issue [done] Each baseline run leaves a diagnostics folder

- T-081 [done] Write the run record test-first -> shared/runs.py with shared/test_runs.py
- T-082 [done] Write the loss curve and the error CDF test-first -> shared/diagnostics.py with shared/test_diagnostics.py
- T-083 [done] Write the worst and median curve overlays test-first -> shared/diagnostics.py
- T-084 [done] Wire the diagnostics into the baseline run -> local/state/51_algorithms/<run>/
- T-092 [done] Flag a run made from uncommitted code -> shared/runs.py and fit_baseline.py

#### W-036 issue [todo] Baseline table for four pairs with fold scores and reference predictors

- T-085 [todo] Write the reference predictors test-first -> shared/surrogate.py
- T-086 [todo] Run the four pairs with fold scores -> build/assets/51_algorithms/51_algorithms_tab_baseline.tex
- T-087 [todo] Add the four-pair baseline table to Section 5.1 -> 50_methodology/51_algorithms/51_algorithms.tex

#### W-037 issue [todo] Charge rebuilt with the true and the predicted mass

- T-088 [todo] Write the charge rebuild test-first -> shared/surrogate.py
- T-089 [todo] Write the charge MARE with true and predicted mass -> 51_algorithms_num.tex and Section 5.1

#### W-054 issue [done] A running fit is visible live from the developer's terminal

- T-093 [done] Write each run's log to its folder -> <run>/train.log and latest.log
- T-094 [done] Log a fit progress line with elapsed and remaining time -> fit_baseline.py
- T-095 [done] Add make follow -> tail the running baseline's log

#### W-055 issue [done] Section 5.1 describes the pipeline and the NS mass baseline

- T-097 [done] Load the baseline numbers only when present -> 00_metadata/article.tex and 51_algorithms.tex
- T-098 [done] Write the Section 5.1 pipeline and baseline text -> 50_methodology/51_algorithms/51_algorithms.tex

#### W-056 issue [done] The epoch table names its time column elapse_s

- T-096 [done] Print the epoch time as elapse_s -> shared/surrogate.py

#### W-057 issue [done] A run tells how far it is, when it may stop and where its outputs go

- T-099 [done] Show k/MAX, epochs since best and the time-left range in the epoch table -> shared/surrogate.py
- T-100 [done] Log a start and an end banner for each run -> 50_methodology/51_algorithms/fit_baseline.py
- T-101 [done] Add make runs and make run -> 50_methodology/51_algorithms/list_runs.py and mk/paper.mk
- T-102 [done] Add make examples, the common commands with examples -> mk/paper.mk
- T-104 [done] Refine the run log from the developer's first look -> shared/surrogate.py and fit_baseline.py
- T-105 [done] Two stop-time columns, the stop criterion on top and a live loss curve -> shared/surrogate.py and fit_baseline.py
- T-106 [done] Print the epoch table at 4 significant figures, elapsed_s to a tenth -> shared/surrogate.py

### E-006 [todo] Data-guided choices tested -- rho_c input, H1, H2 and H3

#### W-038 issue [todo] rho_c input check -- raw against log10

#### W-039 issue [todo] H1 -- log against linear charge target

#### W-040 issue [todo] H2 -- loss x activation at the M_max turning point

#### W-041 issue [todo] H3a -- model pool GPR, XGBoost, MLP and ResNet

#### W-042 issue [todo] H3b -- ripple metric of MLP against ResNet

#### W-043 issue [todo] H3c -- GPR scaling wall

### E-007 [todo] Training refinement, ensemble uncertainty and the final surrogate

#### W-044 spike [todo] Is L-BFGS fine-tuning worth keeping

#### W-045 issue [todo] Ensemble uncertainty over curve-bootstrap resamples

#### W-046 issue [todo] Final surrogate saved and loadable

### E-008 [todo] Speedup and the MCMC application

#### W-047 spike [todo] Speedup baseline and reference posterior without the solver

#### W-048 issue [todo] Pareto front of error against inference time

#### W-049 issue [todo] Mock MCMC recovery with emcee

#### W-050 issue [todo] Surrogate posterior validated against the reference

### E-009 [doing] The paper follows the revised structure, and the release

#### W-051 issue [done] Sections 2.1 to 2.4 written from the revised structure

- T-109 [done] Add Section 2.2's folder and renumber the dataset sections -> 30_physical_framework/32_scalarization, 33_black_holes, 34_neutron_stars
- T-110 [done] Write Section 2.1, the action and field equations -> 30_physical_framework/31_action/31_action.tex
- T-111 [done] Write Section 2.2, the scalarization mechanism -> 30_physical_framework/32_scalarization/32_scalarization.tex
- T-112 [done] Write Section 2.3, the black-hole dataset -> 30_physical_framework/33_black_holes/33_black_holes.tex
- T-113 [done] Write Section 2.4, the neutron-star dataset -> 30_physical_framework/34_neutron_stars/34_neutron_stars.tex

#### W-052 issue [todo] Abstract, introduction, conclusion and title block

#### W-053 spike [todo] What the open-source release publishes

#### W-059 spike [blocked] Which refs.bib keys the revised structure's citations name   [blocked since 2026-10-06, waiting on developer: 14 missing references need sources from the developer; option B lets Section 2 go first with visible markers]

- T-107 [done] Map the numbered citations to refs.bib keys -> 00_metadata/citations.md
- T-108 [doing] Add the missing references to refs.bib -> 00_metadata/refs.bib

#### W-060 issue [done] Title and outline bullets aligned to the revised structure

- T-114 [done] Set the revised title -> 00_metadata/metadata.tex
- T-115 [done] Align the remaining outline bullets to the revised structure -> the todo bullets of 10, 20, 41, 44, 51-53, 61-64, 70, 80
- T-116 [done] Name paper sections by their compiled number in code and make -> mk/, paper.toml, the colocated modules

## Closed

### E-001 [done] Overleaf-ready article built from the chapter folders

closed 2026-10-04 -- outcome: One command produces an Overleaf-ready article from the agreed skeleton, including assets generated by scripts. -- ledger: ledgers/E-001.md

#### W-001 issue [done] Skeleton article compiles from the chapter folders

- T-001 [done] Write the chapter skeleton and shared LaTeX setup -> 00_metadata/ and one <dir>/<dir>.tex per section
- T-002 [done] Add the paper layer -> mk/paper.mk compile builds build/paper/article/main.pdf

#### W-002 issue [done] A script beside its section generates an asset the article uses

- T-003 [done] Add the Python toolchain as a non-package project -> pyproject.toml, mk/python.mk, make check green
- T-004 [done] Write the build stamp script test-first -> 00_metadata/build_stamp.py with test_build_stamp.py
- T-005 [done] Wire generated assets into compile -> build stamp on the title page of main.pdf

#### W-003 issue [done] The article zip is a self-contained Overleaf project

- T-006 [done] Add the zip step -> build/paper/article.zip with main.tex at its root
- T-007 [done] Add the clean-room check -> make paper-verify compiles the unpacked zip

#### W-004 issue [done] build/paper/article/ is a clean upload folder

- T-008 [done] Move LaTeX residue out of the upload folder -> build/paper/article/ sources only, PDF at build/paper/article.pdf
- T-009 [done] Verify the upload folder in a clean room -> paper-verify compiles a copy of build/paper/article/

### E-002 [done] Data audit -- preprocessed datasets behind Sections 2 and 3

closed 2026-10-05 -- outcome: Preprocessed BH and NS tables, a frozen curve-grouped split, and the Section 3.1 numbers rendered from macros, all rebuilt by make from local/initial-data. -- ledger: ledgers/E-002.md

#### W-007 issue [done] BH table from the zero-phi0 dataset

- T-014 [done] Add data dependencies and the state home -> pandas + pyarrow, mk/data.mk STATE, CLAUDE.md gate deviation
- T-015 [done] Write the BH preparation test-first -> 30_physical_framework/32_black_holes/prepare_black_holes.py with its tests
- T-016 [done] Wire the BH preparation into make -> local/state/black_holes.parquet

#### W-008 spike [done] How neutron-stars.dat is laid out and how bad runs show up

- T-017 [done] Probe the NS file layout and failure signatures -> local/scratch/probe_neutron_stars.py and Findings on W-008
- T-018 [done] Decide the NS parsing and filtering rules -> spike.decision on W-008

#### W-009 issue [done] NS table from the filtered dataset

- T-019 [done] Write the NS preparation test-first -> 30_physical_framework/33_neutron_stars/prepare_neutron_stars.py with its tests
- T-020 [done] Wire the NS preparation into make -> local/state/neutron_stars.parquet

#### W-010 issue [done] Frozen split grouped by curve

- T-021 [done] Write the curve-grouped split test-first -> 50_methodology/51_algorithms/split_datasets.py with its tests
- T-022 [done] Wire the split into make with its invariant -> local/state/split.parquet, no curve in two sets

#### W-011 issue [done] Section 3.1 numbers from generated macros

- T-023 [pruned] Write the EDA numbers test-first -> 40_data_analysis/41_eda/eda_numbers.py with its tests   [pruned: superseded: W-017's eda_neutron_stars.py (and W-018's eda_black_holes.py) already generate every Section 3.1 and 3.2 number as macros]
- T-024 [pruned] Render Section 3.1 from the macros -> 41_eda.tex uses generated numbers and states the EDA scope   [pruned: superseded: Section 3.1 was rewritten from the nsEda macros in W-017 (T-054), developer-approved]

#### W-012 bug [done] BH parser fails on repeated block headers and blank lines   (filed during T-016)

- T-025 [done] Record the repeated-header case as a failing test -> test_prepare_black_holes.py RED
- T-026 [done] Skip repeated headers and blank lines in parse_table -> prepare_black_holes.py GREEN

#### W-019 spike [done] Which preprocessing, targets and split each dataset's EDA supports

- T-046 [done] Decide the NS decisions rows with the developer -> NS part of W-019 spike.decision
- T-047 [done] Decide the BH decisions rows with the developer -> BH part of W-019 spike.decision

#### W-020 issue [done] Section 4.3 preprocessing and feature decisions tables

- T-048 [done] Write the NS decisions table -> 40_data_analysis/43_preprocessing/43_preprocessing.tex
- T-049 [done] Write the BH decisions table -> 40_data_analysis/43_preprocessing/43_preprocessing.tex
- T-062 [done] Generate the Section 3.3 numbers test-first -> 40_data_analysis/43_preprocessing/preprocessing_numbers.py
- T-064 [done] Report raw M everywhere in the BH analysis -> eda_black_holes.py, Sections 3.2 and 3.3

#### W-025 bug [done] Two generated EDA tables overflow the text width   (filed during T-048)

- T-063 [done] Make the generated EDA tables fit the text width -> shared/eda.py booktabs and the table headers

#### W-026 issue [done] BH mass figure shows where the curves leave GR

- T-065 [done] Add the GR inset to the BH mass figure -> eda_black_holes.py and Section 3.2

### E-003 [done] Paper config, shared plot style and the per-dataset EDA figures

closed 2026-10-05 -- outcome: shared/plots.py and shared/config.py give a typed PaperConfig and one dataset-aware plot style; Section 4 holds an NS EDA, a BH EDA, a preprocessing-decisions and a hypotheses subsection; the NS and BH EDA scripts beside their sections write the figures paper.toml selects into build/assets, and make compile includes them. -- ledger: ledgers/E-003.md

#### W-013 issue [done] Section choices loaded from paper.toml

- T-027 [done] Add pydantic-settings -> pyproject.toml and uv.lock
- T-028 [done] Write the config loader test-first -> shared/config.py with shared/test_config.py
- T-029 [done] Write paper.toml with the plot and NS EDA sections -> paper.toml

#### W-014 issue [done] One paper-wide plot style with a color family per dataset

- T-030 [done] Add matplotlib and seaborn -> pyproject.toml and uv.lock
- T-031 [done] Write the plot style test-first -> shared/plots.py with shared/test_plots.py
- T-032 [pruned] Add the plot section and render a style sample -> paper.toml and local/scratch/style_sample.pdf   [pruned: replaced by the re-plan: the [plot] section moves into T-029 and the visual review happens on the NS and BH EDA figures (T-041, T-045), not a separate sample]

#### W-016 issue [done] Section 4 split into NS EDA, BH EDA, preprocessing decisions and hypotheses

- T-036 [done] Move the existing Section 4 subsections -> 40_data_analysis/41_neutron_stars and 44_hypotheses
- T-037 [done] Add the BH EDA and preprocessing subsections -> 40_data_analysis/42_black_holes and 43_preprocessing

#### W-017 issue [done] NS EDA figures and numbers selected in paper.toml

- T-038 [done] Write the NS EDA computations test-first -> 40_data_analysis/41_neutron_stars/eda_neutron_stars.py with its tests
- T-039 [done] Draw the NS EDA figures selected in paper.toml -> eda_neutron_stars.py figure functions and main
- T-040 [done] Wire the NS EDA into make and Section 4.1 -> mk/paper.mk rules and 41_neutron_stars.tex
- T-041 [done] Review the NS EDA figures with the developer -> Findings on W-017
- T-052 [done] Compare split strategies on the NS table test-first -> split_strategies in eda_neutron_stars.py
- T-053 [done] Add the NS tables and the extending figures selected in paper.toml -> eda_neutron_stars.py and shared/config.py
- T-054 [done] Write the NS insight paragraphs -> 40_data_analysis/41_neutron_stars/41_neutron_stars.tex

#### W-018 issue [done] BH EDA figures and numbers selected in paper.toml

- T-042 [pruned] Write the BH EDA computations test-first -> 40_data_analysis/42_black_holes/eda_black_holes.py with its tests   [pruned: superseded by the developer's re-plan of 2026-10-05: the shared EDA core is extracted first (T-055) and the BH computations are re-briefed to the BH-specific analysis (T-056)]
- T-043 [done] Draw the BH EDA figures selected in paper.toml -> eda_black_holes.py figure functions and main
- T-044 [done] Wire the BH EDA into make and Section 4.2 -> mk/paper.mk rules and 42_black_holes.tex
- T-045 [done] Review the BH EDA figures with the developer -> Findings on W-018
- T-055 [done] Extract the dataset-agnostic EDA core -> shared/eda.py with shared/test_eda.py
- T-056 [done] Write the BH EDA computations test-first -> 40_data_analysis/42_black_holes/eda_black_holes.py with its tests
- T-057 [done] Write the BH insight paragraphs -> 40_data_analysis/42_black_holes/42_black_holes.tex

#### W-022 bug [done] The NS EDA make rule targets a file the script no longer writes   (filed during T-044)

- T-058 [done] Point the NS EDA rule at its numbers file -> mk/paper.mk

#### W-023 bug [done] Python bytecode caches are tracked in git   (filed during T-057)

- T-059 [done] Untrack and delete the bytecode caches -> a tree without __pycache__

### E-004 [done] The paper builds for Overleaf upload and section-by-section review

closed 2026-10-06 -- outcome: build/assets/ grouped by section with each figure's PNG and wrapper sharing a stem; make overleaf filling build/paper/overleaf/ (flat, plus overleaf.zip) and listing the files added and removed since the last upload; make sections writing one PDF per section with references resolved through the full build; build/paper/article.pdf stays the whole paper. -- ledger: ledgers/E-004.md

#### W-031 issue [done] build/ grouped by section

- T-070 [done] Write each section's assets into its own folder test-first -> both EDA modules and preprocessing_numbers.py
- T-071 [done] Add make assets and move LaTeX residue to build/.work -> mk/paper.mk

#### W-032 issue [done] make overleaf keeps a drag-and-drop Overleaf project in sync

- T-072 [done] Diff the upload folder against the last upload test-first -> 00_metadata/overleaf_upload.py
- T-073 [done] Wire make overleaf -> mk/paper.mk, build/paper/overleaf/ and overleaf.zip

#### W-033 issue [done] One PDF per section with references resolved

- T-074 [done] Write the section entry files test-first -> 00_metadata/section_entry.py
- T-075 [done] Wire make section and make sections -> mk/paper.mk and build/paper/sections/

### W-005 issue [done] (standalone) Python layer follows the COLOCATED level

closed 2026-10-04 -- outcome: Given the COLOCATED mechanics of \~/.claude/rules/python.md, When make check runs, Then it passes with the repository root on pytest pythonpath and the python files matching the add-python COLOCATED templates. -- ledger: ledgers/W-005.md

- T-010 [done] Put the repository root on pytest pythonpath -> pyproject.toml pythonpath = ["."]
- T-011 [done] Align the python files with the add-python COLOCATED templates -> pyproject.toml, mk/python.mk, conftest.py

### W-006 issue [done] (standalone) Python dependencies resolve CPU-only for this machine

closed 2026-10-04 -- outcome: Given pyproject.toml, When torch is added with uv, Then it resolves from https://download.pytorch.org/whl/cpu and the lock holds no nvidia, CUDA or triton package. -- ledger: ledgers/W-006.md

- T-012 [done] Pin torch to the PyTorch CPU index -> pyproject.toml [[tool.uv.index]] and [tool.uv.sources]
- T-013 [done] Prove torch resolves CPU-only -> a scratch copy of pyproject.toml locks torch +cpu with no GPU packages

### W-015 issue [done] (standalone) Paper text for the D floor and the sign symmetry of D

closed 2026-10-05 -- outcome: Given the E-002 D-floor and sign-symmetry Decisions and the Section 3.1 macros, When make compile runs, Then H1, the EDA D bullet and Section 2 state the floor eps, the share of rows below it and the D -> -D symmetry, with no hard-coded number. -- ledger: ledgers/W-015.md

- T-033 [done] Add the D-floor macros -> eps and the share below it in the Section 3.1 macro output
- T-034 [done] Restate H1 and the EDA D bullet for the floor -> 42_hypotheses.tex, 41_eda.tex, 61_h1.tex
- T-035 [done] State the sign symmetry of D in Section 2 -> 30_physical_framework/31_action/31_action.tex

### W-021 issue [done] (standalone) Review markers in the paper with todonotes

closed 2026-10-05 -- outcome: Given the preamble with todonotes, When make compile runs, Then every \todo and every \review note renders inline in its own colour and the upload folder compiles on its own. -- ledger: ledgers/W-021.md

- T-050 [done] Replace the hand-made todo macro with todonotes -> 00_metadata/preamble.tex
- T-051 [done] Add the review note kind and mark the open claims -> preamble.tex and the review notes

### W-024 issue [done] (standalone) Runtime state follows the global local/state rule

closed 2026-10-05 -- outcome: Given the base Makefile with STATE and no local/state Gate deviation, When make clean and then make data run, Then make data rebuilds no state file. -- ledger: ledgers/W-024.md

- T-060 [done] Move STATE into the base Makefile -> Makefile and mk/data.mk
- T-061 [done] Remove the local/state Gate deviation -> CLAUDE.md

### W-027 issue [done] (standalone) make clean runs every area clean, the state included

closed 2026-10-05 -- outcome: Given local/initial-data and a built local/state, When make clean runs, Then local/state is gone and local/initial-data is untouched. -- ledger: ledgers/W-027.md

- T-066 [done] Adopt the area cleans -> Makefile, mk/python.mk and mk/data.mk

### W-028 issue [done] (standalone) Outline text cites stale charge and row numbers

closed 2026-10-05 -- outcome: Given the generated EDA and preprocessing macros, When make compile runs, Then the abstract and Section 2.3 cite the charge range and row counts only through generated macros. -- ledger: ledgers/W-028.md

- T-067 [done] Cite the generated numbers in the abstract and Section 2.3 -> 10_abstract.tex and 33_neutron_stars.tex

### W-029 issue [done] (standalone) Figures render as high-resolution PNG

closed 2026-10-05 -- outcome: Given paper.toml with plot.dpi, When make compile runs, Then every selected EDA figure reaches the PDF as a PNG written at that resolution. -- ledger: ledgers/W-029.md

- T-068 [done] Write the EDA figures as PNG at plot.dpi -> shared/plots.py, both EDA modules, paper.toml

### W-030 issue [done] (standalone) Figures at 450 dpi to keep the paper small

closed 2026-10-05 -- outcome: Given paper.toml with plot.dpi = 450, When make compile runs, Then the figures in build/assets are PNGs written at 450 dpi. -- ledger: ledgers/W-030.md

- T-069 [done] Set plot.dpi to 450 -> paper.toml

### W-058 issue [done] (standalone) Landing settings follow the board tool of 2026-10-06

closed 2026-10-06 -- outcome: Given the board skill's current [landing] keys, When the developer runs make board-status, Then the landing line prints with no [WARN]. -- ledger: ledgers/W-058.md

- T-103 [done] Review [landing] and record the worktree decision -> .board/board.toml

## Diagram
```mermaid
flowchart TD
  E005["E-005 doing: Baseline surrogate through one shared pipeline"]
  E005 --> W034["W-034 issue done: NS mass baseline runs end to end with live feedback"]
  W034 --> T076["T-076 done: Add torch and skorch -> pyproject.toml and uv.lock"]
  W034 --> T077["T-077 done: Write the design-matrix contract test-first -> shared/design.py with shared/test_design.py"]
  W034 --> T078["T-078 done: Write the estimator, metrics and training callbacks test-first -> shared/surrogate.py with shared/test_surrogate.py"]
  W034 --> T079["T-079 done: Add the baseline settings to paper.toml -> a typed methodology section in shared/config.py"]
  W034 --> T080["T-080 done: Write the baseline fit script and make baseline -> build/assets/51_algorithms/"]
  W034 --> T090["T-090 done: Restore the best epoch's weights at early stopping -> shared/surrogate.py"]
  W034 --> T091["T-091 done: Print the baseline error macros in scientific notation -> 50_methodology/51_algorithms/fit_baseline.py"]
  E005 --> W035["W-035 issue done: Each baseline run leaves a diagnostics folder"]
  W035 --> T081["T-081 done: Write the run record test-first -> shared/runs.py with shared/test_runs.py"]
  W035 --> T082["T-082 done: Write the loss curve and the error CDF test-first -> shared/diagnostics.py with shared/test_diagnostics.py"]
  W035 --> T083["T-083 done: Write the worst and median curve overlays test-first -> shared/diagnostics.py"]
  W035 --> T084["T-084 done: Wire the diagnostics into the baseline run -> local/state/51_algorithms/<run>/"]
  W035 --> T092["T-092 done: Flag a run made from uncommitted code -> shared/runs.py and fit_baseline.py"]
  E005 --> W036["W-036 issue todo: Baseline table for four pairs with fold scores and reference predictors"]
  W036 --> T085["T-085 todo: Write the reference predictors test-first -> shared/surrogate.py"]
  W036 --> T086["T-086 todo: Run the four pairs with fold scores -> build/assets/51_algorithms/51_algorithms_tab_baseline.tex"]
  W036 --> T087["T-087 todo: Add the four-pair baseline table to Section 5.1 -> 50_methodology/51_algorithms/51_algorithms.tex"]
  E005 --> W037["W-037 issue todo: Charge rebuilt with the true and the predicted mass"]
  W037 --> T088["T-088 todo: Write the charge rebuild test-first -> shared/surrogate.py"]
  W037 --> T089["T-089 todo: Write the charge MARE with true and predicted mass -> 51_algorithms_num.tex and Section 5.1"]
  E005 --> W054["W-054 issue done: A running fit is visible live from the developer's terminal"]
  W054 --> T093["T-093 done: Write each run's log to its folder -> <run>/train.log and latest.log"]
  W054 --> T094["T-094 done: Log a fit progress line with elapsed and remaining time -> fit_baseline.py"]
  W054 --> T095["T-095 done: Add make follow -> tail the running baseline's log"]
  E005 --> W055["W-055 issue done: Section 5.1 describes the pipeline and the NS mass baseline"]
  W055 --> T097["T-097 done: Load the baseline numbers only when present -> 00_metadata/article.tex and 51_algorithms.tex"]
  W055 --> T098["T-098 done: Write the Section 5.1 pipeline and baseline text -> 50_methodology/51_algorithms/51_algorithms.tex"]
  E005 --> W056["W-056 issue done: The epoch table names its time column elapse_s"]
  W056 --> T096["T-096 done: Print the epoch time as elapse_s -> shared/surrogate.py"]
  E005 --> W057["W-057 issue done: A run tells how far it is, when it may stop and where its outputs go"]
  W057 --> T099["T-099 done: Show k/MAX, epochs since best and the time-left range in the epoch table -> shared/surrogate.py"]
  W057 --> T100["T-100 done: Log a start and an end banner for each run -> 50_methodology/51_algorithms/fit_baseline.py"]
  W057 --> T101["T-101 done: Add make runs and make run -> 50_methodology/51_algorithms/list_runs.py and mk/paper.mk"]
  W057 --> T102["T-102 done: Add make examples, the common commands with examples -> mk/paper.mk"]
  W057 --> T104["T-104 done: Refine the run log from the developer's first look -> shared/surrogate.py and fit_baseline.py"]
  W057 --> T105["T-105 done: Two stop-time columns, the stop criterion on top and a live loss curve -> shared/surrogate.py and fit_baseline.py"]
  W057 --> T106["T-106 done: Print the epoch table at 4 significant figures, elapsed_s to a tenth -> shared/surrogate.py"]
  E006["E-006 todo: Data-guided choices tested -- rho_c input, H1, H2 and H3"]
  E006 --> W038["W-038 issue todo: rho_c input check -- raw against log10"]
  E006 --> W039["W-039 issue todo: H1 -- log against linear charge target"]
  E006 --> W040["W-040 issue todo: H2 -- loss x activation at the M_max turning point"]
  E006 --> W041["W-041 issue todo: H3a -- model pool GPR, XGBoost, MLP and ResNet"]
  E006 --> W042["W-042 issue todo: H3b -- ripple metric of MLP against ResNet"]
  E006 --> W043["W-043 issue todo: H3c -- GPR scaling wall"]
  E007["E-007 todo: Training refinement, ensemble uncertainty and the final surrogate"]
  E007 --> W044["W-044 spike todo: Is L-BFGS fine-tuning worth keeping"]
  E007 --> W045["W-045 issue todo: Ensemble uncertainty over curve-bootstrap resamples"]
  E007 --> W046["W-046 issue todo: Final surrogate saved and loadable"]
  E008["E-008 todo: Speedup and the MCMC application"]
  E008 --> W047["W-047 spike todo: Speedup baseline and reference posterior without the solver"]
  E008 --> W048["W-048 issue todo: Pareto front of error against inference time"]
  E008 --> W049["W-049 issue todo: Mock MCMC recovery with emcee"]
  E008 --> W050["W-050 issue todo: Surrogate posterior validated against the reference"]
  E009["E-009 doing: The paper follows the revised structure, and the release"]
  E009 --> W051["W-051 issue done: Sections 2.1 to 2.4 written from the revised structure"]
  W051 --> T109["T-109 done: Add Section 2.2's folder and renumber the dataset sections -> 30_physical_framework/32_scalarization, 33_black_holes, 34_neutron_stars"]
  W051 --> T110["T-110 done: Write Section 2.1, the action and field equations -> 30_physical_framework/31_action/31_action.tex"]
  W051 --> T111["T-111 done: Write Section 2.2, the scalarization mechanism -> 30_physical_framework/32_scalarization/32_scalarization.tex"]
  W051 --> T112["T-112 done: Write Section 2.3, the black-hole dataset -> 30_physical_framework/33_black_holes/33_black_holes.tex"]
  W051 --> T113["T-113 done: Write Section 2.4, the neutron-star dataset -> 30_physical_framework/34_neutron_stars/34_neutron_stars.tex"]
  E009 --> W052["W-052 issue todo: Abstract, introduction, conclusion and title block"]
  E009 --> W053["W-053 spike todo: What the open-source release publishes"]
  E009 --> W059["W-059 spike blocked on developer: Which refs.bib keys the revised structure's citations name"]
  W059 --> T107["T-107 done: Map the numbered citations to refs.bib keys -> 00_metadata/citations.md"]
  W059 --> T108["T-108 doing: Add the missing references to refs.bib -> 00_metadata/refs.bib"]
  E009 --> W060["W-060 issue done: Title and outline bullets aligned to the revised structure"]
  W060 --> T114["T-114 done: Set the revised title -> 00_metadata/metadata.tex"]
  W060 --> T115["T-115 done: Align the remaining outline bullets to the revised structure -> the todo bullets of 10, 20, 41, 44, 51-53, 61-64, 70, 80"]
  W060 --> T116["T-116 done: Name paper sections by their compiled number in code and make -> mk/, paper.toml, the colocated modules"]
  classDef doing fill:#fff3bf,stroke:#b58900,color:#000;
  classDef blocked fill:#ffe3e3,stroke:#c92a2a,color:#000;
  classDef done fill:#e6ffed,stroke:#2b8a3e,color:#000;
  classDef todo fill:#f8f9fa,stroke:#868e96,color:#000;
  classDef pruned fill:#f1f3f5,stroke:#adb5bd,color:#000;
  class E005,E009,T108 doing
  class W059 blocked
  class W034,T076,T077,T078,T079,T080,T090,T091,W035,T081,T082,T083,T084,T092,W054,T093,T094,T095,W055,T097,T098,W056,T096,W057,T099,T100,T101,T102,T104,T105,T106,W051,T109,T110,T111,T112,T113,T107,W060,T114,T115,T116 done
  class W036,T085,T086,T087,W037,T088,T089,E006,W038,W039,W040,W041,W042,W043,E007,W044,W045,W046,E008,W047,W048,W049,W050,W052,W053 todo
```

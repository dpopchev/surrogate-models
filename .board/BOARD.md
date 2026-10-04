# Board -- surrogate-models
record: local
open path: E-001 > W-004   | blocked: 0 | todo roots: 1 | done: 12 | pruned: 0

## Tree

### E-001 [doing] Overleaf-ready article built from the chapter folders

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

#### W-004 issue [doing] build/paper/article/ is a clean upload folder

- T-008 [done] Move LaTeX residue out of the upload folder -> build/paper/article/ sources only, PDF at build/paper/article.pdf
- T-009 [done] Verify the upload folder in a clean room -> paper-verify compiles a copy of build/paper/article/

### W-005 issue [todo] (standalone) Python layer follows the COLOCATED level

- T-010 [todo] Put the repository root on pytest pythonpath -> pyproject.toml pythonpath = ["."]
- T-011 [todo] Align the python files with the add-python COLOCATED templates -> pyproject.toml, mk/python.mk, conftest.py

## Diagram
```mermaid
flowchart TD
  E001["E-001 doing: Overleaf-ready article built from the chapter folders"]
  E001 --> W001["W-001 issue done: Skeleton article compiles from the chapter folders"]
  W001 --> T001["T-001 done: Write the chapter skeleton and shared LaTeX setup -> 00_metadata/ and one <dir>/<dir>.tex per section"]
  W001 --> T002["T-002 done: Add the paper layer -> mk/paper.mk compile builds build/paper/article/main.pdf"]
  E001 --> W002["W-002 issue done: A script beside its section generates an asset the article uses"]
  W002 --> T003["T-003 done: Add the Python toolchain as a non-package project -> pyproject.toml, mk/python.mk, make check green"]
  W002 --> T004["T-004 done: Write the build stamp script test-first -> 00_metadata/build_stamp.py with test_build_stamp.py"]
  W002 --> T005["T-005 done: Wire generated assets into compile -> build stamp on the title page of main.pdf"]
  E001 --> W003["W-003 issue done: The article zip is a self-contained Overleaf project"]
  W003 --> T006["T-006 done: Add the zip step -> build/paper/article.zip with main.tex at its root"]
  W003 --> T007["T-007 done: Add the clean-room check -> make paper-verify compiles the unpacked zip"]
  E001 --> W004["W-004 issue doing: build/paper/article/ is a clean upload folder"]
  W004 --> T008["T-008 done: Move LaTeX residue out of the upload folder -> build/paper/article/ sources only, PDF at build/paper/article.pdf"]
  W004 --> T009["T-009 done: Verify the upload folder in a clean room -> paper-verify compiles a copy of build/paper/article/"]
  W005["W-005 issue todo, standalone: Python layer follows the COLOCATED level"]
  W005 --> T010["T-010 todo: Put the repository root on pytest pythonpath -> pyproject.toml pythonpath = ['.']"]
  W005 --> T011["T-011 todo: Align the python files with the add-python COLOCATED templates -> pyproject.toml, mk/python.mk, conftest.py"]
  classDef doing fill:#fff3bf,stroke:#b58900,color:#000;
  classDef blocked fill:#ffe3e3,stroke:#c92a2a,color:#000;
  classDef done fill:#e6ffed,stroke:#2b8a3e,color:#000;
  classDef todo fill:#f8f9fa,stroke:#868e96,color:#000;
  classDef pruned fill:#f1f3f5,stroke:#adb5bd,color:#000;
  class E001,W004 doing
  class W001,T001,T002,W002,T003,T004,T005,W003,T006,T007,T008,T009 done
  class W005,T010,T011 todo
```

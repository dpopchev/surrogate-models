"""Render the stored tables and numbers of a section into its assets (W-137).

Run as `uv run python paperlib/tools/render_assets.py <state folder of a section> <assets>`: each
<state>/tables/<name>.json becomes <assets>/<section>/<section>_tab_<name>.tex under the [tables]
layout of paper.toml, and <state>/numbers.json (name -> LaTeX text) <section>_num.tex. A file
whose name holds a dot, <name>.<shape>.json, is a shape of a table, stored for its own page.
"""

import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from paperlib.config import TableLayout, load_config
from paperlib.eda import render_macros
from paperlib.tables import from_json, render

logger = logging.getLogger(__name__)

# --- pure functions ---------------------------------------------------------------------------


def assets_of(
    section: str, tables: Mapping[str, str], numbers: Mapping[str, str] | None, layout: TableLayout
) -> dict[str, str]:
    """The asset file name -> text of a section, from its stored table JSON by name and numbers."""
    found = {
        f"{section}_tab_{name}.tex": render(from_json(text), layout)
        for name, text in sorted(tables.items())
    }
    if numbers is not None:
        found[f"{section}_num.tex"] = render_macros(dict(numbers))
    return found


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Write the table and numbers assets of the section whose state folder is argv[0]."""
    assert len(argv) == 2, f"usage: render_assets.py <state folder> <assets folder>, {argv}"
    state, assets = Path(argv[0]), Path(argv[1])
    assert state.is_dir(), f"state folder not found: {state}"
    # A stored shape is <name>.<shape>.json: another table's file, not an asset of the paper.
    tables = {
        path.stem: path.read_text()
        for path in (state / "tables").glob("*.json")
        if "." not in path.stem
    }
    numbers_file = state / "numbers.json"
    numbers = json.loads(numbers_file.read_text()) if numbers_file.is_file() else None
    found = assets_of(state.name, tables, numbers, load_config().tables.layout)
    out = assets / state.name
    out.mkdir(parents=True, exist_ok=True)
    for name, text in found.items():
        (out / name).write_text(text)
    logger.info("done: %d assets of %s -> %s", len(found), state.name, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

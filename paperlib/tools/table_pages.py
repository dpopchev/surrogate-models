"""Render every stored table, in each layout and shape, as a standalone cropped PDF (W-139).

Nothing is run but the renderer: the inputs are the JSON of local/state/<section>/tables/
(<name>.json, the shapes <name>.<shape>.json, a shape of two tables as <name>.<shape>.1.json
and .2.json, set on one page in order) and the hand-written decisions tables. A page is a
standalone article with the paper's preamble and notation and one table; latexmk compiles it,
pdfcrop trims it to the table.

Run as `uv run python paperlib/tools/table_pages.py <state> <decisions> <latex> <work> <out>`
(mk/paper.mk does); each page lands in <out>/<section>/<section>_tab_<name>_<layout or shape>.pdf.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from paperlib.config import TableLayout
from paperlib.eda import render_macros
from paperlib.tables import from_json, render

logger = logging.getLogger(__name__)

LAYOUTS: tuple[TableLayout, ...] = ("small", "aligned")
LATEXMK = ("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-silent")

# --- pure functions ---------------------------------------------------------------------------


def page_names(section: str, name: str, shapes: Sequence[str]) -> tuple[str, ...]:
    """The page names of a table: one per layout, then one per shape."""
    return tuple(f"{section}_tab_{name}_{kind}" for kind in (*LAYOUTS, *shapes))


def shapes_of(stems: Sequence[str]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Group stored stems by table: name -> shape -> the stems of the shape, in order.

    >>> shapes_of(("t", "t.split.2", "t.split.1", "t.flip"))
    {'t': {'flip': ('t.flip',), 'split': ('t.split.1', 't.split.2')}}
    """
    found: dict[str, dict[str, tuple[str, ...]]] = {}
    for stem in sorted(stems):
        name, *rest = stem.split(".")
        found.setdefault(name, {})
        if rest:
            shape = rest[0]
            found[name][shape] = (*found[name].get(shape, ()), stem)
    return found


def page_tex(numbers: Sequence[str], bodies: Sequence[str]) -> str:
    """A standalone page: the paper's preamble (with the notation), the number macros, then
    the tables in order, on a page tall enough to hold them all."""
    inputs = "".join(f"\\input{{{name}}}\n" for name in numbers)
    return (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[paperwidth=21cm,paperheight=80cm,margin=2cm]{geometry}\n"
        f"\\input{{preamble}}\n{inputs}"
        "\\begin{document}\n\\pagestyle{empty}\n"
        f"{''.join(bodies)}\\end{{document}}\n"
    )


# --- shell ------------------------------------------------------------------------------------


def compile_page(name: str, tex: str, latex: Path, work: Path, out: Path) -> None:
    """Compile one page in work and crop it to its table into out/<name>.pdf."""
    source = work / f"{name}.tex"
    source.write_text(tex)
    env = {**os.environ, "TEXINPUTS": f"{latex.resolve()}:{work.resolve()}:"}
    subprocess.run([*LATEXMK, "-cd", str(source)], check=True, env=env, stdout=subprocess.DEVNULL)
    out.mkdir(parents=True, exist_ok=True)
    pdf = work / f"{name}.pdf"
    subprocess.run(
        ["pdfcrop", "--margins", "10", str(pdf), str(out / f"{name}.pdf")],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main(argv: list[str]) -> None:
    """Write every page of every stored table and of the decisions tables."""
    assert len(argv) == 5, f"usage: table_pages.py <state> <decisions> <latex> <work> <out>, {argv}"
    state, decisions, latex, work, out = (Path(arg) for arg in argv)
    assert state.is_dir(), f"state folder not found: {state}"
    assert (latex / "preamble.tex").is_file(), f"preamble not found in {latex}"
    work.mkdir(parents=True, exist_ok=True)
    numbers = []
    for numbers_file in sorted(state.glob("*/numbers.json")):
        name = f"{numbers_file.parent.name}_num"
        (work / f"{name}.tex").write_text(render_macros(json.loads(numbers_file.read_text())))
        numbers.append(name)
    count = 0
    for tables in sorted(state.glob("*/tables")):
        section = tables.parent.name
        stems = [path.stem for path in tables.glob("*.json")]
        for name, shapes in shapes_of(stems).items():
            names = page_names(section, name, tuple(shapes))
            pages = {
                names[at]: [render(from_json((tables / f"{name}.json").read_text()), layout)]
                for at, layout in enumerate(LAYOUTS)
            }
            for shape, shape_stems in shapes.items():
                bodies = [
                    render(from_json((tables / f"{stem}.json").read_text()), "small")
                    for stem in shape_stems
                ]
                pages[f"{section}_tab_{name}_{shape}"] = bodies
            for page, bodies in pages.items():
                compile_page(page, page_tex(numbers, bodies), latex, work, out / section)
                count += 1
    for hand in sorted(decisions.glob("*_tab_decisions_*.tex")):
        section = hand.name.split("_tab_")[0]
        for layout in LAYOUTS:
            page = f"{hand.stem}_{layout}"
            compile_page(page, page_tex(numbers, [hand.read_text()]), latex, work, out / section)
            count += 1
    logger.info("done: %d table pages -> %s", count, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

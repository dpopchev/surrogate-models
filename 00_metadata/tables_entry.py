"""Render every table of one layout to a page image, to pick the layouts from (W-114).

The entry keeps the article's preamble and generated numbers and sets each table file of the
work folder on a page of its own, in sorted order; latexmk compiles it there, pdfcrop trims each
page to its table (W-125), gs rasterizes the pages (poppler is not installed, gs is) and each
page is named after the table it holds.

Run as `uv run python 00_metadata/tables_entry.py <article.tex> <layout> <work dir> <out dir>`
(mk/paper.mk does), the work folder holding the paper's sources and one layout's assets, flat.
"""

import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# --- pure functions ---------------------------------------------------------------------------

_BEGIN, _END = "\\begin{document}", "\\end{document}"
LATEXMK = ("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-silent")


def tables_entry(article: str, tables: Sequence[str]) -> str:
    """The article's head, then each table input on a page of its own, in sorted order."""
    head = article.split(_BEGIN, 1)[0]
    pages = "".join(f"\\input{{{table}}}\n\\clearpage\n" for table in sorted(tables))
    return f"{head}{_BEGIN}\n\\pagestyle{{empty}}\n{pages}{_END}\n"


def page_names(tables: Sequence[str], layout: str) -> tuple[str, ...]:
    """The image name of each page, in the entry's order: <table stem>_<layout>.png."""
    return tuple(f"{table}_{layout}.png" for table in sorted(tables))


def rasterize_commands(pdf: Path, pages: Path) -> tuple[list[str], list[str]]:
    """pdfcrop trimming each page of pdf to its table with a 10 bp margin, then gs rasterizing
    the cropped pages at 150 dpi into the pages pattern."""
    cropped = pdf.with_name(f"{pdf.stem}-crop.pdf")
    crop = ["pdfcrop", "--margins", "10", str(pdf), str(cropped)]
    raster = [
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=png16m",
        "-r150",
        f"-sOutputFile={pages}",
        str(cropped),
    ]
    return crop, raster


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Write, compile and rasterize the tables entry of one layout; name each page."""
    assert len(argv) == 4, f"usage: tables_entry.py <article.tex> <layout> <work> <out>, {argv}"
    article, layout, work, out = Path(argv[0]), argv[1], Path(argv[2]), Path(argv[3])
    assert article.is_file(), f"article entry not found: {article}"
    tables = tuple(path.stem for path in work.glob("*_tab_*.tex"))
    assert tables, f"no table file (*_tab_*.tex) in {work}"
    entry = work / f"tables_{layout}.tex"
    entry.write_text(tables_entry(article.read_text(), tables))
    subprocess.run([*LATEXMK, "-cd", str(entry)], check=True)
    for stale in work.glob("page-*.png"):
        stale.unlink()
    for command in rasterize_commands(entry.with_suffix(".pdf"), work / "page-%03d.png"):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    pages = sorted(work.glob("page-*.png"))
    if len(pages) != len(tables):
        raise ValueError(f"{entry}: {len(pages)} pages for {len(tables)} tables")
    out.mkdir(parents=True, exist_ok=True)
    for page, name in zip(pages, page_names(tables, layout), strict=True):
        page.replace(out / name)
    logger.info("done: %d tables in the %s layout -> %s", len(tables), layout, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

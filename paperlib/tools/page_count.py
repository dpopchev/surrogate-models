"""Print the pages of the main text and of the appendix from the aux file of a compile.

Run as `uv run python paperlib/tools/page_count.py <main.aux>` (mk/paper.mk does). The appendix
starts at the page of the first label whose number is a letter; the article ends at the highest
page any label names. The References follow the appendix but carry no label, so they are not
counted: the appendix pages here are those up to its last label.
"""

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# \newlabel{name}{{number}{page}...}; the @cref twin of a label is skipped by its name.
_LABEL = re.compile(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}\{(\d+)\}")
_APPENDIX_NUMBER = re.compile(r"[A-Z]")


@dataclass(frozen=True)
class Pages:
    """The page counts of the main text and of the appendix."""

    main: int
    appendix: int


# --- pure functions ---------------------------------------------------------------------------


def read_pages(aux_text: str) -> Pages:
    """Count the pages of the main text and the appendix; raise ValueError without an appendix."""
    labels = [(number, int(page)) for _, number, page in _LABEL.findall(aux_text)]
    starts = [page for number, page in labels if _APPENDIX_NUMBER.fullmatch(number)]
    if not starts:
        raise ValueError("no appendix: no label of the aux text has a letter as its number")
    first = min(starts)
    last = max(page for _, page in labels)
    return Pages(main=first - 1, appendix=last - first + 1)


def verdict(pages: Pages) -> str:
    """Say whether the appendix is no longer than the main text."""
    if pages.appendix <= pages.main:
        return "appendix no longer than the main text"
    return f"[WARN] appendix longer than the main text by {pages.appendix - pages.main} pages"


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Read the aux file named by the one argument and print the pages and the verdict."""
    assert len(argv) == 1, f"usage: page_count.py <main.aux>, got {argv}"
    aux = Path(argv[0])
    if not aux.is_file():
        raise FileNotFoundError(f"no aux file at {aux}: make paper-verify compiles it")
    pages = read_pages(aux.read_text())
    print(f"main text {pages.main} pages, appendix {pages.appendix} pages")
    print(verdict(pages))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

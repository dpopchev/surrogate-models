"""Write the stand-alone entry file of one paper section for review (W-033).

The entry keeps the article's preamble and generated numbers, inputs only the chosen section,
and reads the labels of the full build (main.aux) through xr-hyper, so a reference to another
section or its tables prints as in the whole paper.

Run as `uv run python 00_metadata/section_entry.py <article.tex> <section folder> <out dir>`
(mk/paper.mk does).
"""

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# --- pure functions ---------------------------------------------------------------------------

_BEGIN, _END = "\\begin{document}", "\\end{document}"
_INPUT = re.compile(r"^\\input\{([^}]+)\}$", re.MULTILINE)
_CITE = re.compile(r"\\cite\w*\{")
_DOCUMENTCLASS = re.compile(r"^\\documentclass.*$", re.MULTILINE)


def _parts(article: str) -> tuple[str, str]:
    """The article split into its head (up to the document body) and its body."""
    head, rest = article.split(_BEGIN, 1)
    return head, rest.split(_END, 1)[0]


def sections(article: str) -> tuple[str, ...]:
    """The top-level sections: the inputs of the document body, in order."""
    return tuple(_INPUT.findall(_parts(article)[1]))


def cites(source: str) -> bool:
    """Whether a LaTeX source cites a reference."""
    return _CITE.search(source) is not None


def section_entry(article: str, section: str, cites: bool) -> str:
    """The article entry reduced to one section, with the full build's labels.

    Raises ValueError naming the valid sections when `section` is not one of them.
    """
    valid = sections(article)
    if section not in valid:
        raise ValueError(f"unknown section {section!r}; valid: {', '.join(valid)}")
    head, body = _parts(article)
    # xr-hyper must precede the preamble's hyperref; the external labels come after it.
    with_xr = _DOCUMENTCLASS.sub(lambda m: f"{m[0]}\n\\usepackage{{xr-hyper}}", head, count=1)
    bibliography = [
        line for line in body.splitlines() if line.startswith("\\bibliography") and cites
    ]
    return "".join(
        (
            with_xr,
            "\\externaldocument{main}\n",
            f"{_BEGIN}\n\n\\input{{{section}}}\n\n",
            "".join(f"{line}\n" for line in bibliography),
            f"{_END}\n",
        )
    )


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Write <out dir>/section_<section>.tex for the section whose sources are the given folder."""
    assert len(argv) == 3, f"usage: section_entry.py <article.tex> <section folder> <out>, {argv}"
    article, folder, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    assert article.is_file(), f"article entry not found: {article}"
    assert folder.is_dir(), f"section folder not found: {folder}"
    cited = any(cites(source.read_text()) for source in sorted(folder.rglob("*.tex")))
    entry = section_entry(article.read_text(), folder.name, cites=cited)
    out.mkdir(parents=True, exist_ok=True)
    written = out / f"section_{folder.name}.tex"
    written.write_text(entry)
    logger.info("done: section entry -> %s", written)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

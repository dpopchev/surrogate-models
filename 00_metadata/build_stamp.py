"""Write the build stamp -- commit, work-tree state and day -- as a LaTeX macro for the title page.

Run as `uv run python 00_metadata/build_stamp.py <out.tex>` (mk/paper.mk does).
"""

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import NewType

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

Commit = NewType("Commit", str)

_SHORT_HEX = re.compile(r"[0-9a-f]{7,40}")


class WorkTree(Enum):
    """Whether the build had uncommitted changes."""

    CLEAN = "clean"
    DIRTY = "dirty"


@dataclass(frozen=True)
class BuildStamp:
    """The commit a draft was built from, its work-tree state and the build day."""

    commit: Commit
    tree: WorkTree
    day: date


def make_build_stamp(commit: str, tree: WorkTree, day: date) -> BuildStamp:
    """Build a stamp; raise ValueError when commit is not a lowercase hex id of 7-40 digits."""
    if not _SHORT_HEX.fullmatch(commit):
        raise ValueError(f"commit must be a lowercase hex id of 7-40 digits, got {commit!r}")
    return BuildStamp(Commit(commit), tree, day)


# --- pure functions ---------------------------------------------------------------------------


def stamp_text(stamp: BuildStamp) -> str:
    """Render the stamp as it reads on the title page."""
    dirty = " (dirty)" if stamp.tree is WorkTree.DIRTY else ""
    return f"draft {stamp.commit}{dirty}, {stamp.day.isoformat()}"


def render_tex(stamp: BuildStamp) -> str:
    """Render the stamp as a LaTeX file defining \\buildstamp."""
    return f"\\newcommand{{\\buildstamp}}{{{stamp_text(stamp)}}}\n"


# --- shell ------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout


def main(argv: list[str]) -> None:
    """Read git and the clock, write the stamp file named by the one argument."""
    assert len(argv) == 1, f"usage: build_stamp.py <out.tex>, got {argv}"
    out = Path(argv[0])
    tree = WorkTree.DIRTY if _git("status", "--porcelain").strip() else WorkTree.CLEAN
    stamp = make_build_stamp(_git("rev-parse", "--short", "HEAD").strip(), tree, date.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tex(stamp))
    assert out.read_text() == render_tex(stamp), f"stamp not written to {out}"
    logger.info("done: %s -> %s", stamp_text(stamp), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

"""Tell the drag-and-drop Overleaf project which files to add and which to delete (W-032).

Overleaf's free plan takes files dragged into the project, overwriting same names but never
deleting: the upload folder is compared with the file list of the previous run, the files to
delete on Overleaf are printed, and the list is refreshed.

Run as `uv run python 00_metadata/overleaf_upload.py <upload folder> <last-upload list>`
(mk/paper.mk does).
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class UploadDiff:
    """The file names of the upload folder against those of the previous upload, each sorted."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]


# --- pure functions ---------------------------------------------------------------------------


def diff_upload(current: tuple[str, ...], previous: tuple[str, ...]) -> UploadDiff:
    """Compare the folder's file names with the previous upload's; no previous upload is ()."""
    now, before = set(current), set(previous)
    return UploadDiff(
        tuple(sorted(now - before)), tuple(sorted(before - now)), tuple(sorted(now & before))
    )


def _names(names: tuple[str, ...]) -> str:
    """The names comma-separated, or none."""
    return ", ".join(names) or "none"


def report(diff: UploadDiff) -> str:
    """The added files, the removed files to delete on Overleaf, and the counts."""
    return (
        f"added: {_names(diff.added)}\n"
        f"removed since the last upload (delete these on Overleaf): {_names(diff.removed)}\n"
        f"{len(diff.added)} added, {len(diff.removed)} removed, {len(diff.unchanged)} unchanged\n"
    )


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Print what changed in the upload folder since the last run, then refresh the list."""
    assert len(argv) == 2, f"usage: overleaf_upload.py <upload folder> <list>, got {argv}"
    folder, listing = Path(argv[0]), Path(argv[1])
    assert folder.is_dir(), f"upload folder not found: {folder}"
    entries = tuple(folder.iterdir())
    assert all(p.is_file() for p in entries), f"upload folder must be flat: {folder}"
    current = tuple(sorted(p.name for p in entries))
    previous = tuple(listing.read_text().splitlines()) if listing.is_file() else ()
    print(report(diff_upload(current, previous)), end="")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("".join(f"{name}\n" for name in current))
    logger.info("done: upload list of %s files -> %s", len(current), listing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

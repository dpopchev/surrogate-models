"""Set three figures under each style preset on one contact sheet, to pick a preset (W-115).

Each preset's figures were drawn with PAPER__PLOT__PRESET=<name> and copied to
<root>/<name>/<figure>.png (mk/paper.mk does); the sheet holds a row per preset and a column
per figure, each panel titled by both, written as <root>/presets_variants.png.

Run as `uv run python 00_metadata/presets_sheet.py <root> <preset,...> <figure,...>`.
"""

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from shared.config import load_config
from shared.plots import ARTICLE_TEXT_WIDTH_PT, POINTS_PER_INCH

logger = logging.getLogger(__name__)

# --- pure functions ---------------------------------------------------------------------------


def sheet_images(
    root: Path, presets: Sequence[str], figures: Sequence[str]
) -> dict[tuple[str, str], Path]:
    """Each (preset, figure) image path, preset-major: the sheet's rows in reading order."""
    return {(p, f): root / p / f"{f}.png" for p in presets for f in figures}


def presets_figure(images: Mapping[tuple[str, str], Path]) -> Figure:
    """A row per preset and a column per figure, each panel as wide as the text."""
    presets = list(dict.fromkeys(preset for preset, _ in images))
    figures = list(dict.fromkeys(figure for _, figure in images))
    pictures = {key: plt.imread(path) for key, path in images.items()}
    width = ARTICLE_TEXT_WIDTH_PT / POINTS_PER_INCH
    height = max(width * p.shape[0] / p.shape[1] for p in pictures.values())
    sheet, grid = plt.subplots(
        len(presets),
        len(figures),
        figsize=(width * len(figures), height * 1.1 * len(presets)),
        squeeze=False,
    )
    for row, preset in enumerate(presets):
        for column, figure in enumerate(figures):
            axes = grid[row, column]
            axes.imshow(pictures[preset, figure])
            axes.set_title(f"{preset}: {figure}")
            axes.set_axis_off()
    return sheet


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Write <root>/presets_variants.png from the presets' copied figures."""
    assert len(argv) == 3, f"usage: presets_sheet.py <root> <preset,...> <figure,...>, {argv}"
    root, presets, figures = Path(argv[0]), argv[1].split(","), argv[2].split(",")
    images = sheet_images(root, presets, figures)
    missing = [str(path) for path in images.values() if not path.is_file()]
    assert not missing, f"preset figures not drawn: {missing}"
    sheet = presets_figure(images)
    out = root / "presets_variants.png"
    sheet.savefig(out, dpi=load_config().plot.diagnostics_dpi)
    plt.close(sheet)
    logger.info("done: %d figures under %d presets -> %s", len(figures), len(presets), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

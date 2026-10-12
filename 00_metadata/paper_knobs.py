"""Write the knobs the text cites as LaTeX macros: the baseline network's settings and the
screens' make knobs (W-116).

Reads [methodology.algorithms] of paper.toml; the screens' knobs live in mk/paper.mk only
(Decision on W-066), so make passes their flag strings. Writes one \\knob... macro per setting:
\\knobNet<Field> per network field, \\knob<Screen><Flag> per screen flag, a flag of several
values one macro per value with an ordinal suffix (\\knobFamiliesBudgetsFirst).

Run as `uv run python 00_metadata/paper_knobs.py <out.tex> --representation "<flags>"
--families "<flags>" --tuning "<flags>"` (mk/paper.mk does).
"""

import argparse
import logging
import math
import shlex
import sys
from pathlib import Path

from shared.config import AlgorithmsSection, load_config
from shared.eda import camel, number_tex, render_macros

logger = logging.getLogger(__name__)

# --- pure functions ---------------------------------------------------------------------------

# Network fields that are not a setting of the method: how it logs, and the variant picks.
_NOT_KNOBS = ("log_level", "variants")
_ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")


def knob_tex(value: int | float | str) -> str:
    """A knob as the text writes it in math mode: an exact power of ten from 1000 or below 0.01
    as 10^{k}, any other integer plain, a float as written between 0.01 and 1000 and through
    number_tex beyond, text as it is.

    >>> knob_tex(1024), knob_tex(1000), knob_tex(1e-3), knob_tex(0.2), knob_tex("relu")
    ('1024', '10^{3}', '10^{-3}', '0.2', 'relu')
    """
    if isinstance(value, str):
        return value
    if value > 0 and not 0.01 <= value < 1000:
        exponent = round(math.log10(value))
        if math.isclose(value, 10.0**exponent):
            return f"10^{{{exponent}}}"
    if isinstance(value, int):
        return str(value)
    return f"{value:g}" if 0.01 <= abs(value) < 1000 else number_tex(value)


def _screen_flags(flags: str) -> dict[str, list[str]]:
    """A make knob string as its flags, each with its values: "--k 8 --keeps 3 2"."""
    found: dict[str, list[str]] = {}
    current = ""
    for token in shlex.split(flags):
        if token.startswith("--"):
            current = token[2:]
            found[current] = []
        else:
            assert current, f"a value before any flag in {flags!r}"
            found[current].append(token)
    return found


def _value(text: str) -> int | float:
    """A flag value as the number it names."""
    return int(text) if text.lstrip("-").isdigit() else float(text)


def knob_numbers(net: AlgorithmsSection, screens: dict[str, str]) -> dict[str, str]:
    """Every knob, keyed by its \\knob macro name: the network's fields, then each screen's
    flags (screens maps a macro tag, Repr, to its make knob string)."""
    found = {
        f"knobNet{camel(field)}": knob_tex(getattr(net, field))
        for field in type(net).model_fields
        if field not in _NOT_KNOBS
    }
    for tag, flags in screens.items():
        for flag, values in _screen_flags(flags).items():
            name = f"knob{tag}{camel(flag.replace('-', '_'))}"
            assert 0 < len(values) <= len(_ORDINALS), f"--{flag} takes 1 to 6 values: {values}"
            if len(values) == 1:
                found[name] = knob_tex(_value(values[0]))
            else:
                found |= {
                    f"{name}{ordinal}": knob_tex(_value(v))
                    for ordinal, v in zip(_ORDINALS, values, strict=False)
                }
    return found


# --- shell ------------------------------------------------------------------------------------

# The make option of each screen and the macro tag of its knobs.
SCREENS = {"representation": "Repr", "families": "Families", "tuning": "Tuning"}


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the output file and each screen's make knob string."""
    parser = argparse.ArgumentParser(prog="paper_knobs.py")
    parser.add_argument("out", type=Path)
    for screen in SCREENS:
        parser.add_argument(f"--{screen}", required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    """Write the knobs macro file from paper.toml and the screens' make knobs."""
    args = _arguments(argv)
    screens = {tag: getattr(args, screen) for screen, tag in SCREENS.items()}
    found = knob_numbers(load_config().methodology.algorithms, screens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_macros(found))
    logger.info("done: %d knobs -> %s", len(found), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

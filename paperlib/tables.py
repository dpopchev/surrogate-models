"""A table of the paper as typed data, and its renderer per layout (W-135).

The renderer formats from the cells; it never parses a LaTeX string back.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from paperlib.config import TableLayout


@dataclass(frozen=True)
class Text:
    """A cell of LaTeX text, set as given."""

    tex: str


@dataclass(frozen=True)
class Plain:
    """A plain number, its digits as printed."""

    mantissa: str
    bold: bool = False


@dataclass(frozen=True)
class Power:
    """A number times a power of ten."""

    mantissa: str
    exponent: int
    bold: bool = False


@dataclass(frozen=True)
class Spread:
    """A value with its spread."""

    mantissa: str
    spread: str
    bold: bool = False


@dataclass(frozen=True)
class Span:
    """A text cell spanning several columns, untouched by the aligned layout."""

    width: int
    align: Literal["l", "c", "r"]
    tex: str


Number = Plain | Power | Spread
Cell = Text | Number | Span


@dataclass(frozen=True)
class Column:
    """A column: its alignment and the tabular separators before and after it."""

    align: Literal["l", "c", "r"]
    before: str = ""
    after: str = ""


@dataclass(frozen=True)
class Table:
    """A labelled, captioned table: columns, header rows and body rows of typed cells."""

    label: str
    caption: str
    columns: tuple[Column, ...]
    header: tuple[tuple[Cell, ...], ...]
    rows: tuple[tuple[Cell, ...], ...]


def _small_number(number: Number) -> str:
    match number:
        case Plain(mantissa, _):
            inner = mantissa
        case Power(mantissa, exponent, _):
            inner = f"{mantissa}\\times 10^{{{exponent}}}"
        case Spread(mantissa, spread, _):
            inner = f"{mantissa}_{{\\pm {spread}}}"
        case _:
            assert_never(number)
    if number.bold:
        return f"$\\mathbf{{{inner}}}$"
    return inner if isinstance(number, Plain) else f"${inner}$"


def _small(cell: Cell) -> str:
    match cell:
        case Text(tex):
            return tex
        case Span(width, align, tex):
            return f"\\multicolumn{{{width}}}{{{align}}}{{{tex}}}"
        case Plain() | Power() | Spread():
            return _small_number(cell)
        case _:
            assert_never(cell)


def _si_number(number: Number) -> str:
    match number:
        case Plain(mantissa, _):
            inner = mantissa
        case Power(mantissa, exponent, _):
            inner = f"{mantissa}e{exponent}"
        case Spread(mantissa, spread, _):
            inner = f"{mantissa} +- {spread}"
        case _:
            assert_never(number)
    return f"\\bfseries {inner}" if number.bold else inner


def _siunitx(cell: Cell) -> str:
    """A cell of an S column: a number as siunitx reads it, text braced, a blank left empty."""
    match cell:
        case Text(tex):
            return f"{{{tex}}}" if tex else ""
        case Span():
            return _small(cell)
        case Plain() | Power() | Spread():
            return _si_number(cell)
        case _:
            assert_never(cell)


def _starts(row: tuple[Cell, ...]) -> list[tuple[int, Cell]]:
    """Each cell with the column it starts at, a span moving on by its width."""
    found, at = [], 0
    for cell in row:
        found.append((at, cell))
        at += cell.width if isinstance(cell, Span) else 1
    return found


def _table_format(numbers: list[Number]) -> str:
    """The siunitx table-format reserving the widest sign, integer, decimal, spread and
    exponent parts of the numbers of an S column; a column without a number gets none."""
    if not numbers:
        return ""
    sign = "-" if any(n.mantissa.startswith("-") for n in numbers) else ""
    integer = max(len(n.mantissa.lstrip("-").split(".")[0]) for n in numbers)
    decimals = max(len(n.mantissa.split(".")[1]) if "." in n.mantissa else 0 for n in numbers)
    shape = f"{sign}{integer}" + (f".{decimals}" if decimals else "")
    spreads = [round(float(n.spread) * 10**decimals) for n in numbers if isinstance(n, Spread)]
    if spreads:
        shape += f"({max(len(str(s)) for s in spreads)})"
    exponents = [n.exponent for n in numbers if isinstance(n, Power)]
    if exponents:
        exponent_sign = "-" if any(e < 0 for e in exponents) else ""
        shape += f"e{exponent_sign}{max(len(str(abs(e))) for e in exponents)}"
    return shape


def _numbers_of(table: Table, at: int) -> list[Number]:
    """The numbers in column `at` over the body rows."""
    found: list[Number] = []
    for row in table.rows:
        for start, cell in _starts(row):
            if start == at and isinstance(cell, Plain | Power | Spread):
                found.append(cell)
    return found


def _spec(table: Table, layout: TableLayout) -> str:
    parts = []
    for at, column in enumerate(table.columns):
        match layout:
            case "small":
                letter = column.align
            case "aligned":
                shape = _table_format(_numbers_of(table, at)) if column.align == "r" else ""
                letter = "S" if column.align == "r" else column.align
                if shape:
                    letter = f"S[table-format={shape}]"
            case _:
                assert_never(layout)
        parts.append(f"{column.before}{letter}{column.after}")
    return "".join(parts)


def _small_line(row: tuple[Cell, ...]) -> str:
    """The cells joined by &, an empty cell leaving a bare & as the strings had it."""
    head, *rest = (_small(cell) for cell in row)
    return head + "".join(f" & {cell}" if cell else " &" for cell in rest)


def _aligned_line(table: Table, row: tuple[Cell, ...], body: bool) -> str:
    cells = []
    for at, cell in _starts(row):
        if table.columns[at].align == "r":
            cells.append(_siunitx(cell) if body or isinstance(cell, Text | Span) else _small(cell))
        else:
            cells.append(_small(cell))
    return " & ".join(cells)


def render(table: Table, layout: TableLayout) -> str:
    """One booktabs table environment, labelled tab:<label>, in the small font; aligned, each
    right column becomes an siunitx S column aligning its numbers on the decimal marker."""
    match layout:
        case "small":
            header = [_small_line(row) for row in table.header]
            rows = [_small_line(row) for row in table.rows]
        case "aligned":
            header = [_aligned_line(table, row, body=False) for row in table.header]
            rows = [_aligned_line(table, row, body=True) for row in table.rows]
        case _:
            assert_never(layout)
    head = " \\\\\n".join(header)
    body = "".join(f"{row} \\\\\n" for row in rows)
    return (
        "\\begin{table}[!htb]\n\\centering\n\\small\n"
        f"\\caption{{{table.caption}}}\n\\label{{tab:{table.label}}}\n"
        f"\\begin{{tabular}}{{{_spec(table, layout)}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

"""A table of the paper as typed data, and its renderer per layout (W-135).

The renderer formats from the cells; it never parses a LaTeX string back.
"""

import math
from dataclasses import dataclass
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, Field

from paperlib.config import TableLayout


@dataclass(frozen=True)
class Text:
    """A cell of LaTeX text, set as given."""

    tex: str


@dataclass(frozen=True)
class Plain:
    """A plain number, its digits as printed, in math mode when `math`."""

    mantissa: str
    bold: bool = False
    math: bool = False


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
    """A labelled, captioned table: columns, header rows and body rows of typed cells.

    `small_spec`, when set, is the tabular spec the small layout writes as given, for a table
    whose spec repeats a column group (`*{4}{r}`); the aligned layout reads the columns.
    """

    label: str
    caption: str
    columns: tuple[Column, ...]
    header: tuple[tuple[Cell, ...], ...]
    rows: tuple[tuple[Cell, ...], ...]
    small_spec: str = ""


def aligns(letters: str) -> tuple[Column, ...]:
    """The columns of a spec of plain alignment letters, e.g. "lrr"."""
    return tuple(Column(letter) for letter in letters)  # type: ignore[arg-type]


def fixed(value: float, decimals: int) -> Cell:
    """A number to a fixed count of decimals, as text when it is not finite."""
    text = f"{value:.{decimals}f}"
    return Plain(text) if math.isfinite(value) else Text(text)


def power(value: float, bold: bool = False) -> Power:
    """A number as a mantissa of three significant digits times a power of ten."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return Power(mantissa, int(exponent), bold)


def number(value: float) -> Plain | Power:
    """A number in math mode: 0 as 0, three significant digits between 0.01 and 1000, else a
    power of ten."""
    if value == 0:
        return Plain("0", math=True)
    if not 0.01 <= abs(value) < 1000:
        return power(value)
    decimals = max(2 - math.floor(math.log10(abs(value))), 0)
    return Plain(f"{value:.{decimals}f}", math=True)


class _TextJson(BaseModel):
    kind: Literal["text"]
    tex: str


class _PlainJson(BaseModel):
    kind: Literal["plain"]
    mantissa: str
    bold: bool
    math: bool


class _PowerJson(BaseModel):
    kind: Literal["power"]
    mantissa: str
    exponent: int
    bold: bool


class _SpreadJson(BaseModel):
    kind: Literal["spread"]
    mantissa: str
    spread: str
    bold: bool


class _SpanJson(BaseModel):
    kind: Literal["span"]
    width: int
    align: Literal["l", "c", "r"]
    tex: str


_CellJson = Annotated[
    _TextJson | _PlainJson | _PowerJson | _SpreadJson | _SpanJson, Field(discriminator="kind")
]


class _ColumnJson(BaseModel):
    align: Literal["l", "c", "r"]
    before: str
    after: str


class _TableJson(BaseModel):
    label: str
    caption: str
    columns: list[_ColumnJson]
    header: list[list[_CellJson]]
    rows: list[list[_CellJson]]
    small_spec: str


def _cell_json(cell: Cell) -> _CellJson:
    match cell:
        case Text(tex):
            return _TextJson(kind="text", tex=tex)
        case Plain(mantissa, bold, math):
            return _PlainJson(kind="plain", mantissa=mantissa, bold=bold, math=math)
        case Power(mantissa, exponent, bold):
            return _PowerJson(kind="power", mantissa=mantissa, exponent=exponent, bold=bold)
        case Spread(mantissa, spread, bold):
            return _SpreadJson(kind="spread", mantissa=mantissa, spread=spread, bold=bold)
        case Span(width, align, tex):
            return _SpanJson(kind="span", width=width, align=align, tex=tex)
        case _:
            assert_never(cell)


def _cell_of(item: _CellJson) -> Cell:
    match item:
        case _TextJson():
            return Text(item.tex)
        case _PlainJson():
            return Plain(item.mantissa, item.bold, item.math)
        case _PowerJson():
            return Power(item.mantissa, item.exponent, item.bold)
        case _SpreadJson():
            return Spread(item.mantissa, item.spread, item.bold)
        case _SpanJson():
            return Span(item.width, item.align, item.tex)
        case _:
            assert_never(item)


def to_json(table: Table) -> str:
    """The table as JSON text: primitives only, each cell a tagged object."""
    return _TableJson(
        label=table.label,
        caption=table.caption,
        columns=[_ColumnJson(align=c.align, before=c.before, after=c.after) for c in table.columns],
        header=[[_cell_json(c) for c in row] for row in table.header],
        rows=[[_cell_json(c) for c in row] for row in table.rows],
        small_spec=table.small_spec,
    ).model_dump_json()


def from_json(text: str) -> Table:
    """The table of JSON text; raises pydantic.ValidationError when the text is not a table."""
    raw = _TableJson.model_validate_json(text)
    return Table(
        label=raw.label,
        caption=raw.caption,
        columns=tuple(Column(c.align, c.before, c.after) for c in raw.columns),
        header=tuple(tuple(_cell_of(c) for c in row) for row in raw.header),
        rows=tuple(tuple(_cell_of(c) for c in row) for row in raw.rows),
        small_spec=raw.small_spec,
    )


def _small_number(number: Number) -> str:
    match number:
        case Plain(mantissa, _, _):
            inner = mantissa
        case Power(mantissa, exponent, _):
            inner = f"{mantissa}\\times 10^{{{exponent}}}"
        case Spread(mantissa, spread, _):
            inner = f"{mantissa}_{{\\pm {spread}}}"
        case _:
            assert_never(number)
    if number.bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$" if not isinstance(number, Plain) or number.math else inner


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
    if layout == "small" and table.small_spec:
        return table.small_spec
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

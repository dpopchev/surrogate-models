"""Facts about the Table object and its renderer, on a tiny synthetic table."""

import pytest

from paperlib.config import TableLayout
from paperlib.eda import booktabs, number_tex
from paperlib.tables import (
    Column,
    Plain,
    Power,
    Span,
    Spread,
    Table,
    Text,
    aligns,
    fixed,
    number,
    power,
    render,
)

SPEC = "@{}ll@{\\hspace{8pt}}rr@{}"
HEADER = "A & B & \\multicolumn{2}{c}{C} \\\\\n & & Test & Folds"
ROWS = [
    "x & y & $5.89\\times 10^{14}$ & $1.96_{\\pm 0.02}$",
    "z & w & 0.57 & $\\mathbf{2.36_{\\pm 0.04}}$",
    "u & v & -0.03 & 0.4",
]

SAMPLE = Table(
    label="x",
    caption="A table.",
    columns=(
        Column("l", before="@{}"),
        Column("l", after="@{\\hspace{8pt}}"),
        Column("r"),
        Column("r", after="@{}"),
    ),
    header=(
        (Text("A"), Text("B"), Span(2, "c", "C")),
        (Text(""), Text(""), Text("Test"), Text("Folds")),
    ),
    rows=(
        (Text("x"), Text("y"), Power("5.89", 14), Spread("1.96", "0.02")),
        (Text("z"), Text("w"), Plain("0.57"), Spread("2.36", "0.04", bold=True)),
        (Text("u"), Text("v"), Plain("-0.03"), Plain("0.4")),
    ),
)


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_table_renders_as_booktabs_did(layout: TableLayout) -> None:
    assert render(SAMPLE, layout) == booktabs("x", "A table.", SPEC, HEADER, ROWS, layout)


VALUES = [0.0, 0.5, 12.3, 345.6, 0.001, 5.89e14, -0.03, -2000.0, 3.82e-8]


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_number_cells_render_as_number_tex_did(layout: TableLayout) -> None:
    body = tuple((Text("v"), number(v)) for v in VALUES)
    table = Table("n", "N.", aligns("lr"), ((Text("h"), Text("x")),), body)
    rows = [f"v & ${number_tex(v)}$" for v in VALUES]
    assert render(table, layout) == booktabs("n", "N.", "lr", "h & x", rows, layout)


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_fixed_and_power_cells_render_as_f_strings_did(layout: TableLayout) -> None:
    row = (Text("a"), fixed(0.456, 2), power(8.2e-8), fixed(float("nan"), 1))
    header = tuple(Text(h) for h in "wxyz")
    table = Table("f", "F.", aligns("lrrr"), (header,), (row,))
    rows = ["a & 0.46 & $8.20\\times 10^{-8}$ & nan"]
    assert render(table, layout) == booktabs("f", "F.", "lrrr", "w & x & y & z", rows, layout)


@pytest.mark.functional
def test_aligns_gives_a_column_per_letter() -> None:
    assert aligns("lrc") == (Column("l"), Column("r"), Column("c"))

"""Facts about the Table object and its renderer, on a tiny synthetic table."""

import pytest

from paperlib.config import TableLayout
from paperlib.eda import booktabs
from paperlib.tables import Column, Plain, Power, Span, Spread, Table, Text, render

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

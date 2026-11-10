"""Facts about the Table object and its renderer, on a tiny synthetic table."""

import pytest

from paperlib.config import TableLayout
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


HEAD = "\\begin{table}[!htb]\n\\centering\n\\small\n"
FOOT = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

SAMPLE_TEX = {
    "small": (
        f"{HEAD}\\caption{{A table.}}\n\\label{{tab:x}}\n"
        "\\begin{tabular}{@{}ll@{\\hspace{8pt}}rr@{}}\n\\toprule\n"
        "A & B & \\multicolumn{2}{c}{C} \\\\\n & & Test & Folds \\\\\n\\midrule\n"
        "x & y & $5.89\\times 10^{14}$ & $1.96_{\\pm 0.02}$ \\\\\n"
        "z & w & 0.57 & $\\mathbf{2.36_{\\pm 0.04}}$ \\\\\n"
        f"u & v & -0.03 & 0.4 \\\\\n{FOOT}"
    ),
    "aligned": (
        f"{HEAD}\\caption{{A table.}}\n\\label{{tab:x}}\n"
        "\\begin{tabular}{@{}ll@{\\hspace{8pt}}S[table-format=-1.2e2]S[table-format=1.2(1)]@{}}\n"
        "\\toprule\nA & B & \\multicolumn{2}{c}{C} \\\\\n &  & {Test} & {Folds} \\\\\n\\midrule\n"
        "x & y & 5.89e14 & 1.96 +- 0.02 \\\\\n"
        "z & w & 0.57 & \\bfseries 2.36 +- 0.04 \\\\\n"
        f"u & v & -0.03 & 0.4 \\\\\n{FOOT}"
    ),
}


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_table_renders_as_the_paper_writes_it(layout: TableLayout) -> None:
    assert render(SAMPLE, layout) == SAMPLE_TEX[layout]


VALUES = [0.0, 0.5, 12.3, 345.6, 0.001, 5.89e14, -0.03, -2000.0, 3.82e-8]

NUMBERS_TEX = {
    "small": (
        f"{HEAD}\\caption{{N.}}\n\\label{{tab:n}}\n\\begin{{tabular}}{{lr}}\n\\toprule\n"
        "h & x \\\\\n\\midrule\n"
        "v & $0$ \\\\\nv & $0.500$ \\\\\nv & $12.3$ \\\\\nv & $346$ \\\\\n"
        "v & $1.00\\times 10^{-3}$ \\\\\nv & $5.89\\times 10^{14}$ \\\\\n"
        "v & $-0.0300$ \\\\\nv & $-2.00\\times 10^{3}$ \\\\\n"
        f"v & $3.82\\times 10^{{-8}}$ \\\\\n{FOOT}"
    ),
    "aligned": (
        f"{HEAD}\\caption{{N.}}\n\\label{{tab:n}}\n"
        "\\begin{tabular}{lS[table-format=-3.4e-2]}\n\\toprule\n"
        "h & {x} \\\\\n\\midrule\n"
        "v & 0 \\\\\nv & 0.500 \\\\\nv & 12.3 \\\\\nv & 346 \\\\\n"
        "v & 1.00e-3 \\\\\nv & 5.89e14 \\\\\nv & -0.0300 \\\\\nv & -2.00e3 \\\\\n"
        f"v & 3.82e-8 \\\\\n{FOOT}"
    ),
}


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_number_cells_render_as_the_paper_writes_them(layout: TableLayout) -> None:
    body = tuple((Text("v"), number(v)) for v in VALUES)
    table = Table("n", "N.", aligns("lr"), ((Text("h"), Text("x")),), body)
    assert render(table, layout) == NUMBERS_TEX[layout]


FIXED_TEX = {
    "small": (
        f"{HEAD}\\caption{{F.}}\n\\label{{tab:f}}\n\\begin{{tabular}}{{lrrr}}\n\\toprule\n"
        "w & x & y & z \\\\\n\\midrule\n"
        f"a & 0.46 & $8.20\\times 10^{{-8}}$ & nan \\\\\n{FOOT}"
    ),
    "aligned": (
        f"{HEAD}\\caption{{F.}}\n\\label{{tab:f}}\n"
        "\\begin{tabular}{lS[table-format=1.2]S[table-format=1.2e-1]S}\n\\toprule\n"
        "w & {x} & {y} & {z} \\\\\n\\midrule\n"
        f"a & 0.46 & 8.20e-8 & {{nan}} \\\\\n{FOOT}"
    ),
}


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_fixed_and_power_cells_render_as_the_paper_writes_them(layout: TableLayout) -> None:
    row = (Text("a"), fixed(0.456, 2), power(8.2e-8), fixed(float("nan"), 1))
    header = tuple(Text(h) for h in "wxyz")
    table = Table("f", "F.", aligns("lrrr"), (header,), (row,))
    assert render(table, layout) == FIXED_TEX[layout]


@pytest.mark.view
@pytest.mark.parametrize("layout", ["small", "aligned"])
def test_a_small_spec_with_a_repeat_is_written_as_given_in_the_small_layout(
    layout: TableLayout,
) -> None:
    columns = (Column("l", before="@{}"), *aligns("rrr")[:-1], Column("r", after="@{}"))
    row = (Text("a"), Plain("1"), Plain("2"), Plain("3"))
    header = tuple(Text(h) for h in "wxyz")
    table = Table("s", "S.", columns, (header,), (row,), small_spec="@{}l*{3}{r}@{}")
    column = "S[table-format=1]"
    spec = "@{}l*{3}{r}@{}" if layout == "small" else f"@{{}}l{column * 3}@{{}}"
    assert f"\\begin{{tabular}}{{{spec}}}" in render(table, layout)


@pytest.mark.functional
def test_aligns_gives_a_column_per_letter() -> None:
    assert aligns("lrc") == (Column("l"), Column("r"), Column("c"))

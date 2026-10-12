"""Facts about the stand-alone entry that sets every table of one layout on its own page."""

import pytest
from tables_entry import tables_entry

ARTICLE = (
    "\\documentclass{article}\n"
    "\\input{preamble}\n"
    "\\input{x_num}\n"
    "\\begin{document}\n"
    "\\maketitle\n"
    "\\input{10_a}\n"
    "\\end{document}\n"
)


@pytest.mark.view
def test_tables_entry_inputs_every_table_file() -> None:
    entry = tables_entry(ARTICLE, ("b_tab_y", "a_tab_x"))
    inputs = [line for line in entry.splitlines() if "_tab_" in line]
    assert inputs == ["\\input{a_tab_x}", "\\input{b_tab_y}"]

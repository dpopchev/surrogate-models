"""Facts about the standalone page of every table layout and shape (W-139)."""

import pytest

from paperlib.tools.table_pages import page_names


@pytest.mark.view
def test_page_names_of_a_table_with_two_layouts_and_one_shape() -> None:
    assert page_names("98_families", "families", ("rows_column",)) == (
        "98_families_tab_families_small",
        "98_families_tab_families_aligned",
        "98_families_tab_families_rows_column",
    )

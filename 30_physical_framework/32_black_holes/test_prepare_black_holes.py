"""Facts about turning a '#'-headed whitespace table into the prepared BH table."""

import pandas as pd
import pytest
from prepare_black_holes import (
    FixedColumnVariesError,
    NonPositiveTargetError,
    make_table_spec,
    parse_table,
    prepare,
)

TEXT = "#  a   b   k   t\n1.0 2.0 7.0 0.5 \n3.0 5.0 7.0 0.25 \n"
SPEC = make_table_spec(rename={"a": "x", "t": "y"}, fixed=("k",), positive="y")


def test_parse_reads_header_names() -> None:
    assert list(parse_table(TEXT).columns) == ["a", "b", "k", "t"]


def test_parse_reads_values_despite_trailing_whitespace() -> None:
    assert parse_table(TEXT)["b"].tolist() == [2.0, 5.0]


def test_text_without_a_header_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="header"):
        parse_table("1.0 2.0\n")


def test_spec_whose_positive_column_is_not_kept_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_table_spec(rename={"a": "x"}, fixed=(), positive="y")


def test_prepare_keeps_and_renames_the_mapped_columns() -> None:
    assert prepare(TEXT, SPEC).equals(pd.DataFrame({"x": [1.0, 3.0], "y": [0.5, 0.25]}))


def test_fixed_column_that_varies_is_rejected() -> None:
    varying = "# a k t\n1.0 7.0 0.5\n2.0 8.0 0.5\n"
    with pytest.raises(FixedColumnVariesError, match="k"):
        prepare(varying, SPEC)


def test_non_positive_target_is_rejected() -> None:
    zero = "# a k t\n1.0 7.0 0.5\n2.0 7.0 0.0\n"
    with pytest.raises(NonPositiveTargetError, match="1 row"):
        prepare(zero, SPEC)

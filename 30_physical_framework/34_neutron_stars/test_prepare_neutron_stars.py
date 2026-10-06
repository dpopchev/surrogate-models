"""Facts about turning '#'-opened curve blocks into the prepared NS table."""

from pathlib import Path

import pandas as pd
import pytest
from prepare_neutron_stars import (
    CurveParamMismatchError,
    FixedColumnVariesError,
    HeaderLayoutError,
    NonPositiveTargetError,
    NonRisingOrderError,
    RawCounts,
    main,
    make_curve_spec,
    parse_blocks,
    parse_params,
    prepare,
    raw_counts,
)

# Two runs of curve parameter g; the second did not converge and has no header or rows.
TEXT = (
    "# s = 7.0, g = 1.0\n   o   y   g   k\n1.0 2.0 1.0 9.0 \n2.0 5.0 1.0 9.0 \n# s = 7.0, g = 2.0\n"
)


def test_param_line_reads_name_value_pairs() -> None:
    assert parse_params("# s = 7.0, g = 5e-05") == {"s": 7.0, "g": 5e-05}


def test_every_param_line_opens_a_block_even_without_rows() -> None:
    assert len(parse_blocks(TEXT)) == 2


def test_empty_block_has_no_rows() -> None:
    assert parse_blocks(TEXT)[1].rows == ()


def test_raw_counts_give_runs_empty_runs_and_rows() -> None:
    assert raw_counts(parse_blocks(TEXT)) == RawCounts(runs=2, empty_runs=1, rows=2)


def test_block_rows_are_read_under_its_header() -> None:
    assert parse_blocks(TEXT)[0].rows == ((1.0, 2.0, 1.0, 9.0), (2.0, 5.0, 1.0, 9.0))


def test_block_names_are_its_header_columns() -> None:
    assert parse_blocks(TEXT)[0].names == ("o", "y", "g", "k")


def test_text_not_opened_by_a_param_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="'#'"):
        parse_blocks("   o   y\n1.0 2.0\n")


# o orders a curve, y is the peaked and positive target, g the curve parameter, k fixed.
SPEC = make_curve_spec(
    rename={"g": "g", "o": "x", "y": "y"},
    fixed_params=("s",),
    fixed_columns=("k",),
    curve_params={"g": "g"},
    order="o",
    peak="y",
    positive="y",
)


def curve(ys: tuple[float, ...], g: float = 1.0, s: float = 7.0) -> str:
    """One block of curve g whose y values follow ys along o = 1, 2, ..."""
    rows = "".join(f"{o:.1f} {y} {g} 9.0\n" for o, y in enumerate(ys, start=1))
    return f"# s = {s}, g = {g}\n   o   y   g   k\n{rows}"


def test_curve_is_cut_after_its_peak_keeping_the_peak_row() -> None:
    assert prepare(curve((2.0, 5.0, 4.0)), SPEC)["y"].tolist() == [2.0, 5.0]


def test_curve_without_a_peak_is_kept_whole() -> None:
    assert prepare(curve((3.0, 6.0)), SPEC)["y"].tolist() == [3.0, 6.0]


def test_curve_is_cut_at_its_first_peak_on_a_tie() -> None:
    assert prepare(curve((2.0, 5.0, 5.0, 4.0)), SPEC)["y"].tolist() == [2.0, 5.0]


def test_empty_block_is_dropped() -> None:
    assert len(prepare(TEXT, SPEC)) == 2


def test_prepare_keeps_and_renames_the_mapped_columns() -> None:
    assert list(prepare(TEXT, SPEC).columns) == ["g", "x", "y"]


def test_text_without_any_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="no rows"):
        prepare("# s = 7.0, g = 2.0\n", SPEC)


def test_a_second_header_layout_is_rejected() -> None:
    reordered = "# s = 7.0, g = 2.0\n   o   g   y   k\n1.0 2.0 3.0 9.0\n"
    with pytest.raises(HeaderLayoutError, match="2 header layouts"):
        prepare(curve((2.0,)) + reordered, SPEC)


def test_fixed_param_that_varies_across_blocks_is_rejected() -> None:
    with pytest.raises(FixedColumnVariesError, match="'s'"):
        prepare(curve((2.0,), g=1.0) + curve((3.0,), g=2.0, s=8.0), SPEC)


def test_fixed_column_that_varies_is_rejected() -> None:
    varying = "# s = 7.0, g = 1.0\n   o   y   g   k\n1.0 2.0 1.0 9.0\n2.0 3.0 1.0 8.0\n"
    with pytest.raises(FixedColumnVariesError, match="'k'"):
        prepare(varying, SPEC)


def test_header_param_unlike_its_column_is_rejected() -> None:
    mismatch = "# s = 7.0, g = 1.0\n   o   y   g   k\n1.0 2.0 2.0 9.0\n"
    with pytest.raises(CurveParamMismatchError, match="'g'"):
        prepare(mismatch, SPEC)


def test_order_column_that_does_not_rise_strictly_is_rejected() -> None:
    falling = "# s = 7.0, g = 1.0\n   o   y   g   k\n2.0 2.0 1.0 9.0\n2.0 3.0 1.0 9.0\n"
    with pytest.raises(NonRisingOrderError, match="'o'"):
        prepare(falling, SPEC)


def test_non_positive_target_is_rejected() -> None:
    with pytest.raises(NonPositiveTargetError, match="1 row"):
        prepare(curve((-1.0, 2.0)), SPEC)


def test_main_writes_the_prepared_table_as_parquet(tmp_path: Path) -> None:
    source, out = tmp_path / "in.dat", tmp_path / "out.parquet"
    source.write_text(curve((2.0, 5.0, 4.0)))
    main([str(source), str(out)], spec=SPEC)
    assert pd.read_parquet(out)["y"].tolist() == [2.0, 5.0]


def test_main_keeps_the_raw_counts_with_the_table(tmp_path: Path) -> None:
    source, out = tmp_path / "in.dat", tmp_path / "out.parquet"
    source.write_text(curve((2.0, 5.0, 4.0)))
    main([str(source), str(out)], spec=SPEC)
    assert pd.read_parquet(out).attrs == {"runs": 1, "empty_runs": 0, "rows": 3}


def test_spec_whose_positive_column_is_not_kept_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_curve_spec({"o": "x"}, (), (), {}, order="o", peak="o", positive="y")

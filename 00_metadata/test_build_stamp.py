"""Facts about the build stamp shown on the title page of every draft."""

from datetime import date

import pytest
from build_stamp import WorkTree, make_build_stamp, render_tex, stamp_text

DAY = date(2026, 1, 2)


def test_clean_tree_stamp_names_commit_and_day() -> None:
    stamp = make_build_stamp("abc1234", WorkTree.CLEAN, DAY)
    assert stamp_text(stamp) == "draft abc1234, 2026-01-02"


def test_dirty_tree_stamp_is_marked_dirty() -> None:
    stamp = make_build_stamp("abc1234", WorkTree.DIRTY, DAY)
    assert stamp_text(stamp) == "draft abc1234 (dirty), 2026-01-02"


def test_tex_defines_the_buildstamp_macro() -> None:
    stamp = make_build_stamp("abc1234", WorkTree.CLEAN, DAY)
    assert render_tex(stamp) == "\\newcommand{\\buildstamp}{draft abc1234, 2026-01-02}\n"


@pytest.mark.parametrize("commit", ["", "abc", "xyz1234", "ABC1234"])
def test_commit_that_is_not_a_short_hex_id_is_rejected(commit: str) -> None:
    with pytest.raises(ValueError, match="commit"):
        make_build_stamp(commit, WorkTree.CLEAN, DAY)

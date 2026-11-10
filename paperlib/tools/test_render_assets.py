"""Facts about rendering the stored tables and numbers of a section into assets (W-137)."""

from pathlib import Path

import pytest

from paperlib.tables import Plain, Table, Text, aligns, to_json
from paperlib.tools.render_assets import main

TABLE = Table(
    label="demo",
    caption="A demo.",
    columns=aligns("lr"),
    header=((Text("a"), Text("b")),),
    rows=((Text("x"), Plain("1.5")),),
)


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "local" / "state" / "41_demo"
    (state / "tables").mkdir(parents=True)
    (state / "tables" / "demo.json").write_text(to_json(TABLE))
    (state / "numbers.json").write_text('{"nA": "1", "nB": "2"}')
    return state


@pytest.mark.view
def test_render_assets_writes_the_table_and_the_numbers(tmp_path: Path) -> None:
    main([str(_state(tmp_path)), str(tmp_path / "assets")])

    written = {path.name for path in (tmp_path / "assets" / "41_demo").iterdir()}

    assert written == {"41_demo_tab_demo.tex", "41_demo_num.tex"}

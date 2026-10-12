"""Facts about the contact sheet that shows three figures under each style preset."""

from pathlib import Path

import pytest
from presets_sheet import sheet_images


@pytest.mark.view
def test_presets_sheet_orders_rows_by_preset() -> None:
    images = sheet_images(Path("root"), ("paper", "large"), ("a", "b"))
    assert list(images.values()) == [
        Path("root/paper/a.png"),
        Path("root/paper/b.png"),
        Path("root/large/a.png"),
        Path("root/large/b.png"),
    ]

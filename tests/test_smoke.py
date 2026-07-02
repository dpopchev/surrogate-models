"""Smoke test: the package imports cleanly."""

import surrogate_models


def test_package_imports() -> None:
    assert surrogate_models is not None

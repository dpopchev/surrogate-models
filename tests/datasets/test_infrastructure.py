"""Tests for the datasets infrastructure -- factory_datasetid (imperative shell).

factory_datasetid mints identity from a UUID, so it is nondeterministic and lives
in the shell, never the domain. One assert per test.
"""

from surrogate_models.datasets.infrastructure import factory_datasetid


def test_factory_datasetid_is_eight_characters() -> None:
    assert len(factory_datasetid()) == 8


def test_factory_datasetid_is_lowercase_hex() -> None:
    assert all(char in "0123456789abcdef" for char in factory_datasetid())


def test_factory_datasetid_mints_distinct_ids() -> None:
    assert factory_datasetid() != factory_datasetid()

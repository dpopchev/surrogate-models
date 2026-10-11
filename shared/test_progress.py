"""Facts about the progress text of a run, on hand-picked epochs and seconds."""

import pytest

from shared.progress import approx_minutes, stop_window


class TestStopWindow:
    pytestmark = pytest.mark.view

    def test_spans_patience_to_max_epochs_after_a_new_best(self) -> None:
        assert stop_window(epoch=17, best_epoch=17, patience=20, max_epochs=500) == (20, 483)

    def test_never_lets_the_earliest_stop_pass_max_epochs(self) -> None:
        assert stop_window(epoch=495, best_epoch=490, patience=20, max_epochs=500) == (5, 5)


class TestApproxMinutes:
    pytestmark = pytest.mark.view

    def test_rounds_to_the_nearest_minute(self) -> None:
        assert approx_minutes(epochs=20, epoch_seconds=8.0) == "~3m"

    def test_states_hours_beyond_an_hour(self) -> None:
        assert approx_minutes(epochs=483, epoch_seconds=8.0) == "~1h 4m"

    def test_reads_less_than_a_minute_under_half_a_minute(self) -> None:
        assert approx_minutes(epochs=3, epoch_seconds=8.0) == "~<1m"

"""Facts about the run diagnostics figures, on hand-written histories and errors."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from shared.diagnostics import chosen_curves, curve_overlay, error_cdf, loss_curve

# Three epochs; the valid loss is lowest at epoch 2.
HISTORY = [
    {"epoch": 1, "train_loss": 1.0, "valid_loss": 0.8, "event_lr": 1e-3},
    {"epoch": 2, "train_loss": 0.4, "valid_loss": 0.5, "event_lr": 9e-4},
    {"epoch": 3, "train_loss": 0.2, "valid_loss": 0.6, "event_lr": 8e-4},
]


def test_the_loss_curve_carries_the_learning_rate_on_a_twin_axis() -> None:
    assert len(loss_curve(HISTORY).axes) == 2


# Five curves of four rows along x; curve 3 is predicted 50% too high, the others within 1-5%.
CURVE_X = np.tile(np.arange(4.0), 5)
CURVE_IDS = np.repeat(np.arange(5), 4)
CURVE_TRUE = 1.0 + CURVE_X / 10
CURVE_PRED = CURVE_TRUE * np.repeat([1.01, 1.02, 1.03, 1.5, 1.05], 4)


def test_the_worst_curve_is_chosen_first() -> None:
    assert chosen_curves(CURVE_TRUE, CURVE_PRED, CURVE_IDS, k=2)[0] == 3


def test_the_median_curve_comes_last() -> None:
    assert chosen_curves(CURVE_TRUE, CURVE_PRED, CURVE_IDS, k=2)[-1] == 2


def test_the_overlay_draws_a_true_and_a_predicted_line_per_chosen_curve() -> None:
    figure = curve_overlay(CURVE_X, CURVE_TRUE, CURVE_PRED, CURVE_IDS, k=2)
    assert len(figure.axes[0].get_lines()) == 6


def test_the_overlay_labels_each_curve_by_its_name() -> None:
    names = {curve: f"p = {curve}" for curve in range(5)}
    figure = curve_overlay(CURVE_X, CURVE_TRUE, CURVE_PRED, CURVE_IDS, k=2, names=names)
    legend = figure.axes[0].get_legend()
    texts = legend.get_texts() if legend is not None else []
    assert texts[0].get_text().startswith("p = 3,")


def test_the_error_cdf_legend_names_the_median_the_tail_and_the_worst() -> None:
    legend = error_cdf(np.linspace(0.001, 0.1, 100)).axes[0].get_legend()
    texts = legend.get_texts() if legend is not None else []
    assert [text.get_text().split()[0] for text in texts] == [
        "p50",
        "p95",
        "p99",
        "max",
    ]


def test_the_best_epoch_is_marked_at_the_lowest_valid_loss() -> None:
    lines = loss_curve(HISTORY).axes[0].get_lines()
    best = next(line for line in lines if str(line.get_label()).startswith("best epoch"))
    assert np.asarray(best.get_xdata()).tolist() == [2, 2]

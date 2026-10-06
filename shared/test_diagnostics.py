"""Facts about the run diagnostics figures, on hand-written histories and errors."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from shared.diagnostics import error_cdf, loss_curve

# Three epochs; the valid loss is lowest at epoch 2.
HISTORY = [
    {"epoch": 1, "train_loss": 1.0, "valid_loss": 0.8, "event_lr": 1e-3},
    {"epoch": 2, "train_loss": 0.4, "valid_loss": 0.5, "event_lr": 9e-4},
    {"epoch": 3, "train_loss": 0.2, "valid_loss": 0.6, "event_lr": 8e-4},
]


def test_the_loss_curve_carries_the_learning_rate_on_a_twin_axis() -> None:
    assert len(loss_curve(HISTORY).axes) == 2


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

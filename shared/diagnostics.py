"""The figures a surrogate run leaves for the developer, not the paper (W-035).

loss_curve shows how training went (train and valid loss, the learning rate, the best epoch);
error_cdf shows the shape of the test errors behind one MARE (median, tail and worst row).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# --- pure functions ---------------------------------------------------------------------------


def loss_curve(history: Sequence[Mapping[str, Any]]) -> Figure:
    """Train and valid loss per epoch (log scale), the learning rate on a twin axis and the
    epoch of the lowest valid loss marked."""
    epochs = [row["epoch"] for row in history]
    figure, loss = plt.subplots(layout="constrained")
    loss.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    loss.plot(epochs, [row["valid_loss"] for row in history], label="valid loss")
    best = min(history, key=lambda row: row["valid_loss"])["epoch"]
    loss.axvline(best, color="black", linestyle="--", linewidth=0.8, label=f"best epoch {best}")
    loss.set_yscale("log")
    loss.set_xlabel("epoch")
    loss.set_ylabel("loss (standardized target)")
    rate = loss.twinx()
    rate.plot(epochs, [row["event_lr"] for row in history], color="grey", linestyle=":")
    rate.set_ylabel("learning rate")
    loss.legend(loc="upper right")
    return figure


def error_cdf(relative_errors: np.ndarray) -> Figure:
    """The empirical CDF of the relative errors with p50, p95, p99 and the maximum marked."""
    ordered = np.sort(relative_errors)
    share = np.arange(1, len(ordered) + 1) / len(ordered)
    figure, axes = plt.subplots(layout="constrained")
    axes.plot(ordered, share, color="black", linewidth=1.0)
    marks = {
        "p50": np.percentile(ordered, 50),
        "p95": np.percentile(ordered, 95),
        "p99": np.percentile(ordered, 99),
        "max": ordered[-1],
    }
    for (name, value), style in zip(marks.items(), ("-", "--", "-.", ":"), strict=True):
        axes.axvline(value, linestyle=style, linewidth=0.8, label=f"{name} {value:.2e}")
    axes.set_xscale("log")
    axes.set_xlabel("relative error")
    axes.set_ylabel("share of test rows")
    axes.legend(loc="lower right")
    return figure

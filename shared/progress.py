"""How far a run has come, as text for the logs: durations and the window an early stop may fall in.

duration_text reads elapsed and remaining seconds for the batch progress line and the run
listings; stop_window and approx_minutes feed the per-epoch stop estimate of the training
callbacks (shared/surrogate.py).
"""

# --- pure functions ---------------------------------------------------------------------------


def stop_window(epoch: int, best_epoch: int, patience: int, max_epochs: int) -> tuple[int, int]:
    """Epochs left until the earliest stop (patience runs out without a new best) and the
    latest (max_epochs)."""
    latest = max_epochs - epoch
    return min(patience - (epoch - best_epoch), latest), latest


def duration_text(seconds: float) -> str:
    """Seconds as "1h 4m 24s", "2m 40s" or "12s"."""
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _minutes_text(seconds: float) -> str:
    """Seconds to the nearest minute: "1h 4m", "3m", or "<1m" under half a minute."""
    hours, minutes = divmod(round(seconds / 60), 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m" if minutes else "<1m"


def approx_minutes(epochs: int, epoch_seconds: float) -> str:
    """The approximate time a number of epochs takes, to the nearest minute: "~3m"."""
    return f"~{_minutes_text(epochs * epoch_seconds)}"

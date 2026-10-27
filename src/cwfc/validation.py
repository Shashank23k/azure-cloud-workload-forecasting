from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: object
    validation_start: object
    validation_end: object


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = frame[timestamp_col]
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out

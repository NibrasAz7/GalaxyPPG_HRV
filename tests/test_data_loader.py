"""Unit tests for time-index handling in the data loading and windowing layers.

Device exports can contain interleaved or duplicated timestamps; the pipeline
must tolerate them (regression tests for the non-monotonic index failure that
aborted a cohort run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from run_pipeline import _window
from src.data_loader import _sort_by_time


def _frame(times: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"value": values},
        index=pd.DatetimeIndex([pd.Timestamp(t) for t in times]),
    )


def test_sort_by_time_orders_interleaved_rows():
    df = _frame(
        ["2024-03-19 00:00:02", "2024-03-19 00:00:00", "2024-03-19 00:00:01"],
        [2.0, 0.0, 1.0],
    )
    ordered = _sort_by_time(df)

    assert ordered.index.is_monotonic_increasing
    assert ordered["value"].tolist() == [0.0, 1.0, 2.0]


def test_sort_by_time_drops_duplicate_timestamps():
    df = _frame(
        ["2024-03-19 00:00:00", "2024-03-19 00:00:01", "2024-03-19 00:00:01"],
        [0.0, 1.0, 99.0],
    )
    ordered = _sort_by_time(df)

    assert len(ordered) == 2
    assert ordered["value"].tolist() == [0.0, 1.0]


def test_window_is_robust_to_unsorted_index():
    df = _frame(
        ["2024-03-19 00:00:02", "2024-03-19 00:00:00", "2024-03-19 00:00:04"],
        [2.0, 0.0, 4.0],
    )
    window = _window(
        df,
        pd.Timestamp("2024-03-19 00:00:01"),
        pd.Timestamp("2024-03-19 00:00:03"),
    )

    assert window["value"].tolist() == [2.0]


def test_window_returns_empty_outside_range():
    df = _frame(["2024-03-19 00:00:00"], [1.0])
    window = _window(
        df,
        pd.Timestamp("2024-03-19 01:00:00"),
        pd.Timestamp("2024-03-19 02:00:00"),
    )

    assert window.empty


def test_window_on_empty_frame():
    window = _window(
        pd.DataFrame(),
        pd.Timestamp("2024-03-19 00:00:00"),
        pd.Timestamp("2024-03-19 01:00:00"),
    )
    assert window.empty
    assert np.isfinite(1.0)

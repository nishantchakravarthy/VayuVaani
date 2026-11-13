import numpy as np
import pandas as pd

from vayuvaani.pipeline import clean_series


def _series(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="h"), dtype=float)


def test_removes_negative_and_huge_values():
    s = clean_series(_series([20, -5, 22, 5000, 25, 26, 27, 28, 29, 30]))
    assert (s.dropna() > 0).all() and (s.dropna() < 999).all()


def test_removes_stuck_sensor():
    s = clean_series(_series([20, 21] + [40] * 10 + [22, 23]))
    assert s.iloc[2:12].isna().all()


def test_removes_isolated_spike():
    vals = [20 + (i % 3) for i in range(30)]
    vals[15] = 300
    s = clean_series(_series(vals))
    assert s.iloc[15] < 100  # spike removed and gap interpolated


def test_fills_short_gaps_only():
    vals = [20.0, np.nan, np.nan, 26.0] + [30.0 + i for i in range(3)] + [np.nan] * 6 + [40.0]
    s = clean_series(_series(vals))
    assert s.iloc[1:3].notna().all()
    assert s.iloc[7:13].isna().all()

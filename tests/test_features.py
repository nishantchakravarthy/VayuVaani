import numpy as np
import pandas as pd

from vayuvaani import pipeline
from vayuvaani.features import ALL_COLS, add_origin_features, make_rows


def _table(settings):
    pipeline.fetch_demo(settings, "2026-01-01", "2026-01-20")
    return add_origin_features(pipeline.build(settings))


def test_rows_point_forward_in_time(settings):
    df = _table(settings)
    X, y, meta = make_rows(df, 48, 4)
    assert list(X.columns) == ALL_COLS
    gap = (pd.to_datetime(meta["target_time"]) - pd.to_datetime(meta["origin_time"])) / pd.Timedelta(hours=1)
    assert (gap == meta["horizon"]).all()
    assert meta["horizon"].between(1, 48).all()


def test_origin_features_use_only_the_past(settings):
    df = _table(settings)
    X, y, meta = make_rows(df, 48, 2)
    lookup = df.set_index(["station_id", "time"])["pm25"]
    idx = pd.MultiIndex.from_arrays([meta["station_id"], pd.to_datetime(meta["origin_time"])])
    np.testing.assert_allclose(X["pm25_lag0"].to_numpy(), lookup.reindex(idx).to_numpy())
    # The target value must not appear among the origin features.
    assert not np.allclose(X["pm25_lag0"].to_numpy(), y.to_numpy())


def test_targets_stay_within_one_station(settings):
    df = _table(settings)
    _, _, meta = make_rows(df, 48, None)
    last = df.groupby("station_id")["time"].max()
    assert (pd.to_datetime(meta["target_time"]) <= meta["station_id"].map(last)).all()


def test_live_rows_match_training_rows(settings):
    """A live forecast must build exactly the same features as training did."""
    from vayuvaani import forecast
    from vayuvaani.features import WEATHER_COLS, make_inference_rows

    pipeline.fetch_demo(settings, "2026-01-01", "2026-01-20")
    df = pipeline.build(settings)
    t0 = df["time"].max() - pd.Timedelta(hours=100)
    hist = df[(df["time"] <= t0) & (df["time"] > t0 - pd.Timedelta(hours=72))]
    fut = df[df["time"] > t0][["station_id", "time"] + WEATHER_COLS]
    ctx = forecast._future_context(settings, hist["time"].min(), fut["time"].max())
    Xi, mi = make_inference_rows(hist, fut, ctx, 48)

    d = add_origin_features(df)
    Xt, _, mt = make_rows(d, 48, None, origin_mask=(d["time"] == t0).to_numpy())
    Xi.index = mi["station_id"].astype(str) + "_" + mi["horizon"].astype(str)
    Xt.index = mt["station_id"].astype(str) + "_" + mt["horizon"].astype(str)
    np.testing.assert_allclose(Xt.loc[Xi.index].to_numpy(float), Xi.to_numpy(float), atol=1e-6)

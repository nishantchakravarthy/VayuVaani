"""NASA FIRMS fire data: downloading in chunks, rejecting a bad key, and retraining with it."""

import numpy as np
import pandas as pd
import pytest

from vayuvaani import pipeline, train
from vayuvaani.app import server
from vayuvaani.sources import firms


class FakeResp:
    def __init__(self, text):
        self.text = text


def test_download_counts_fires_and_rejects_bad_key(settings, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "abc123")
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        day = url.rstrip("/").split("/")[-1]
        return FakeResp(f"latitude,longitude,acq_date,frp\n12.97,77.59,{day},5.0\n13.5,77.0,{day},2.0\n")

    monkeypatch.setattr(firms, "get", fake_get)
    fires = firms.download(settings, "2026-01-01", "2026-01-20")
    assert len(calls) == 4                      # 20 days in 5-day chunks
    daily = firms.daily_counts(settings, fires)
    assert daily["fire_count"].sum() == 8       # both detections are within the radius

    monkeypatch.setattr(firms, "get", lambda url, **kw: FakeResp("Invalid MAP_KEY."))
    with pytest.raises(RuntimeError, match="rejected"):
        firms.download(settings, "2026-01-01", "2026-01-05")


def test_fire_data_reaches_training(settings, monkeypatch):
    pipeline.fetch_demo(settings, "2026-03-01", "2026-05-31")
    pipeline.build(settings)
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    monkeypatch.setattr(firms, "download", lambda s, start, end, source=None: pd.DataFrame({
        "latitude": 12.97, "longitude": 77.59, "frp": 3.0,
        "acq_date": pd.date_range(start, end, freq="2D").strftime("%Y-%m-%d")}))
    server._download_fires(settings)
    assert server._fire_days(settings) > 20
    pipeline.build(settings)
    df = pipeline.load_processed(settings)
    assert df["fires_prev_3d"].gt(0).mean() > 0.5
    train.run(settings, df)

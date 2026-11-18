"""Exercise the live forecast path with the network calls replaced by fake data."""

import numpy as np
import pandas as pd

from vayuvaani import forecast, pipeline, train
from vayuvaani.sources import openaq, weather


def test_live_forecast_with_fake_apis(settings, monkeypatch):
    pipeline.fetch_demo(settings, "2026-03-01", "2026-05-31")
    pipeline.build(settings)
    train.run(settings, pipeline.load_processed(settings))

    raw = settings.path("raw")
    st = pd.read_csv(raw / "stations.csv")
    st["pm25_sensor_id"] = np.arange(len(st)) + 100
    st["pm10_sensor_id"] = np.nan
    st.to_csv(raw / "stations.csv", index=False)

    now = pd.Timestamp.now(tz=settings.timezone).tz_localize(None).floor("h")

    def fake_hours(settings, sensor_id, start, end, **kw):
        t = pd.date_range(pd.Timestamp(start), now - pd.Timedelta(hours=2), freq="h")
        utc = t.tz_localize(settings.timezone).tz_convert("UTC")
        return pd.DataFrame({"time_utc": utc, "value": 40 + 10 * np.sin(np.arange(len(t)) / 4)})

    def fake_weather(settings, lat, lon, past_days=3, days=3, **kw):
        t = pd.date_range(now.normalize() - pd.Timedelta(days=past_days), periods=24 * (past_days + days), freq="h")
        return pd.DataFrame({"time": t, "temperature_2m": 25.0, "relative_humidity_2m": 60.0,
                             "precipitation": 0.0, "wind_speed_10m": 8.0, "wind_direction_10m": 90.0,
                             "surface_pressure": 912.0, "cloud_cover": 40.0})

    monkeypatch.setattr(openaq, "sensor_hours", fake_hours)
    monkeypatch.setattr(weather, "forecast", fake_weather)
    monkeypatch.setattr(weather, "forecast_many",
                        lambda settings, points, past_days=3, days=3, **kw:
                        [fake_weather(settings, a, b, past_days, days) for a, b in points])
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)

    out = forecast.live(settings)
    assert out["station_id"].nunique() == len(st)
    assert out.groupby("station_id")["horizon"].max().eq(48).all()
    # The latest reading is 2 hours old, so the forecast starts from it.
    assert (pd.to_datetime(out["origin_time"]) == now - pd.Timedelta(hours=2)).all()
    assert out["pm25_q50"].notna().all()
    assert out["t_wind_speed"].notna().all() if "t_wind_speed" in out else True


def test_live_reports_rejected_key(settings, monkeypatch):
    import pytest
    pipeline.fetch_demo(settings, "2026-03-01", "2026-04-15")
    raw = settings.path("raw")
    st = pd.read_csv(raw / "stations.csv")
    st["pm25_sensor_id"], st["pm10_sensor_id"] = np.arange(len(st)) + 100, np.nan
    st.to_csv(raw / "stations.csv", index=False)

    def unauthorized(*a, **kw):
        raise RuntimeError("401 Client Error: Unauthorized")

    monkeypatch.setattr(openaq, "sensor_hours", unauthorized)
    with pytest.raises(RuntimeError, match="rejected the API key"):
        forecast.live(settings)

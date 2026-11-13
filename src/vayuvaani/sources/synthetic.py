"""Synthetic demo data with the same shape as the real sources.

Use it to run and test the whole pipeline before you have API keys.
The numbers are made up. Never report results from this data as real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from . import events as events_src

DEMO_STATIONS = [
    ("demo_a", "Demo Station A (central)", 12.975, 77.590, 1.10),
    ("demo_b", "Demo Station B (east)", 12.970, 77.750, 1.20),
    ("demo_c", "Demo Station C (south)", 12.910, 77.600, 1.00),
    ("demo_d", "Demo Station D (north)", 13.060, 77.590, 0.90),
    ("demo_e", "Demo Station E (west)", 12.960, 77.500, 0.85),
    ("demo_f", "Demo Station F (south-east)", 12.915, 77.665, 1.25),
]


def generate(settings: Settings, start: str, end: str, seed: int = 7):
    """Return (stations, measurements, weather, fire_daily) like the real fetchers."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    n = len(times)
    doy = times.dayofyear.values
    hour = times.hour.values
    month = times.month.values

    # Weather shared across the city, with small per-station noise added later.
    temp = 24 + 4 * np.sin(2 * np.pi * (doy - 60) / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    monsoon = np.isin(month, [6, 7, 8, 9])
    wind = np.clip(2.5 + 3.0 * monsoon + 1.5 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.gamma(2, 0.6, n), 0.2, None)
    wdir = np.where(monsoon, 260, 90) + rng.normal(0, 40, n)
    rain_prob = np.where(monsoon, 0.18, 0.03)
    precip = np.where(rng.random(n) < rain_prob, rng.gamma(1.2, 2.5, n), 0.0)
    rh = np.clip(55 + 20 * monsoon - 15 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 6, n), 15, 100)
    pressure = 912 + rng.normal(0, 1.5, n)
    cloud = np.clip(30 + 50 * monsoon + rng.normal(0, 15, n), 0, 100)

    # Fires: many in the dry season, few in the monsoon.
    days = pd.date_range(start, end, freq="D")
    dry = days.month.isin([1, 2, 3, 4, 12])
    fire_count = rng.poisson(np.where(dry, 40, 6))
    fire_daily = pd.DataFrame({"date": days, "fire_count": fire_count, "fire_frp": fire_count * rng.gamma(2, 6, len(days))})
    fire_hourly = pd.Series(fire_count, index=days).reindex(times.normalize()).values

    flags = events_src.daily_flags(events_src.load(settings), start, end).set_index("date")
    fireworks = flags["event_fireworks"].reindex(times.normalize()).fillna(0).values

    seasonal = 1 + 0.45 * np.cos(2 * np.pi * (doy - 15) / 365)
    diurnal = 1 + 0.35 * np.exp(-((hour - 9) ** 2) / 4) + 0.45 * np.exp(-((hour - 22) ** 2) / 6) - 0.25 * np.exp(-((hour - 15) ** 2) / 8)

    stations, meas, weather = [], [], []
    for sid, name, lat, lon, scale in DEMO_STATIONS:
        noise = np.zeros(n)
        eps = rng.normal(0, 0.18, n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + eps[i]
        washout = np.exp(-0.08 * pd.Series(precip).rolling(6, min_periods=1).sum().values)
        pm25 = (
            32 * scale * seasonal * diurnal * (3.0 / (wind + 1.0)) ** 0.6 * washout
            + 0.25 * fire_hourly
            + fireworks * np.where((hour >= 19) | (hour <= 2), 90, 25)
        ) * np.exp(noise)
        pm25 = np.clip(pm25, 2, None)
        pm10 = pm25 * (1.7 + rng.normal(0, 0.15, n))
        # Drop ~3% of readings, like real station outages.
        missing = rng.random(n) < 0.03
        pm25[missing] = np.nan
        pm10[missing] = np.nan

        stations.append({"station_id": sid, "name": name, "lat": lat, "lon": lon, "provider": "SYNTHETIC", "is_monitor": True})
        for param, vals in (("pm25", pm25), ("pm10", pm10)):
            meas.append(pd.DataFrame({"station_id": sid, "time": times, "parameter": param, "value": np.round(vals, 1)}))
        weather.append(pd.DataFrame({
            "time": times, "station_id": sid,
            "temperature_2m": np.round(temp + rng.normal(0, 0.4, n), 1),
            "relative_humidity_2m": np.round(rh, 0),
            "precipitation": np.round(precip, 1),
            "wind_speed_10m": np.round(wind * 3.6, 1),  # Open-Meteo reports km/h
            "wind_direction_10m": np.round(wdir % 360, 0),
            "surface_pressure": np.round(pressure, 1),
            "cloud_cover": np.round(cloud, 0),
        }))
    return (pd.DataFrame(stations), pd.concat(meas, ignore_index=True),
            pd.concat(weather, ignore_index=True), fire_daily)

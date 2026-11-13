"""Hourly weather from Open-Meteo (no API key needed).

- archive(): past weather (ERA5 reanalysis) for training.
- forecast(): next few days, used as the future weather when forecasting AQI.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Settings
from .http import get

log = logging.getLogger(__name__)


def _to_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for v in variables:
        df[v] = hourly.get(v)
    return df


def archive(settings: Settings, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    cfg = settings.sources["open_meteo"]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(cfg["hourly"]),
        "timezone": settings.timezone,
    }
    return _to_frame(get(cfg["archive_url"], params=params).json(), cfg["hourly"])


def forecast(settings: Settings, lat: float, lon: float, past_days: int = 3, days: int = 3) -> pd.DataFrame:
    cfg = settings.sources["open_meteo"]
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(cfg["hourly"]),
        "timezone": settings.timezone,
        "past_days": past_days, "forecast_days": days,
    }
    return _to_frame(get(cfg["forecast_url"], params=params).json(), cfg["hourly"])


def for_stations(settings: Settings, stations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Archive weather for every station. Stations closer than ~3 km share a grid cell,
    so results are cached by rounded coordinates."""
    cache: dict[tuple, pd.DataFrame] = {}
    frames = []
    for _, st in stations.iterrows():
        key = (round(st["lat"], 2), round(st["lon"], 2))
        if key not in cache:
            log.info("Weather for %s", st["name"])
            cache[key] = archive(settings, st["lat"], st["lon"], start, end)
        df = cache[key].copy()
        df["station_id"] = st["station_id"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

"""Collect raw data and build one clean hourly table per station.

Raw files (data/raw):
    stations.csv          station_id, name, lat, lon, provider, is_monitor
    measurements.csv.gz   station_id, time, parameter (pm25/pm10), value
    weather.csv.gz        station_id, time, <Open-Meteo hourly variables>
    fires_daily.csv       date, fire_count, fire_frp

Processed file (data/processed/hourly.csv.gz): one row per station per hour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Settings
from .sources import cpcb, events, firms, openaq, synthetic, weather

log = logging.getLogger(__name__)

MAX_PM = 999.0          # readings above this are sensor faults
FLATLINE_HOURS = 8      # identical readings for this long are treated as stuck sensors
MAX_GAP_FILL_HOURS = 3  # short gaps are interpolated; longer gaps stay missing


# ---------------------------------------------------------------- raw files

def _raw(settings: Settings, name: str):
    return settings.path("raw") / name


def save_raw(settings: Settings, stations, meas, wx, fires_daily) -> None:
    stations.to_csv(_raw(settings, "stations.csv"), index=False)
    meas.to_csv(_raw(settings, "measurements.csv.gz"), index=False)
    wx.to_csv(_raw(settings, "weather.csv.gz"), index=False)
    fires_daily.to_csv(_raw(settings, "fires_daily.csv"), index=False)
    log.info("Saved raw data: %d stations, %d measurements, %d weather rows, %d fire days",
             len(stations), len(meas), len(wx), len(fires_daily))


def load_raw(settings: Settings):
    stations = pd.read_csv(_raw(settings, "stations.csv"))
    meas = pd.read_csv(_raw(settings, "measurements.csv.gz"), parse_dates=["time"])
    wx = pd.read_csv(_raw(settings, "weather.csv.gz"), parse_dates=["time"])
    fires_daily = pd.read_csv(_raw(settings, "fires_daily.csv"), parse_dates=["date"])
    return stations, meas, wx, fires_daily


def fetch(settings: Settings, start: str, end: str, skip_fires: bool = False) -> None:
    """Download real data for [start, end] and save it to data/raw."""
    stations, meas = openaq.download(settings, start, end)
    ccr_stations, ccr_meas = cpcb.load_ccr_files(settings)
    if not ccr_stations.empty:
        stations = pd.concat([stations, ccr_stations], ignore_index=True).drop_duplicates("station_id")
        meas = pd.concat([meas, ccr_meas], ignore_index=True)
    if stations.empty:
        raise RuntimeError("No stations found. Check OPENAQ_API_KEY and the city bbox in settings.json.")
    wx = weather.for_stations(settings, stations, start, end)
    if skip_fires:
        fires_daily = pd.DataFrame(columns=["date", "fire_count", "fire_frp"])
    else:
        fires_daily = firms.daily_counts(settings, firms.download(settings, start, end))
    save_raw(settings, stations, meas, wx, fires_daily)


def fetch_demo(settings: Settings, start: str, end: str) -> None:
    """Write synthetic data in the same format as fetch()."""
    save_raw(settings, *synthetic.generate(settings, start, end))


# ---------------------------------------------------------------- cleaning

def clean_series(s: pd.Series) -> pd.Series:
    """Quality-control one station's hourly series (index = hourly time)."""
    s = s.astype(float).copy()
    s[(s <= 0) | (s > MAX_PM)] = np.nan
    # Stuck sensor: the same value repeated for many hours.
    group = (~s.diff().eq(0)).cumsum()          # consecutive equal values share a group
    run_length = s.groupby(group).transform("size")
    s[(run_length >= FLATLINE_HOURS) & s.notna()] = np.nan
    # Isolated spikes: far above both neighbours and the local median.
    med = s.rolling(25, center=True, min_periods=6).median()
    spike = (s > 4 * med) & (s > s.shift(1) * 3) & (s > s.shift(-1) * 3)
    s[spike] = np.nan
    return fill_short_gaps(s, MAX_GAP_FILL_HOURS)


def fill_short_gaps(s: pd.Series, max_hours: int) -> pd.Series:
    """Linearly interpolate gaps of at most max_hours; leave longer gaps empty.
    (pandas' interpolate(limit=...) would partly fill long gaps too.)"""
    missing = s.isna()
    gap_id = (~missing).cumsum()
    gap_len = missing.groupby(gap_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    filled[missing & (gap_len > max_hours)] = np.nan
    return filled


def build(settings: Settings) -> pd.DataFrame:
    """Merge raw files into data/processed/hourly.csv.gz."""
    stations, meas, wx, fires_daily = load_raw(settings)
    meas["time"] = meas["time"].dt.floor("h")
    wide = (meas.pivot_table(index=["station_id", "time"], columns="parameter", values="value", aggfunc="mean")
                .reset_index())
    for col in ("pm25", "pm10"):
        if col not in wide:
            wide[col] = np.nan

    frames = []
    for sid, g in wide.groupby("station_id"):
        g = g.set_index("time").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full)
        g.index.name = "time"
        g["station_id"] = sid
        g["pm25"] = clean_series(g["pm25"])
        g["pm10"] = clean_series(g["pm10"])
        frames.append(g.reset_index())
    df = pd.concat(frames, ignore_index=True)

    df = df.merge(stations[["station_id", "name", "lat", "lon"]], on="station_id", how="left")
    df = df.merge(wx, on=["station_id", "time"], how="left")

    start, end = df["time"].min(), df["time"].max()
    df = add_daily_context(df, settings, fires_daily, start, end)

    out = settings.path("processed") / "hourly.csv.gz"
    df.to_csv(out, index=False)
    coverage = df["pm25"].notna().mean() * 100
    log.info("Built %s: %d rows, %d stations, PM2.5 coverage %.1f%%", out, len(df), df["station_id"].nunique(), coverage)
    return df


def add_daily_context(df: pd.DataFrame, settings: Settings, fires_daily: pd.DataFrame, start, end) -> pd.DataFrame:
    """Add fire counts (previous days) and event flags, joined by local date."""
    days = pd.DataFrame({"date": pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")})
    fd = days.merge(fires_daily, on="date", how="left")
    for c in ("fire_count", "fire_frp"):
        fd[c] = pd.to_numeric(fd[c], errors="coerce").fillna(0.0) if c in fd else 0.0
    # Use only fires already observed: yesterday's count and the 3-day total up to yesterday.
    fd["fires_prev_day"] = fd["fire_count"].shift(1).fillna(0)
    fd["fires_prev_3d"] = fd["fire_count"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    fd["frp_prev_3d"] = fd["fire_frp"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    fd = fd[["date", "fires_prev_day", "fires_prev_3d", "frp_prev_3d"]]

    flags = events.daily_flags(events.load(settings), start, end)
    df = df.copy()
    df["date"] = df["time"].dt.normalize()
    df = df.merge(fd, on="date", how="left").merge(flags, on="date", how="left")
    return df.drop(columns="date")


def load_processed(settings: Settings) -> pd.DataFrame:
    return pd.read_csv(settings.path("processed") / "hourly.csv.gz", parse_dates=["time"])

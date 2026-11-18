"""Make a 48-hour forecast for every station.

live():  refreshes the last few days of PM2.5 and weather from the APIs, then
         forecasts the next 48 hours using the Open-Meteo weather forecast.
replay(): for demos and testing: pretends "now" is a past hour and uses the
         observed weather as the forecast, so you can compare against what happened.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import pipeline
from .config import Settings
from .features import EVENT_COLS, WEATHER_COLS, make_inference_rows
from .model import QuantileForecaster, summarise
from .sources import events, firms, openaq, weather

log = logging.getLogger(__name__)

HISTORY_HOURS = 72  # enough for the 48-hour lag and 24-hour rolling features


def _future_context(settings: Settings, start, end) -> pd.DataFrame:
    return events.daily_flags(events.load(settings), start, end)[["date"] + EVENT_COLS]


def _predict(settings: Settings, history: pd.DataFrame, future_wx: pd.DataFrame) -> pd.DataFrame:
    fc = QuantileForecaster.load(settings.path("models") / "forecaster.joblib")
    max_h = settings.model["max_horizon_hours"]
    start = history["time"].min()
    ctx = _future_context(settings, start, future_wx["time"].max() + pd.Timedelta(hours=max_h))
    X, meta = make_inference_rows(history, future_wx, ctx, max_h)
    if X.empty:
        raise RuntimeError("No station has a recent PM2.5 reading to forecast from.")
    pred = summarise(fc.predict(X), fc.quantiles)
    out = pd.concat([meta, pred.round(1)], axis=1)
    # Keep the model inputs so the app can explain each forecast hour.
    feats = pd.concat([meta[["station_id", "target_time"]], X.reset_index(drop=True)], axis=1)
    feats.to_csv(settings.path("reports") / "latest_features.csv", index=False)
    path = settings.path("reports") / "latest_forecast.csv"
    out.to_csv(path, index=False)
    log.info("Saved %d forecast rows to %s", len(out), path)
    return out


def replay(settings: Settings, issue_time: str | None = None) -> pd.DataFrame:
    df = pipeline.load_processed(settings)
    max_h = settings.model["max_horizon_hours"]
    t0 = pd.Timestamp(issue_time) if issue_time else df["time"].max() - pd.Timedelta(hours=max_h)
    history = df[(df["time"] <= t0) & (df["time"] > t0 - pd.Timedelta(hours=HISTORY_HOURS))]
    future = df[(df["time"] > t0) & (df["time"] <= t0 + pd.Timedelta(hours=max_h))]
    out = _predict(settings, history, future[["station_id", "time"] + WEATHER_COLS])
    actual = future.set_index(["station_id", "time"])["pm25"]
    out["actual_pm25"] = actual.reindex(pd.MultiIndex.from_arrays([out["station_id"], out["target_time"]])).to_numpy()
    out.to_csv(settings.path("reports") / "latest_forecast.csv", index=False)
    return out


def live(settings: Settings) -> pd.DataFrame:
    """Forecast from now using fresh data. Needs OPENAQ_API_KEY (FIRMS_MAP_KEY optional)."""
    raw = settings.path("raw")
    stations = pd.read_csv(raw / "stations.csv")
    now = pd.Timestamp.now(tz=settings.timezone).tz_localize(None).floor("h")
    start = (now - pd.Timedelta(hours=HISTORY_HOURS + 24)).strftime("%Y-%m-%dT%H:%M:%S")
    end = now.strftime("%Y-%m-%dT%H:%M:%S")

    t0 = time.time()
    # Recent PM readings: all sensors at once (a few in parallel, kind to the rate limit).
    jobs = []
    for _, st in stations.iterrows():
        for param in ("pm25", "pm10"):
            sid = st.get(f"{param}_sensor_id")
            if not pd.isna(sid):
                jobs.append((st["station_id"], st["name"], param, int(sid)))

    def fetch(job):
        station_id, name, param, sid = job
        try:
            d = openaq.sensor_hours(settings, sid, start, end, retries=2, timeout=25)
        except Exception as exc:  # one slow or broken sensor shouldn't hold up the rest
            log.warning("Skipped %s %s (%s)", name, param, str(exc)[:80])
            return None
        if d.empty:
            return None
        d["time"] = d["time_utc"].dt.tz_convert(settings.timezone).dt.tz_localize(None)
        d["station_id"], d["parameter"] = station_id, param
        return d.drop(columns="time_utc")

    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = [f for f in pool.map(fetch, jobs) if f is not None]
    if not frames:
        raise RuntimeError("No recent readings from OpenAQ.")
    meas = pd.concat(frames, ignore_index=True)
    log.info("Latest readings from %d sensors in %.0f s", len(frames), time.time() - t0)

    # Weather: last 4 days plus next 3 days for every station, in one request.
    t1 = time.time()
    points = list(zip(stations["lat"], stations["lon"]))
    try:
        frames_wx = weather.forecast_many(settings, points, past_days=4, days=3)
    except Exception as exc:
        log.warning("Combined weather request failed (%s); asking per station", str(exc)[:80])
        with ThreadPoolExecutor(max_workers=6) as pool:
            frames_wx = list(pool.map(lambda p: weather.forecast(settings, p[0], p[1], past_days=4, days=3), points))
    for w, sid in zip(frames_wx, stations["station_id"]):
        w["station_id"] = sid
    wx = pd.concat(frames_wx, ignore_index=True)
    log.info("Weather forecast in %.0f s", time.time() - t1)

    try:
        fires = firms.daily_counts(settings, firms.download(
            settings, (now - pd.Timedelta(days=5)).date().isoformat(), now.date().isoformat(),
            source=settings.sources["firms"]["source_recent"]))
    except RuntimeError:
        log.warning("FIRMS_MAP_KEY not set; fire features will be zero.")
        fires = pd.DataFrame(columns=["date", "fire_count", "fire_frp"])

    history = _recent_table(settings, stations, meas, wx[wx["time"] <= now], fires)
    history = drop_stale(settings, history, stations, now)
    # Pass all weather hours: if a station's latest reading is a few hours old,
    # the forecast starts from that reading and needs the weather in between.
    out = _predict(settings, history, wx)
    log.info("Forecast ready in %.0f s", time.time() - t0)
    return out


MAX_STALE_HOURS = 6


def drop_stale(settings, history: pd.DataFrame, stations: pd.DataFrame, now) -> pd.DataFrame:
    """Leave out stations whose latest reading is too old to forecast from,
    and record them so the app can show them as offline."""
    last = history.dropna(subset=["pm25"]).groupby("station_id")["time"].max()
    fresh = last[last >= now - pd.Timedelta(hours=MAX_STALE_HOURS)].index
    offline = []
    for _, st in stations.iterrows():
        if st["station_id"] not in fresh:
            t = last.get(st["station_id"])
            offline.append({"station_id": st["station_id"], "name": st["name"],
                            "last_reading": None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")})
    import json
    with open(settings.path("reports") / "offline_stations.json", "w", encoding="utf-8") as f:
        json.dump(offline, f, indent=2)
    if offline:
        log.info("Skipping %d stations with no reading in the last %d hours", len(offline), MAX_STALE_HOURS)
    return history[history["station_id"].isin(fresh)]


def _recent_table(settings, stations, meas, wx, fires) -> pd.DataFrame:
    meas = meas.copy()
    meas["time"] = meas["time"].dt.floor("h")
    wide = meas.pivot_table(index=["station_id", "time"], columns="parameter", values="value").reset_index()
    for col in ("pm25", "pm10"):
        if col not in wide:
            wide[col] = np.nan
    frames = []
    for sid, g in wide.groupby("station_id"):
        g = g.set_index("time").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), wx["time"].max(), freq="h"))
        g.index.name = "time"
        g["station_id"] = sid
        g["pm25"] = pipeline.clean_series(g["pm25"])
        g["pm10"] = pipeline.clean_series(g["pm10"])
        frames.append(g.reset_index())
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(stations[["station_id", "name", "lat", "lon"]], on="station_id", how="left")
    df = df.merge(wx, on=["station_id", "time"], how="left")
    return pipeline.add_daily_context(df, settings, fires, df["time"].min(), df["time"].max())


def format_forecast(out: pd.DataFrame) -> str:
    """Short text summary: worst hour in the next 24 h per station."""
    lines = []
    for sid, g in out.groupby("station_id", sort=False):
        nxt = g[g["horizon"] <= 24]
        worst = nxt.loc[nxt["pm25_q50"].idxmax()]
        lines.append(
            f"{worst['name'][:34]:34}  peak {worst['pm25_q50']:5.0f} µg/m³ "
            f"({worst['pm25_q10']:.0f}–{worst['pm25_q90']:.0f}) at {pd.Timestamp(worst['target_time']):%a %H:%M}  "
            f"{worst['category']:<12} chance of Poor+: {worst['p_poor_plus'] * 100:3.0f}%"
        )
    return "\n".join(lines)

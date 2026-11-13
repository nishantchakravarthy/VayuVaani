"""Turn the hourly station table into model rows.

Each row answers: "standing at hour t (the origin) at station s, what will PM2.5 be
at hour t + h?" for a horizon h of 1–48 hours.

- Origin features describe what is known at t: recent PM2.5/PM10, current weather,
  fires seen in the previous days.
- Target features describe hour t + h: forecast weather, time of day, day of week,
  events. In training, observed weather stands in for the weather forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
                "wind_direction_10m", "surface_pressure", "cloud_cover"]
EVENT_COLS = ["event_fireworks", "event_festival", "event_holiday", "event_dry_season"]
FIRE_COLS = ["fires_prev_day", "fires_prev_3d", "frp_prev_3d"]

PM_LAGS = [0, 1, 2, 3, 6, 12, 24, 48]

ORIGIN_COLS = (
    [f"pm25_lag{l}" for l in PM_LAGS]
    + ["pm25_mean3", "pm25_mean6", "pm25_mean24", "pm25_std24", "pm25_max24", "pm25_trend3",
       "pm10_lag0", "pm10_mean24", "pm25_pm10_ratio",
       "o_wind_speed", "o_rh", "o_temp", "o_rain6", "lat", "lon"]
    + FIRE_COLS
)
TARGET_COLS = [
    "horizon", "t_hour_sin", "t_hour_cos", "t_dow", "t_weekend",
    "t_temp", "t_rh", "t_wind_speed", "t_wind_u", "t_wind_v", "t_rain", "t_rain6",
    "t_pressure", "t_cloud", "t_wind_mean_h", "t_rain_sum_h",
] + [f"t_{c}" for c in EVENT_COLS]
FEATURES = ORIGIN_COLS + TARGET_COLS
# Day-of-year features made the model memorise last year's season instead of
# reacting to current conditions (tested on real Bengaluru data: average error
# 11.5 -> 10.6 µg/m³ without them). They are still produced so that models
# trained with them keep working, but new models don't use them.
EXTRA_COLS = ["t_doy_sin", "t_doy_cos"]
ALL_COLS = FEATURES + EXTRA_COLS


def _per_station_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)
    for c in WEATHER_COLS + EVENT_COLS + FIRE_COLS:
        if c not in df:
            df[c] = np.nan if c in WEATHER_COLS else 0
    return df


def add_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ORIGIN_COLS. Assumes one continuous hourly series per station."""
    df = _per_station_sorted(df)
    g = df.groupby("station_id", sort=False)
    pm = g["pm25"]
    for l in PM_LAGS:
        df[f"pm25_lag{l}"] = pm.shift(l)
    for w in (3, 6, 24):
        df[f"pm25_mean{w}"] = pm.transform(lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    df["pm25_std24"] = pm.transform(lambda s: s.rolling(24, min_periods=12).std())
    df["pm25_max24"] = pm.transform(lambda s: s.rolling(24, min_periods=12).max())
    df["pm25_trend3"] = df["pm25_lag0"] - df["pm25_lag3"]
    df["pm10_lag0"] = df["pm10"]
    df["pm10_mean24"] = g["pm10"].transform(lambda s: s.rolling(24, min_periods=12).mean())
    df["pm25_pm10_ratio"] = df["pm25_mean24"] / df["pm10_mean24"].replace(0, np.nan)
    df["o_wind_speed"] = df["wind_speed_10m"]
    df["o_rh"] = df["relative_humidity_2m"]
    df["o_temp"] = df["temperature_2m"]
    df["o_rain6"] = g["precipitation"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    return df


def target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour features describing a target hour (calendar, weather, events)."""
    t = df["time"]
    rad = np.deg2rad(df["wind_direction_10m"])
    ws = df["wind_speed_10m"]
    out = pd.DataFrame({
        "t_hour_sin": np.sin(2 * np.pi * t.dt.hour / 24),
        "t_hour_cos": np.cos(2 * np.pi * t.dt.hour / 24),
        "t_dow": t.dt.dayofweek,
        "t_weekend": (t.dt.dayofweek >= 5).astype(int),
        "t_doy_sin": np.sin(2 * np.pi * t.dt.dayofyear / 365.25),
        "t_doy_cos": np.cos(2 * np.pi * t.dt.dayofyear / 365.25),
        "t_temp": df["temperature_2m"],
        "t_rh": df["relative_humidity_2m"],
        "t_wind_speed": ws,
        # Wind components: direction the wind blows FROM, so negate.
        "t_wind_u": -ws * np.sin(rad),
        "t_wind_v": -ws * np.cos(rad),
        "t_rain": df["precipitation"],
        "t_rain6": df.groupby("station_id", sort=False)["precipitation"].transform(
            lambda s: s.rolling(6, min_periods=1).sum()),
        "t_pressure": df["surface_pressure"],
        "t_cloud": df["cloud_cover"],
    }, index=df.index)
    for c in EVENT_COLS:
        out[f"t_{c}"] = df[c].fillna(0)
    return out


def _cum(df: pd.DataFrame, col: str) -> np.ndarray:
    """Per-station cumulative sum, used to average weather between origin and target."""
    return df.groupby("station_id", sort=False)[col].transform(lambda s: s.fillna(0).cumsum()).to_numpy()


def make_rows(df: pd.DataFrame, max_h: int = 48, horizons_per_origin: int | None = 8,
              origin_step: int = 1, rng: np.random.Generator | None = None,
              origin_mask: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X, y, meta) from an hourly table that already has origin features.

    horizons_per_origin: sample this many random horizons per origin (training).
                         None means use every horizon 1..max_h (evaluation).
    origin_mask: optional boolean array selecting which rows may be origins.
    """
    rng = rng or np.random.default_rng(0)
    df = df.reset_index(drop=True)
    tf = target_frame(df)
    cum_wind, cum_rain = _cum(df, "wind_speed_10m"), _cum(df, "precipitation")

    station = df["station_id"].to_numpy()
    n = len(df)
    ok = df["pm25_lag0"].notna().to_numpy()
    if origin_mask is not None:
        ok = ok & origin_mask
    if origin_step > 1:
        ok = ok & ((df["time"].dt.hour.to_numpy() % origin_step) == 0)
    origins = np.flatnonzero(ok)

    if horizons_per_origin is None:
        o_idx = np.repeat(origins, max_h)
        h = np.tile(np.arange(1, max_h + 1), len(origins))
    else:
        o_idx = np.repeat(origins, horizons_per_origin)
        h = rng.integers(1, max_h + 1, size=len(o_idx))
    t_idx = o_idx + h
    valid = t_idx < n
    o_idx, h, t_idx = o_idx[valid], h[valid], t_idx[valid]
    # Target must belong to the same station (series are stored back to back).
    same = station[o_idx] == station[t_idx]
    o_idx, h, t_idx = o_idx[same], h[same], t_idx[same]

    X = df.loc[o_idx, ORIGIN_COLS].reset_index(drop=True)
    T = tf.iloc[t_idx].reset_index(drop=True)
    X = pd.concat([X, T], axis=1)
    X["horizon"] = h
    X["t_wind_mean_h"] = (cum_wind[t_idx] - cum_wind[o_idx]) / h
    X["t_rain_sum_h"] = cum_rain[t_idx] - cum_rain[o_idx]
    X = X[ALL_COLS]

    y = df["pm25"].to_numpy()[t_idx]
    meta = pd.DataFrame({
        "station_id": station[o_idx],
        "origin_time": df["time"].to_numpy()[o_idx],
        "target_time": df["time"].to_numpy()[t_idx],
        "horizon": h,
        "pm25_origin": df["pm25"].to_numpy()[o_idx],
    })
    keep = ~np.isnan(y) if horizons_per_origin is not None else np.ones(len(y), bool)
    return X[keep].reset_index(drop=True), pd.Series(y[keep], name="pm25"), meta[keep].reset_index(drop=True)


def make_inference_rows(history: pd.DataFrame, future_weather: pd.DataFrame,
                        future_context: pd.DataFrame, max_h: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows for a live forecast.

    history: hourly table (with PM and weather) up to the latest observed hour.
    future_weather: station_id, time, WEATHER_COLS for the next max_h hours.
    future_context: date-level event flags for the future dates (date + EVENT_COLS).
    """
    hist = add_origin_features(history)
    last = hist[hist["pm25_lag0"].notna()].groupby("station_id").tail(1)

    fw = future_weather.copy()
    fw["date"] = fw["time"].dt.normalize()
    fw = fw.merge(future_context, on="date", how="left").drop(columns="date")
    station_cols = ["station_id", "name", "lat", "lon"]
    rows = []
    for _, o in last.iterrows():
        times = pd.date_range(o["time"] + pd.Timedelta(hours=1), periods=max_h, freq="h")
        fut = pd.DataFrame({"station_id": o["station_id"], "time": times}).merge(
            fw[fw["station_id"] == o["station_id"]], on=["station_id", "time"], how="left")
        # Combine the station's recent history with the future hours so rolling
        # windows (rain over 6 h) and running averages see both.
        recent = hist[(hist["station_id"] == o["station_id"]) & (hist["time"] <= o["time"])].tail(6)
        seq = pd.concat([recent[["station_id", "time"] + WEATHER_COLS + EVENT_COLS], fut], ignore_index=True)
        for c in EVENT_COLS:
            seq[c] = seq[c].fillna(0)
        tf = target_frame(seq).iloc[len(recent):].reset_index(drop=True)
        ws = fut["wind_speed_10m"].fillna(o["wind_speed_10m"]).to_numpy()
        rain = fut["precipitation"].fillna(0).to_numpy()
        h = np.arange(1, max_h + 1)
        X = pd.DataFrame([o[ORIGIN_COLS].to_numpy()] * max_h, columns=ORIGIN_COLS).astype(float)
        X = pd.concat([X, tf], axis=1)
        X["horizon"] = h
        X["t_wind_mean_h"] = np.cumsum(ws) / h
        X["t_rain_sum_h"] = np.cumsum(rain)
        meta = pd.DataFrame({c: o[c] for c in station_cols}, index=range(max_h))
        meta["origin_time"] = o["time"]
        meta["target_time"] = times
        meta["horizon"] = h
        meta["pm25_origin"] = o["pm25"]
        rows.append((X[ALL_COLS], meta))
    if not rows:
        return pd.DataFrame(columns=ALL_COLS), pd.DataFrame()
    return (pd.concat([r[0] for r in rows], ignore_index=True),
            pd.concat([r[1] for r in rows], ignore_index=True))

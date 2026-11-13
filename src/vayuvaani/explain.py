"""Explain a forecast in plain language.

For one forecast hour this gives:
- drivers: how much each group of inputs (wind, rain, time of day, recent
  pollution, festivals...) pushed the forecast up or down. With LightGBM these
  are exact SHAP values (TreeSHAP, built into LightGBM). With the scikit-learn
  fallback they are estimated by resetting each group to its typical value.
- what_if: the forecast re-run with one thing changed (stronger wind, rain,
  no festival), so people can see what would make the air cleaner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import QuantileForecaster

GROUPS = [
    ("Recent pollution levels", ("pm25_", "pm10_", "pm25_pm10")),
    ("Wind", ("t_wind_", "o_wind_")),
    ("Rain", ("t_rain", "o_rain")),
    ("Time of day", ("t_hour_",)),
    ("Weekday or weekend", ("t_dow", "t_weekend")),
    ("Season", ("t_doy_",)),
    ("Temperature and humidity", ("t_temp", "t_rh", "o_temp", "o_rh", "t_pressure", "t_cloud")),
    ("Fires in the region", ("fires_", "frp_")),
    ("Festivals and events", ("t_event_",)),
    ("Station location", ("lat", "lon")),
    ("How far ahead", ("horizon",)),
]

DESCRIBE = {
    "Recent pollution levels": ("The air has been polluted over the last day", "The air has been fairly clean over the last day"),
    "Wind": ("Light wind lets pollution build up", "Wind is spreading pollution out"),
    "Rain": ("No rain to wash pollution out", "Rain is washing pollution out"),
    "Time of day": ("Busy traffic hours and a low mixing layer", "Afternoon air mixes and dilutes pollution"),
    "Weekday or weekend": ("Weekday traffic", "Lighter weekend traffic"),
    "Season": ("Dry-season conditions", "Monsoon-season conditions"),
    "Temperature and humidity": ("Cool, still or humid air traps pollution", "Warm, mixed air helps pollution disperse"),
    "Fires in the region": ("Garbage or crop fires nearby", "Few fires nearby"),
    "Festivals and events": ("Festival or fireworks period", "No festival effect"),
    "Station location": ("This area is usually more polluted", "This area is usually cleaner"),
    "How far ahead": ("Forecast uncertainty further ahead", "Forecast uncertainty further ahead"),
}


def group_of(feature: str) -> str:
    for name, prefixes in GROUPS:
        if feature.startswith(prefixes) or feature in prefixes:
            return name
    return "Other"


def _contributions(fc: QuantileForecaster, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contributions to the median forecast, in log(1 + PM2.5) units."""
    X = X[fc.features].astype(float)
    m = fc.models[0.5]
    if fc.backend == "lightgbm":
        contrib = m.predict(X, pred_contrib=True)[:, :-1]
        return pd.DataFrame(contrib, columns=fc.features, index=X.index)
    # Fallback: reset each group to its typical value and see how the forecast moves.
    ref = fc.reference or X.median().to_dict()
    base = m.predict(X)
    out = pd.DataFrame(0.0, columns=fc.features, index=X.index)
    for name, _ in GROUPS:
        cols = [f for f in fc.features if group_of(f) == name]
        if not cols:
            continue
        Xr = X.copy()
        for c in cols:
            Xr[c] = ref.get(c, X[c].median())
        out[cols[0]] = base - m.predict(Xr)
    return out


def drivers(fc: QuantileForecaster, row: pd.DataFrame) -> list[dict]:
    """Grouped drivers for one row, largest effect first."""
    contrib = _contributions(fc, row).iloc[0]
    grouped = contrib.groupby([group_of(f) for f in contrib.index]).sum()
    out = []
    relative = getattr(fc, "relative", False)
    for name, value in grouped.items():
        if name in ("How far ahead", "Other") or abs(value) < 0.02:
            continue
        pct = (np.exp(value) - 1) * 100
        up, down = DESCRIBE.get(name, (name, name))
        if relative and name == "Recent pollution levels":
            # Relative models start from the current reading, so this factor is about the trend.
            up, down = "Recent readings point to a rise", "Recent readings point to it easing off"
        out.append({"factor": name, "effect_pct": round(float(pct)), "direction": "up" if value > 0 else "down",
                    "text": up if value > 0 else down})
    out.sort(key=lambda d: abs(d["effect_pct"]), reverse=True)
    return out[:6]


def what_if(fc: QuantileForecaster, row: pd.DataFrame) -> list[dict]:
    """Re-run the forecast for one row with simple changes."""
    base = float(fc.predict(row)["pm25_q50"].iloc[0])
    scenarios = []

    def run(label: str, changes: dict):
        r = row.copy()
        for col, fn in changes.items():
            if col in r:
                r[col] = fn(r[col])
        new = float(fc.predict(r)["pm25_q50"].iloc[0])
        scenarios.append({"scenario": label, "pm25": round(new, 1), "change_pct": round((new / base - 1) * 100) if base else 0})

    stronger = lambda v: v * 1.6
    run("Wind 60% stronger", {"t_wind_speed": stronger, "t_wind_mean_h": stronger,
                              "t_wind_u": stronger, "t_wind_v": stronger})
    run("Calm wind", {"t_wind_speed": lambda v: v * 0.4, "t_wind_mean_h": lambda v: v * 0.4,
                      "t_wind_u": lambda v: v * 0.4, "t_wind_v": lambda v: v * 0.4})
    run("Steady rain (5 mm)", {"t_rain": lambda v: v * 0 + 1.0, "t_rain6": lambda v: v * 0 + 5.0,
                               "t_rain_sum_h": lambda v: v + 5.0})
    if float(row.get("t_event_fireworks", pd.Series([0])).iloc[0]) > 0:
        run("No fireworks", {"t_event_fireworks": lambda v: v * 0})
    else:
        run("If it were a fireworks night", {"t_event_fireworks": lambda v: v * 0 + 1})
    return [{"scenario": "As forecast", "pm25": round(base, 1), "change_pct": 0}] + scenarios


def explain_station(fc: QuantileForecaster, features: pd.DataFrame, station_id: str,
                    target_time: str | None = None, tz: str | None = None) -> dict:
    """Explain the worst hour in the next 24 h (or a chosen hour) for one station."""
    f = features[features["station_id"] == station_id].reset_index(drop=True)
    if f.empty:
        return {}
    X = f[fc.features].astype(float)
    times = pd.to_datetime(f["target_time"])
    if target_time:
        idx = int((times == pd.Timestamp(target_time)).idxmax())
    else:
        # Worst of the next 24 hours from now (readings arrive a little late).
        now = pd.Timestamp.now(tz=tz).tz_localize(None).floor("h") if tz else times.min()
        upcoming = [i for i in X.index if times[i] >= now - pd.Timedelta(hours=1)][:24] or list(X.index[:24])
        med = fc.predict(X.loc[upcoming])["pm25_q50"]
        idx = int(med.idxmax())
    row = X.iloc[[idx]]
    return {
        "station_id": station_id,
        "target_time": pd.Timestamp(f.loc[idx, "target_time"]).strftime("%Y-%m-%dT%H:%M"),
        "method": "SHAP (LightGBM TreeSHAP)" if fc.backend == "lightgbm" else "Estimated (reset each factor to typical)",
        "baseline": "the current reading" if getattr(fc, "relative", False) else "a typical hour",
        "drivers": drivers(fc, row),
        "what_if": what_if(fc, row),
    }

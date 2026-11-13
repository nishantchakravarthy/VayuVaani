"""Indian National AQI (CPCB) sub-indices and categories.

The official AQI is the maximum sub-index over at least three pollutants
(one of them PM2.5 or PM10), each averaged over 24 hours. VayuVaani forecasts
PM2.5, so the AQI it reports is the PM2.5 sub-index. Label it as such in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# AQI value at each category edge.
AQI_EDGES = [0, 50, 100, 200, 300, 400, 500]
# Matching concentration edges in µg/m³ (24-hour average), from the CPCB NAQI table.
PM25_EDGES = [0, 30, 60, 90, 120, 250, 380]
PM10_EDGES = [0, 50, 100, 250, 350, 430, 510]


def _sub_index(conc, edges):
    c = np.asarray(conc, dtype=float)
    out = np.interp(np.clip(c, 0, None), edges, AQI_EDGES, right=500.0)
    out = np.where(np.isnan(c), np.nan, np.round(out))
    return float(out) if out.ndim == 0 else out


def pm25_sub_index(conc):
    """PM2.5 sub-index (0–500) for a scalar or array of µg/m³ values."""
    return _sub_index(conc, PM25_EDGES)


def pm10_sub_index(conc):
    return _sub_index(conc, PM10_EDGES)


def category(aqi):
    """Category name for one AQI value."""
    if aqi is None or np.isnan(aqi):
        return None
    for name, hi in zip(CATEGORIES, AQI_EDGES[1:]):
        if aqi <= hi:
            return name
    return CATEGORIES[-1]


def categories(aqi_values) -> pd.Series:
    return pd.Series([category(a) for a in np.asarray(aqi_values, float)])


def pm25_category(conc):
    return category(pm25_sub_index(conc))


def pm25_threshold(category_name: str) -> float:
    """Lowest PM2.5 concentration (µg/m³) that falls in the given category."""
    i = CATEGORIES.index(category_name)
    return float(PM25_EDGES[i])

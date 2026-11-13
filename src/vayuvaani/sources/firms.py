"""NASA FIRMS active-fire detections around Bengaluru.

Fire counts stand in for garbage and biomass burning in the region.
API: /api/area/csv/[MAP_KEY]/[SOURCE]/[west,south,east,north]/[DAY_RANGE]/[DATE]
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from ..config import Settings, secret
from .http import get

log = logging.getLogger(__name__)

CHUNK_DAYS = 5  # small chunks stay within the API's day-range limit


def download(settings: Settings, start: str, end: str, source: str | None = None) -> pd.DataFrame:
    """Fire detections between start and end dates (inclusive)."""
    cfg = settings.sources["firms"]
    key = secret("FIRMS_MAP_KEY")
    area = ",".join(str(v) for v in cfg["bbox"])
    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    # Standard-processing data lags a few months; use near-real-time for recent dates.
    recent_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=90)
    frames = []
    day = start_d
    while day <= end_d:
        n = min(CHUNK_DAYS, (end_d - day).days + 1)
        src = source or (cfg["source_history"] if day < recent_cutoff else cfg["source_recent"])
        url = f"{cfg['base_url']}/{key}/{src}/{area}/{n}/{day.date()}"
        text = get(url).text
        if text.strip() and not text.lower().startswith("invalid"):
            df = pd.read_csv(io.StringIO(text))
            if not df.empty:
                frames.append(df)
        day += pd.Timedelta(days=n)
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"])
    fires = pd.concat(frames, ignore_index=True)
    log.info("Downloaded %d fire detections", len(fires))
    return fires


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def daily_counts(settings: Settings, fires: pd.DataFrame) -> pd.DataFrame:
    """Daily fire count and total fire radiative power within radius_km of the city centre."""
    cfg = settings.sources["firms"]
    lat0, lon0 = settings.city["centre"]
    if fires.empty:
        return pd.DataFrame(columns=["date", "fire_count", "fire_frp"])
    dist = haversine_km(lat0, lon0, fires["latitude"].astype(float), fires["longitude"].astype(float))
    near = fires[dist <= cfg["radius_km"]].copy()
    near["date"] = pd.to_datetime(near["acq_date"]).dt.normalize()
    near["frp"] = pd.to_numeric(near.get("frp", 0), errors="coerce").fillna(0)
    out = near.groupby("date").agg(fire_count=("frp", "size"), fire_frp=("frp", "sum")).reset_index()
    return out

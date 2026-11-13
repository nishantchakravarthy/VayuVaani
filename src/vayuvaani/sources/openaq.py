"""OpenAQ v3 client: find stations in the city and download hourly PM2.5/PM10.

OpenAQ carries the CPCB CAAQMS stations, so this is the main historical source.
Docs: https://docs.openaq.org
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import Settings, secret
from .http import get

log = logging.getLogger(__name__)

PAGE_LIMIT = 1000


def _headers() -> dict:
    return {"X-API-Key": secret("OPENAQ_API_KEY")}


def list_stations(settings: Settings) -> pd.DataFrame:
    """Stations inside the city bounding box that report PM2.5."""
    cfg = settings.sources["openaq"]
    bbox = ",".join(str(v) for v in settings.city["bbox"])
    rows, page = [], 1
    while True:
        resp = get(
            f"{cfg['base_url']}/locations",
            params={"bbox": bbox, "parameters_id": cfg["pm25_parameter_id"], "limit": PAGE_LIMIT, "page": page},
            headers=_headers(),
        ).json()
        results = resp.get("results", [])
        for loc in results:
            sensors = {s["parameter"]["name"]: s["id"] for s in loc.get("sensors", [])}
            if "pm25" not in sensors:
                continue
            provider = (loc.get("provider") or {}).get("name", "")
            rows.append({
                "station_id": f"openaq_{loc['id']}",
                "name": loc.get("name") or f"OpenAQ {loc['id']}",
                "lat": loc["coordinates"]["latitude"],
                "lon": loc["coordinates"]["longitude"],
                "provider": provider,
                "is_monitor": bool(loc.get("isMonitor", True)),
                "pm25_sensor_id": sensors.get("pm25"),
                "pm10_sensor_id": sensors.get("pm10"),
            })
        if len(results) < PAGE_LIMIT:
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("OpenAQ returned no PM2.5 stations in the bounding box.")
        return df
    # Prefer reference-grade monitors (CPCB/KSPCB) over low-cost sensors.
    df = df[df["is_monitor"]].reset_index(drop=True)
    log.info("Found %d PM2.5 monitoring stations", len(df))
    return df


def sensor_hours(settings: Settings, sensor_id: int, start: str, end: str) -> pd.DataFrame:
    """Hourly mean values for one sensor between start and end (ISO dates)."""
    base = settings.sources["openaq"]["base_url"]
    rows, page = [], 1
    while True:
        resp = get(
            f"{base}/sensors/{sensor_id}/hours",
            params={"datetime_from": start, "datetime_to": end, "limit": PAGE_LIMIT, "page": page},
            headers=_headers(),
        ).json()
        results = resp.get("results", [])
        for r in results:
            rows.append({
                "time_utc": r["period"]["datetimeFrom"]["utc"],
                "value": r.get("value"),
            })
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(0.2)  # stay well under the rate limit
    df = pd.DataFrame(rows, columns=["time_utc", "value"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def download(settings: Settings, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stations, measurements). Measurements are long format:
    station_id, time (local, tz-naive), parameter, value."""
    stations = list_stations(settings)
    frames = []
    for _, st in stations.iterrows():
        for param in ("pm25", "pm10"):
            sid = st[f"{param}_sensor_id"]
            if pd.isna(sid):
                continue
            log.info("Downloading %s for %s", param, st["name"])
            df = sensor_hours(settings, int(sid), start, end)
            if df.empty:
                continue
            df["station_id"] = st["station_id"]
            df["parameter"] = param
            frames.append(df)
    if not frames:
        return stations, pd.DataFrame(columns=["station_id", "time", "parameter", "value"])
    meas = pd.concat(frames, ignore_index=True)
    meas["time"] = meas["time_utc"].dt.tz_convert(settings.timezone).dt.tz_localize(None)
    meas = meas.drop(columns="time_utc")
    return stations, meas
